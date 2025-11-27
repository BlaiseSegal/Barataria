# sanson_agent.py
# -----------------------------------------------------------------------------
# Sansón  - (Academic Analyst, deterministic)
#
# Role:
#   - Reads Knowledge Assets from the DKG (revenues, subsidies, correlations)
#   - Computes SR_G, CF_G, IS_G (if needed), and then σ(G) according to the Barataria Manifesto
#   - Publishes a new integrity KA: ka://analytics/integrity/<guild>/v<k>
#
# Features:
#   - 100% deterministic (no LLM in the loop)
#   - Traceable: reasoning_digest SHA-256
#   - DKG-centric: all facts come from the DKG (or controlled mocks)
#
# -----------------------------------------------------------------------------

import asyncio
import json
import time
import hashlib
from typing import TypedDict, Dict, Any, Optional, List, Literal

import os
import sys

# Allow running as a script: add repo root to sys.path so `core` imports work.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.dkg_config import get_dkg_client

DEFAULT_JSONLD_CONTEXT = "https://schema.barataria.arch/v1"

# ---------------------------------------------------------------------------
# 1. Types & Configuration
# ---------------------------------------------------------------------------

HealthStatus = Literal["OK", "DKG_UNREACHABLE", "SAFE_MODE", "ERROR"]

class DKGConfig(TypedDict):
    endpoint: str
    repository: str
    use_ssl: bool

class SansonState(TypedDict, total=False):
    # Inputs
    target_guild_id: str              

    # Raw DKG responses
    raw_yield_ka: Dict[str, Any]
    raw_cartel_ka: Dict[str, Any]
    raw_prev_integrity_ka: Optional[Dict[str, Any]]

    # Parsed facts
    financials: Dict[str, float]      # {E_internes, T_corrélés, F_nets}
    network_stats: Dict[str, float]   # {rho, omega}
    base_integrity: float             # IS_G default (0..1)
    subsidy_ratio: float              # SR_G
    cartel_flag: bool                 # CF_G (True/False)
    sigma: float                      # σ(G) final in [0,1]

    # Reasoning / trace
    reasoning_trace: List[str]
    reasoning_digest: str

    # Health
    health_status: HealthStatus

    # Publication
    new_integrity_ka: Dict[str, Any]
    published: bool


# We can either reuse the global DKG_CONFIG from core.dkg_config,
# or define a fallback here if it is missing.
try:
    from core.dkg_config import DKG_CONFIG as _CORE_DKG_CONFIG  # type: ignore
    DKG_CONFIG: DKGConfig = {
        "endpoint": os.getenv("OT_NODE_URL", getattr(_CORE_DKG_CONFIG, "endpoint", "http://localhost:8900")),  # type: ignore
        "repository": getattr(_CORE_DKG_CONFIG, "repository", "barataria_repository"),  # type: ignore
        "use_ssl": getattr(_CORE_DKG_CONFIG, "use_ssl", False),  # type: ignore
    }
except Exception:
    DKG_CONFIG: DKGConfig = {
        "endpoint": os.getenv("OT_NODE_URL", "http://localhost:8900"),
        "repository": "barataria_repository",
        "use_ssl": False,
    }


# ---------------------------------------------------------------------------
# 2. DKG Client Wrapper (Sansón)
# ---------------------------------------------------------------------------

class SansonDKGClient:
    """
    Minimal DKG client for Sansón.
    - Retrieves the necessary KAs:
      * Yields (revenues / subsidies)
      * Cartel index (rho, omega)
      * Last integrity KA (optional)
    """

    def __init__(self, config: DKGConfig):
        self.config = config
        # Allow forcing mock mode for demos or when the node has no data yet.
        force_mock = os.getenv("SANSON_FORCE_MOCK", "").lower() in ("1", "true", "yes")
        if force_mock:
            self.client = None
            self.mock_mode = True
            return

        try:
            self.client = get_dkg_client()
            self.mock_mode = False
        except Exception:
            self.client = None
            self.mock_mode = True

    async def query_sparql(self, query: str) -> Dict[str, Any]:
        """
        Exécute SPARQL sur le DKG (ou mock contrôlé).
        """
        if self.mock_mode:
            await asyncio.sleep(0.2)
            # MOCK : on renvoie des structures stables pour tests unitaires.
            if "barataria:GuildYield" in query:
                
                return {
                    "@context": DEFAULT_JSONLD_CONTEXT,
                    "results": {
                        "bindings": [
                            {
                                "internalEmissions": {"type": "literal", "value": "5000"},
                                "correlatedTransfers": {"type": "literal", "value": "4000"},
                                "netFeesEarned": {"type": "literal", "value": "1000"},
                                "sourceKA": {"type": "uri", "value": "ka://guild/0xABC/yield/v5"},
                            }
                        ]
                    },
                }
            elif "barataria:CartelIndex" in query:
                # KA de corrélations : ka://analytics/cartel-index/vX
                return {
                    "@context": DEFAULT_JSONLD_CONTEXT,
                    "results": {
                        "bindings": [
                            {
                                "verdictCorrelation": {"type": "literal", "value": "0.95"},
                                "jurorOverlap": {"type": "literal", "value": "0.40"},
                                "sourceKA": {"type": "uri", "value": "ka://analytics/cartel-index/v12"},
                            }
                        ]
                    },
                }
            elif "barataria:IntegrityAnalysis" in query:
                # Dernier KA d'intégrité (optionnel)
                return {
                    "@context": DEFAULT_JSONLD_CONTEXT,
                    "results": {
                        "bindings": [
                            {
                                "integrityScore": {"type": "literal", "value": "0.85"},
                                "sigma": {"type": "literal", "value": "0.80"},
                                "version": {"type": "literal", "value": "5"},
                                "sourceKA": {"type": "uri", "value": "ka://analytics/integrity/0xABC/v5"},
                            }
                        ]
                    },
                }
            else:
                return {
                    "@context": DEFAULT_JSONLD_CONTEXT,
                    "results": {"bindings": []},
                }
        else:
            # Intégration réelle DKG SDK : supporte à la fois sync et async.
            res = self.client.graph.query(query)  # type: ignore
            if asyncio.iscoroutine(res):
                return await res
            return res

    @staticmethod
    def _parse_number(value: str) -> float:
        try:
            if "." in value or "e" in value.lower():
                return float(value)
            return float(int(value))
        except Exception:
            return float("nan")

    # --------------------- Parsing "Yields" KA ---------------------

    def parse_yield_ka(self, result: Dict[str, Any]) -> Dict[str, float]:
        """
        Extrait E_internes, T_corrélés, F_nets à partir du KA yield.
        """
        bindings = result if isinstance(result, list) else result.get("results", {}).get("bindings", [])
        if not bindings:
            raise ValueError("No yield KA bindings for guild.")
        b = bindings[-1]  # Last Write Wins

        def getf(name: str) -> float:
            if name not in b:
                raise KeyError(name)
            return self._parse_number(b[name]["value"])

        einternes = getf("internalEmissions")
        tcorreles = getf("correlatedTransfers")
        fnets = getf("netFeesEarned")
        return {
            "internal_emissions": einternes,
            "correlated_transfers": tcorreles,
            "net_fees_earned": fnets,
        }

    # --------------------- Parsing "CartelIndex" KA ---------------------

    def parse_cartel_ka(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extrait rho (verdictCorrelation), omega (jurorOverlap).
        """
        bindings = result if isinstance(result, list) else result.get("results", {}).get("bindings", [])
        if not bindings:
            raise ValueError("No cartel index KA bindings.")
        b = bindings[-1]

        def getf(name: str) -> float:
            if name not in b:
                raise KeyError(name)
            return self._parse_number(b[name]["value"])

        rho = getf("verdictCorrelation")
        omega = getf("jurorOverlap")
        src = b.get("sourceKA", {}).get("value", "ka://analytics/cartel-index/unknown")
        return {
            "verdict_correlation": rho,
            "juror_overlap": omega,
            "source_ka": src,
        }

    # --------------------- Parsing "Integrity" KA (previous) ---------------------

    def parse_prev_integrity_ka(self, result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Extrait éventuellement l'ancien integrityScore / sigma pour continuité.
        """
        bindings = result if isinstance(result, list) else result.get("results", {}).get("bindings", [])
        if not bindings:
            return None
        b = bindings[-1]

        def getf(name: str, default: str) -> float:
            if name not in b:
                return float(default)
            return self._parse_number(b[name]["value"])

        integrity = getf("integrityScore", "0.8")
        sigma = getf("sigma", "0.8")
        version = int(getf("version", "0"))
        src = b.get("sourceKA", {}).get("value", "ka://analytics/integrity/unknown")
        return {
            "integrity_score": integrity,
            "sigma": sigma,
            "version": version,
            "source_ka": src,
        }


dkg_client_sanson = SansonDKGClient(DKG_CONFIG)


# ---------------------------------------------------------------------------
# 3. Moteur Mathématique de Sansón (deterministic)
# ---------------------------------------------------------------------------

class SansonMathEngine:
    """
    Implémente les définitions du Manifeste Barataria pour:
      - SR_G (Subsidy Ratio)
      - CF_G (Cartel Flag)
      - σ(G) (coefficient d'assurance)
    """

    def compute_subsidy_ratio(self, financials: Dict[str, float]) -> float:
        """
        SR_G = (E_internes + T_corrélés) / F_nets
        Avec garde: si F_nets <= 0 -> SR_G = +inf (ici capé arbitrairement à 1e6).
        """
        e_int = financials["internal_emissions"]
        t_corr = financials["correlated_transfers"]
        f_nets = financials["net_fees_earned"]

        if f_nets <= 0:
            # cas pathologique : aucun revenu réel
            return 1e6

        sr = (e_int + t_corr) / f_nets
        return float(sr)

    def compute_cartel_flag(self, network_stats: Dict[str, float]) -> bool:
        """
        CF_G = 1{rho_G(verdicts) > 0.9 ∧ ω_G(jurés) > 0.3}
        """
        rho = network_stats["verdict_correlation"]
        omega = network_stats["juror_overlap"]
        return bool(rho > 0.9 and omega > 0.3)

    def compute_sigma(
        self,
        base_integrity: float,
        subsidy_ratio: float,
        cartel_flag: bool,
    ) -> float:
        """
        σ(G) = (1 - CF_G) * IS_G * (1 - SR_G_clamped)
        avec SR_G_clamped dans [0, 1] pour éviter les valeurs négatives évidentes.
        Si SR_G > 1 -> on considère que (1 - SR_G) est négatif => σ → 0.
        """
        if cartel_flag:
            return 0.0

        # On interprète SR > 1 comme "sur-subvention", pénalisée agressivement.
        sr_clamped = max(0.0, min(subsidy_ratio, 1.0))
        sigma = (1.0 - 0.0) * base_integrity * (1.0 - sr_clamped)

        # Bornage final [0, 1]
        return max(0.0, min(float(sigma), 1.0))


math_engine = SansonMathEngine()


# ---------------------------------------------------------------------------
# 4. Publication DKG (KA d'intégrité)
# ---------------------------------------------------------------------------

async def tool_publish_dkg(data_jsonld: Dict[str, Any]) -> bool:
    """
    Stub de publication DKG (même pattern que dans sancho_agent.py).
    À remplacer par un appel DKG SDK réel (asset.create).
    """
    print("[DKG PUBLISH] KA:", json.dumps(data_jsonld, indent=2))
    await asyncio.sleep(0.2)
    return True


def _compute_reasoning_digest(trace: List[str]) -> str:
    blob = "\n".join(trace).encode("utf-8")
    return "0x" + hashlib.sha256(blob).hexdigest()


# ---------------------------------------------------------------------------
# 5. Workflow Sansón (séquentiel, déterministe)
# ---------------------------------------------------------------------------

async def interrogate_dkg_for_guild(state: SansonState) -> SansonState:
    """
    Étape 1 : récupérer les KAs nécessaires pour une guilde donnée.
    """
    guild = state["target_guild_id"]
    trace = state.setdefault("reasoning_trace", [])
    trace.append(f"Sansón: interrogation du DKG pour la guilde {guild}.")

    # SPARQL pour GuildYield
    query_yield = f"""
    PREFIX barataria: <https://schema.barataria.arch/v1#>

    SELECT ?internalEmissions ?correlatedTransfers ?netFeesEarned ?sourceKA
    WHERE {{
      ?yield a barataria:GuildYield ;
             barataria:guildId "{guild}" ;
             barataria:internalEmissions ?internalEmissions ;
             barataria:correlatedTransfers ?correlatedTransfers ;
             barataria:netFeesEarned ?netFeesEarned ;
             barataria:generatedFromKA ?sourceKA .
    }}
    ORDER BY DESC(?sourceKA)
    LIMIT 1
    """

    # SPARQL pour CartelIndex
    query_cartel = f"""
    PREFIX barataria: <https://schema.barataria.arch/v1#>

    SELECT ?verdictCorrelation ?jurorOverlap ?sourceKA
    WHERE {{
      ?ci a barataria:CartelIndex ;
          barataria:guildId "{guild}" ;
          barataria:verdictCorrelation ?verdictCorrelation ;
          barataria:jurorOverlap ?jurorOverlap ;
          barataria:generatedFromKA ?sourceKA .
    }}
    ORDER BY DESC(?sourceKA)
    LIMIT 1
    """

    # SPARQL pour dernier IntegrityAnalysis (optionnel)
    query_prev_integrity = f"""
    PREFIX barataria: <https://schema.barataria.arch/v1#>

    SELECT ?integrityScore ?sigma ?version ?sourceKA
    WHERE {{
      ?ia a barataria:IntegrityAnalysis ;
          barataria:guildId "{guild}" ;
          barataria:integrityScore ?integrityScore ;
          barataria:hasSigma ?sigma ;
          barataria:version ?version ;
          barataria:generatedFromKA ?sourceKA .
    }}
    ORDER BY DESC(?version)
    LIMIT 1
    """

    try:
        raw_yield = await dkg_client_sanson.query_sparql(query_yield)
        raw_cartel = await dkg_client_sanson.query_sparql(query_cartel)
        raw_prev_int = await dkg_client_sanson.query_sparql(query_prev_integrity)

        state["raw_yield_ka"] = raw_yield
        state["raw_cartel_ka"] = raw_cartel
        state["raw_prev_integrity_ka"] = raw_prev_int
        state["health_status"] = "OK"
        trace.append("Sansón: KAs récupérés avec succès.")
        return state
    except Exception as e:
        state["health_status"] = "DKG_UNREACHABLE"
        trace.append(f"Sansón: erreur critique DKG → {str(e)}.")
        return state


async def compute_metrics(state: SansonState) -> SansonState:
    """
    Étape 2 : applique les formules (SR, CF, σ) de manière purement déterministe.
    """
    trace = state.setdefault("reasoning_trace", [])

    if state.get("health_status") != "OK":
        trace.append("Sansón: compute_metrics annulé (DKG non disponible).")
        return state

    # 1. Parsing
    financials = dkg_client_sanson.parse_yield_ka(state["raw_yield_ka"])
    cartel = dkg_client_sanson.parse_cartel_ka(state["raw_cartel_ka"])
    prev_int = dkg_client_sanson.parse_prev_integrity_ka(state["raw_prev_integrity_ka"])

    state["financials"] = financials
    state["network_stats"] = {
        "verdict_correlation": cartel["verdict_correlation"],
        "juror_overlap": cartel["juror_overlap"],
    }

    # Base integrity : soit issu du KA précédent, soit défaut (0.8)
    base_integrity = prev_int["integrity_score"] if prev_int else 0.8
    state["base_integrity"] = base_integrity

    trace.append(
        f"Sansón: financials = {financials}, base_integrity = {base_integrity:.4f}, "
        f"rho = {cartel['verdict_correlation']:.4f}, omega = {cartel['juror_overlap']:.4f}."
    )

    # 2. SR_G
    sr_g = math_engine.compute_subsidy_ratio(financials)
    state["subsidy_ratio"] = sr_g
    trace.append(f"Sansón: SR_G = {sr_g:.4f}.")

    # 3. CF_G
    cf_g = math_engine.compute_cartel_flag(state["network_stats"])
    state["cartel_flag"] = cf_g
    trace.append(f"Sansón: CF_G = {cf_g}.")

    # 4. σ(G)
    sigma = math_engine.compute_sigma(
        base_integrity=base_integrity,
        subsidy_ratio=sr_g,
        cartel_flag=cf_g,
    )
    state["sigma"] = sigma
    trace.append(f"Sansón: σ(G) = {sigma:.4f}.")

    # 5. digest
    digest = _compute_reasoning_digest(trace)
    state["reasoning_digest"] = digest
    trace.append(f"Sansón: reasoning_digest = {digest}.")

    return state


async def publish_integrity_ka(state: SansonState) -> SansonState:
    """
    Étape 3 : publie un nouveau KA d'intégrité sur le DKG.
    """
    trace = state.setdefault("reasoning_trace", [])

    if state.get("health_status") != "OK":
        trace.append("Sansón: publication annulée (état non OK).")
        state["published"] = False
        return state

    guild = state["target_guild_id"]
    now_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    sr = state["subsidy_ratio"]
    cf = state["cartel_flag"]
    sigma = state["sigma"]
    base_integrity = state["base_integrity"]
    digest = state["reasoning_digest"]

    # On récupère éventuellement les KAs sources pour la provenance
    yield_src = state["raw_yield_ka"].get("results", {}).get("bindings", [{}])[-1].get(
        "sourceKA", {}
    ).get("value", "ka://guild/unknown/yield")
    cartel_src = state["raw_cartel_ka"].get("results", {}).get("bindings", [{}])[-1].get(
        "sourceKA", {}
    ).get("value", "ka://analytics/cartel-index/unknown")

    prev_integrity = dkg_client_sanson.parse_prev_integrity_ka(
        state.get("raw_prev_integrity_ka", {})
    )
    prev_version = prev_integrity["version"] if prev_integrity else 0
    new_version = prev_version + 1

    integrity_ka = {
        "@context": DEFAULT_JSONLD_CONTEXT,
        "@type": "BaratariaIntegrityAnalysis",
        "guildId": guild,
        "version": new_version,
        "generatedAtTime": now_iso,
        "integrityScore": round(base_integrity, 4),
        "subsidyRatio": round(sr, 4),
        "cartelFlag": cf,
        "sigma": round(sigma, 4),
        "reasoningDigest": digest,
        "prov:wasDerivedFrom": [
            {"@id": yield_src},
            {"@id": cartel_src},
        ],
    }

    ok = await tool_publish_dkg(integrity_ka)
    state["new_integrity_ka"] = integrity_ka
    state["published"] = ok
    if ok:
        trace.append(
            f"Sansón: Integrity KA publié pour {guild}, version={new_version}, σ={sigma:.4f}."
        )
    else:
        trace.append("Sansón: échec de publication du Integrity KA.")
        state["health_status"] = "ERROR"

    return state


async def run_sanson_once(target_guild_id: str) -> SansonState:
    """
    Pipeline complet de Sansón (deterministic):
      1) Interroger DKG
      2) Calculer SR, CF, σ
      3) Publier KA d'intégrité
    """
    state: SansonState = {
        "target_guild_id": target_guild_id,
        "reasoning_trace": [],
        "health_status": "OK",
    }

    state = await interrogate_dkg_for_guild(state)
    state = await compute_metrics(state)
    state = await publish_integrity_ka(state)
    return state


# ---------------------------------------------------------------------------
# 6. Exemple d'exécution autonome
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    async def _demo():
        guild = "guild:don_quichotte_inc"
        final_state = await run_sanson_once(guild)
        print("### Sansón Final State ###")
        print(json.dumps(final_state, indent=2, default=str))

    asyncio.run(_demo())
