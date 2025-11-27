# sancho_agent.py
# -----------------------------------------------------------------------------
# Sancho  - (Full-Prod Polkadot / NeuroWeb)
#
# Architecture:
#   - DKGConnector: SPARQL, JSON-LD, parsing, rules + reputation fetch
#   - SymbolicReasoner: Neuro-symbolic, BPS, dynamic rules
#   - XCMBuilder + SubstrateConnector: XCM V3/V4, Transact + SCALE call
#   - ResilienceLayer: timeouts, circuit-breaker, health, safe-mode
#
# NOTE: Certain parts (DKG SDK, SCALE encoder, Substrate API) are
#       defined as adapters/hooks, to be implemented behind
#       clear interfaces in the actual infrastructure.
# -----------------------------------------------------------------------------

import asyncio
import time
import json
import hashlib
import os
import sys
from typing import (
    TypedDict,
    Literal,
    Optional,
    List,
    Dict,
    Any,
    Callable,
)

# Allow running as a script: add repo root to sys.path so `core` imports work.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.dkg_config import get_dkg_client
from core.xcm_engine import tool_execute_xcm_payment_x402 as real_xcm_tool

# ---------------------------------------------------------------------------
# 1. Types & Configuration
# ---------------------------------------------------------------------------

HealthStatus = Literal["OK", "DEGRADED", "DKG_UNREACHABLE", "XCM_FAILED", "SAFE_MODE"]
DecisionType = Literal["REJECT", "PAY", "COUNTER_OFFER", "ERROR"]
ExecutionMode = Literal["TRANSFER_ONLY", "CONTRACT_CALL"]

DEFAULT_JSONLD_CONTEXT = "https://schema.barataria.arch/v1"

class DKGConfig(TypedDict):
    endpoint: str
    repository: str
    use_ssl: bool

class NetworkConfig(TypedDict):
    relay_chain: str
    parachain_id: int
    asset_multilocation: Dict[str, Any]
    beneficiary_account: str  # hex-encoded AccountId32
    guild_tcr_address: str    # contract address on target chain

class AgentSanchoState(TypedDict, total=False):
    # Input strict
    input_task: Dict[str, Any]              # {service_id, amount, mode, ...}

    # DKG (symbolic)
    dkg_query_reputation: str
    dkg_query_rules: str
    dkg_facts: Optional[Dict[str, Any]]
    rules_dkg: Optional[Dict[str, Any]]
    raw_dkg_response_reputation: Optional[Dict[str, Any]]
    raw_dkg_response_rules: Optional[Dict[str, Any]]
    dkg_query_latency: float

    # Reasoning (neuro-symbolic)
    final_decision: DecisionType
    calculated_amount: float
    logical_decision: str
    full_reasoning_trace: List[str]

    # XCM / Trust Layer
    multilocation_destination: Dict[str, Any]
    xcm_call_payload: Dict[str, Any]
    asset_id: str
    tx_hash: Optional[str]
    execution_mode: ExecutionMode

    # Health / Resilience
    health_status: HealthStatus
    retry_counter_dkg: int
    retry_counter_xcm: int
    last_success_timestamp: Optional[float]
    consecutive_failures: int
    safe_mode_until: Optional[float]

    # Audit
    reasoning_digest: Optional[str]


# DKG configuration (local fallback for SPARQL query builder)
DKG_CONFIG: DKGConfig = {
    "endpoint": os.getenv("OT_NODE_URL", "http://localhost:8900"),
    "repository": "barataria_repository",
    "use_ssl": False,
}

# Network configuration / XCM (NeuroWeb)
NETWORK_CONFIG: NetworkConfig = {
    "relay_chain": "polkadot",
    "parachain_id": 2043,  # NeuroWeb
    "asset_multilocation": {
        "parents": 1,
        "interior": {"X1": {"Parachain": 2043}}
    },
    "beneficiary_account": "0xRecipientAddressHere0000000000000000000000000000",
    "guild_tcr_address": "0xGuildTcrContractAddress000000000000000000000000",
}

# Local fallback (if KA rules not found)
FALLBACK_RULES: Dict[str, Any] = {
    "sigma_min_bps": 8000,      # 0.80
    "max_subsidy_ratio": 0.30,  # 30%
    "reject_if_cartel": True,
    "circuit_breaker_threshold": 5,   # ERRORs consecutive
    "safe_mode_cooldown_sec": 600,    # 10 minutes
}

GLOBAL_TIMEOUT_SEC = 15.0  # global timeout for a Sancho execution


# ---------------------------------------------------------------------------
# 2. DKG Connector (SPARQL + JSON-LD + Parsing)
# ---------------------------------------------------------------------------

class DKGClientWrapper:
    """
    Wrapper around the DKG SDK (OriginTrail).
    In production: connect kdz.Client() or equivalent.
    """
    def __init__(self, config: DKGConfig):
        self.config = config
        self.mock_mode = True  # set to False in production

    async def query_sparql(self, query: str) -> Dict[str, Any]:
        """
        Executes SPARQL and returns JSON of type SPARQL results.
        """
        if self.mock_mode:
            await asyncio.sleep(0.3)
            # MOCK: we simulate a single consistent binding
            if "barataria:IntegrityAnalysis" in query:
                return {
                    "@context": DEFAULT_JSONLD_CONTEXT,
                    "head": {
                        "vars": [
                            "sigma",
                            "cartelFlag",
                            "integrityScore",
                            "subsidyRatio",
                            "sourceKA",
                            "validUntil",
                            "version",
                        ]
                    },
                    "results": {
                        "bindings": [
                            {
                                "sigma": {"type": "literal", "value": "0.92"},
                                "cartelFlag": {"type": "literal", "value": "false"},
                                "integrityScore": {"type": "literal", "value": "0.95"},
                                "subsidyRatio": {"type": "literal", "value": "0.10"},
                                "sourceKA": {"type": "uri", "value": "ka://guild/12345"},
                                "validUntil": {"type": "literal", "value": "4102444800"},  # 2100-01-01
                                "version": {"type": "literal", "value": "2"},
                            }
                        ]
                    },
                }
            elif "barataria:Ruleset" in query:
                return {
                    "@context": DEFAULT_JSONLD_CONTEXT,
                    "head": {
                        "vars": [
                            "sigmaMinBps",
                            "maxSubsidyRatio",
                            "rejectIfCartel",
                            "circuitBreakerThreshold",
                            "safeModeCooldownSec",
                        ]
                    },
                    "results": {
                        "bindings": [
                            {
                                "sigmaMinBps": {"type": "literal", "value": "8000"},
                                "maxSubsidyRatio": {"type": "literal", "value": "0.3"},
                                "rejectIfCartel": {"type": "literal", "value": "true"},
                                "circuitBreakerThreshold": {"type": "literal", "value": "5"},
                                "safeModeCooldownSec": {"type": "literal", "value": "600"},
                            }
                        ]
                    },
                }
            else:
                return {
                    "@context": DEFAULT_JSONLD_CONTEXT,
                    "head": {"vars": []},
                    "results": {"bindings": []},
                }
        else:
            # In production, implement the call to the DKG SDK here
            # e.g. self.client.graph.query(query)
            raise NotImplementedError("DKG SDK integration not implemented.")

    @staticmethod
    def _parse_number(value: str) -> float:
        try:
            if "." in value or "e" in value.lower():
                return float(value)
            return float(int(value))
        except Exception:
            return float(value)

    def _validate_context(self, result: Dict[str, Any]) -> None:
        ctx = result.get("@context")
        if ctx and ctx != DEFAULT_JSONLD_CONTEXT:
            raise ValueError(f"Unexpected JSON-LD context: {ctx}")

    def _pick_best_binding(
        self,
        bindings: List[Dict[str, Any]],
        key_for_max: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        If there are multiple bindings:
        - if key_for_max is provided: take the one with the maximum value for this field
        - otherwise: last binding (Last Write Wins).
        """
        if not bindings:
            raise ValueError("No bindings in SPARQL result.")
        if key_for_max is None:
            return bindings[-1]

        best = bindings[0]
        best_val = self._parse_number(best[key_for_max]["value"])
        for b in bindings[1:]:
            v = self._parse_number(b[key_for_max]["value"])
            if v > best_val:
                best = b
                best_val = v
        return best

    def parse_integrity_facts(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse le KA d'intégrité produit par Sansón Ultra-Pro.
        """
        self._validate_context(result)
        bindings = result.get("results", {}).get("bindings", [])
        if not bindings:
            raise ValueError("No IntegrityAnalysis KA found for the guild.")

        b = bindings[-1]  # Last Write Wins

        def getf(name: str, default: Optional[str] = None):
            if name not in b:
                if default is None:
                    raise KeyError(name)
                return default
            return b[name]["value"]

        sigma = self._parse_number(getf("sigma"))
        cartel_flag = getf("cartelFlag").lower() == "true"
        subsidy_ratio = self._parse_number(getf("subsidyRatio"))
        integrity_score = self._parse_number(getf("integrityScore"))
        version = int(float(getf("version", "1")))
        source_ka = getf("sourceKA")
        valid_until = int(float(getf("validUntil", str(int(time.time()) + 3600))))

        return {
            "sigma": sigma,
            "cartelFlag": cartel_flag,
            "subsidyRatio": subsidy_ratio,
            "integrityScore": integrity_score,
            "version": version,
            "sourceKA": source_ka,
            "validUntil": valid_until,
        }

    def parse_rules_facts(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse the SPARQL result of the Ruleset.
        """
        self._validate_context(result)
        bindings = result.get("results", {}).get("bindings", [])
        if not bindings:
            # Fallback to local rules
            return FALLBACK_RULES.copy()

        binding = bindings[0]  # only one ruleset expected

        def get_field(name: str, default: Optional[str] = None) -> str:
            if name not in binding:
                if default is None:
                    raise KeyError(name)
                return default
            return binding[name]["value"]

        return {
            "sigma_min_bps": int(float(get_field("sigmaMinBps", "8000"))),
            "max_subsidy_ratio": float(get_field("maxSubsidyRatio", "0.3")),
            "reject_if_cartel": get_field("rejectIfCartel", "true").lower() == "true",
            "circuit_breaker_threshold": int(float(get_field("circuitBreakerThreshold", "5"))),
            "safe_mode_cooldown_sec": int(float(get_field("safeModeCooldownSec", "600"))),
        }


dkg_client = DKGClientWrapper(DKG_CONFIG)


# ---------------------------------------------------------------------------
# 3. Symbolic Reasoner (Neuro-Symbolique)
# ---------------------------------------------------------------------------

class SymbolicReasoner:
    """
    Decision logic using the IntegrityAnalysis KA from Sansón.
    """
    def decide(self, facts: Dict[str, Any], rules: Dict[str, Any], initial_amount: float):
        trace = []

        sigma = float(facts["sigma"])
        integrity = float(facts["integrityScore"])
        subsidy_ratio = float(facts["subsidyRatio"])
        cartel_flag = bool(facts["cartelFlag"])

        sigma_bps = int(sigma * 10000)
        integrity_bps = int(integrity * 10000)

        now = int(time.time())
        valid_until = int(facts["validUntil"])

        trace.append(f"Analyse KA Sansón: σ={sigma:.4f}, integrity={integrity:.4f}, subsidy={subsidy_ratio:.4f}, cartel={cartel_flag}")
        trace.append(f"Validity: now={now}, valid_until={valid_until}")

        # INITIAL DECISION
        decision: DecisionType = "PAY"
        calculated_amount = initial_amount

        # 1. Expiration du KA
        if now > valid_until:
            decision = "REJECT"
            calculated_amount = 0.0
            trace.append("Integrity KA expired → REJECT.")
            return {"decision": decision, "calculated_amount": calculated_amount, "trace": trace}

        # 2. Cartel Flag
        if cartel_flag and rules.get("reject_if_cartel", True):
            decision = "REJECT"
            calculated_amount = 0.0
            trace.append("CartelFlag=True → REJECT.")
            return {"decision": decision, "calculated_amount": calculated_amount, "trace": trace}

        # 3. Sigma minimum
        sigma_min_bps = int(rules.get("sigma_min_bps", 8000))
        if sigma_bps < sigma_min_bps:
            decision = "COUNTER_OFFER"
            calculated_amount = round(initial_amount * sigma, 2)
            trace.append(f"σ={sigma_bps} < min={sigma_min_bps} → COUNTER_OFFER {calculated_amount}.")
            return {"decision": decision, "calculated_amount": calculated_amount, "trace": trace}

        # 4. Subsidy Ratio
        max_subsidy_ratio = float(rules.get("max_subsidy_ratio", 0.3))
        if subsidy_ratio > max_subsidy_ratio:
            decision = "COUNTER_OFFER"
            calculated_amount = round(initial_amount * (1 - subsidy_ratio), 2)
            trace.append(f"SubsidyRatio too high → COUNTER_OFFER {calculated_amount}.")
            return {"decision": decision, "calculated_amount": calculated_amount, "trace": trace}

        trace.append("All conditions OK → PAY.")
        return {"decision": "PAY", "calculated_amount": initial_amount, "trace": trace}


reasoner = SymbolicReasoner()


# ---------------------------------------------------------------------------
# 4. XCM Builder + Substrate Connector
# ---------------------------------------------------------------------------

class ScaleEncoder:
    """
    Abstraction for encoding SCALE calls.
    In production: connect py-scale-codec, polkadot-js-api, or substrate-interface.
    """
    def encode_guild_tcr_update(
        self,
        contract_address: str,
        guild_id: str,
        metrics: Dict[str, Any],
    ) -> str:
        """
        Returns a hex string (0x...) representing the SCALE-encoded call.
        For now, stub/documentation.
        """
        # TODO: use a real SCALE encoder
        pseudo_call = {
            "contract": contract_address,
            "method": "updateGuildMetrics",
            "params": {
                "guildId": guild_id,
                "metrics": metrics,
            },
        }
        # We simulate a hex SCALE by hashing the JSON structure.
        blob = json.dumps(pseudo_call, sort_keys=True).encode("utf-8")
        return "0x" + hashlib.sha256(blob).hexdigest()


scale_encoder = ScaleEncoder()


class XCMBuilder:
    """
    Builds XCM V3/V4 messages for simple transfer or contract call.
    """

    def __init__(self, network_config: NetworkConfig):
        self.net = network_config

    def _amount_to_chain_units(self, amount: float, decimals: int = 12) -> int:
        return int(amount * (10**decimals))

    def build_transfer_only(
        self,
        amount: float,
        asset_multilocation: Dict[str, Any],
        beneficiary: str,
    ) -> Dict[str, Any]:
        amount_chain = self._amount_to_chain_units(amount)
        return {
            "V3": [
                {
                    "WithdrawAsset": [
                        {
                            "id": {"Concrete": asset_multilocation},
                            "fun": {"Fungible": amount_chain},
                        }
                    ]
                },
                {
                    "BuyExecution": {
                        "fees": {
                            "id": {"Concrete": asset_multilocation},
                            "fun": {"Fungible": amount_chain // 10},  # 10% pour les fees
                        },
                        "weight_limit": "Unlimited",
                    }
                },
                {
                    "DepositAsset": {
                        "assets": {"Wild": "All"},
                        "beneficiary": {
                            "parents": 0,
                            "interior": {
                                "X1": {
                                    "AccountId32": {
                                        "network": None,
                                        "id": beneficiary,
                                    }
                                }
                            },
                        },
                    }
                },
            ]
        }

    def build_contract_call(
        self,
        amount: float,
        asset_multilocation: Dict[str, Any],
        beneficiary: str,
        scale_call_hex: str,
        estimated_weight: int = 1_000_000_000,
    ) -> Dict[str, Any]:
        amount_chain = self._amount_to_chain_units(amount)
        return {
            "V3": [
                {
                    "WithdrawAsset": [
                        {
                            "id": {"Concrete": asset_multilocation},
                            "fun": {"Fungible": amount_chain},
                        }
                    ]
                },
                {
                    "BuyExecution": {
                        "fees": {
                            "id": {"Concrete": asset_multilocation},
                            "fun": {"Fungible": amount_chain // 10},
                        },
                        "weight_limit": "Unlimited",
                    }
                },
                {
                    "Transact": {
                        "origin_kind": "Native",
                        "require_weight_at_most": estimated_weight,
                        "call": scale_call_hex,
                    }
                },
                {
                    "DepositAsset": {
                        "assets": {"Wild": "All"},
                        "beneficiary": {
                            "parents": 0,
                            "interior": {
                                "X1": {
                                    "AccountId32": {
                                        "network": None,
                                        "id": beneficiary,
                                    }
                                }
                            },
                        },
                    }
                },
            ]
        }


xcm_builder = XCMBuilder(NETWORK_CONFIG)


class SubstrateConnector:
    """
    Send the XCM message on the Relay Chain/Parachain.
    In production: substrate-interface, polkadot-js, custom RPC, etc.
    """
    def __init__(self, network_config: NetworkConfig):
        self.net = network_config
        self.mock_mode = True

    async def send_xcm(
        self,
        xcm_payload: Dict[str, Any],
        destination: Dict[str, Any],
    ) -> Dict[str, Any]:
        if self.mock_mode:
            print(f"[XCM] Sending to destination {json.dumps(destination)}")
            print(f"[XCM] Payload: {json.dumps(xcm_payload, indent=2)}")
            await asyncio.sleep(0.8)
            return {
                "status": "SUCCESS",
                "tx_hash": "0xdeadbeefcafebabefeedface0000000000000000000000000000000000000000",
                "block_number": 123456,
            }
        else:
            # TODO: implement the actual RPC/extrinsic call
            raise NotImplementedError("Substrate RPC not implemented.")


substrate_connector = SubstrateConnector(NETWORK_CONFIG)


# ---------------------------------------------------------------------------
# 5. Tools (DKG / XCM / Publish)
# ---------------------------------------------------------------------------

async def tool_query_dkg(query_sparql: str) -> Dict[str, Any]:
    try:
        result = await dkg_client.query_sparql(query_sparql)
        return result
    except Exception as e:
        return {"error": str(e)}

async def tool_execute_xcm_payment(
    xcm_payload: Dict[str, Any],
    destination: Dict[str, Any],
) -> Dict[str, Any]:
    try:
        print("[BRIDGE] 🌉 Sancho activates the real XCM engine (via xcm_engine)...")
        
        target_para = NETWORK_CONFIG["parachain_id"]
        guild_addr = NETWORK_CONFIG["beneficiary_account"]
        
        amount_to_pay = 10.0 

        receipt = real_xcm_tool(
            source_chain="Moonbeam",
            target_parachain_id=target_para,
            guild_address=guild_addr,
            amount=amount_to_pay
        )
        
        return receipt

    except Exception as e:
        print(f"[BRIDGE ERROR] {str(e)}")
        return {"status": "ERROR", "error": str(e)}

async def tool_publish_dkg(data_jsonld: Dict[str, Any]) -> bool:
    """
    In production: creation of a KA on the DKG (asset.create).
    Here: log + mock.
    """
    print("[DKG PUBLISH] KA:", json.dumps(data_jsonld, indent=2))
    await asyncio.sleep(0.2)
    return True


# ---------------------------------------------------------------------------
# 6. Resilience Layer (timeouts, circuit-breaker, safe-mode)
# ---------------------------------------------------------------------------

def is_in_safe_mode(state: AgentSanchoState) -> bool:
    until = state.get("safe_mode_until")
    if not until:
        return False
    return time.time() < until

def update_circuit_breaker_on_failure(
    state: AgentSanchoState,
    rules: Dict[str, Any],
) -> AgentSanchoState:
    state["consecutive_failures"] = state.get("consecutive_failures", 0) + 1
    threshold = int(rules.get("circuit_breaker_threshold", FALLBACK_RULES["circuit_breaker_threshold"]))
    cooldown = int(rules.get("safe_mode_cooldown_sec", FALLBACK_RULES["safe_mode_cooldown_sec"]))

    if state["consecutive_failures"] >= threshold:
        state["health_status"] = "SAFE_MODE"
        state["safe_mode_until"] = time.time() + cooldown
        state.setdefault("full_reasoning_trace", []).append(
            f"Resilience: circuit-breaker triggered, safe-mode for {cooldown}s."
        )
    return state

def reset_circuit_breaker_on_success(state: AgentSanchoState) -> AgentSanchoState:
    state["consecutive_failures"] = 0
    state["safe_mode_until"] = None
    return state


# ---------------------------------------------------------------------------
# 7. Agent Nodes (LangGraph-like)
# ---------------------------------------------------------------------------

async def schedule_verification(state: AgentSanchoState) -> AgentSanchoState:
    state.setdefault("full_reasoning_trace", [])
    state.setdefault("health_status", "OK")
    state.setdefault("retry_counter_dkg", 0)
    state.setdefault("retry_counter_xcm", 0)
    state.setdefault("consecutive_failures", 0)

    tache = state.get("input_task", {})
    service_id = tache.get("service_id", "ka://service/default")
    amount = float(tache.get("amount", 0.0))
    mode: ExecutionMode = tache.get("execution_mode", "TRANSFER_ONLY")  # or "CONTRACT_CALL"

    state["calculated_amount"] = amount
    state["execution_mode"] = mode

    # SPARQL query : IntegrityAnalysis KA produit par Sansón
    state["dkg_query_reputation"] = f"""
PREFIX barataria: <https://schema.barataria.arch/v1#>

SELECT ?sigma ?cartelFlag ?subsidyRatio ?integrityScore ?version ?sourceKA ?validUntil
WHERE {{
  # Récupération de la guilde assurant le service
  ?service a barataria:Service ;
           barataria:serviceId "{service_id}" ;
           barataria:insuredBy ?guild .

  # Récupération du KA d'intégrité publié par Sansón
  ?ia a barataria:IntegrityAnalysis ;
      barataria:guildId ?guild ;
      barataria:sigma ?sigma ;
      barataria:cartelFlag ?cartelFlag ;
      barataria:subsidyRatio ?subsidyRatio ;
      barataria:integrityScore ?integrityScore ;
      barataria:version ?version ;
      barataria:generatedFromKA ?sourceKA ;
      barataria:validUntil ?validUntil .
}}
ORDER BY DESC(?version)
LIMIT 1
"""

    # SPARQL query : ruleset global
    state["dkg_query_rules"] = """
    PREFIX barataria: <https://schema.barataria.arch/v1#>

    SELECT ?sigmaMinBps ?maxSubsidyRatio ?rejectIfCartel ?circuitBreakerThreshold ?safeModeCooldownSec
    WHERE {
      ?ruleset a barataria:Ruleset ;
               barataria:sigmaMinBps ?sigmaMinBps ;
               barataria:maxSubsidyRatio ?maxSubsidyRatio ;
               barataria:rejectIfCartel ?rejectIfCartel ;
               barataria:circuitBreakerThreshold ?circuitBreakerThreshold ;
               barataria:safeModeCooldownSec ?safeModeCooldownSec .
    }
    LIMIT 1
    """

    # MultiLocation destination (NeuroWeb)
    state["multilocation_destination"] = {
        "V3": {
            "parents": 1,
            "interior": {"X1": {"Parachain": NETWORK_CONFIG["parachain_id"]}},
        }
    }

    state["asset_id"] = "xcTRAC"
    state["full_reasoning_trace"].append(
        f"plan_verification: SPARQL for service {service_id}, mode={mode}."
    )
    return state


async def auto_diagnostique(state: AgentSanchoState) -> AgentSanchoState:
    now = time.time()
    last_success = state.get("last_success_timestamp")

    if is_in_safe_mode(state):
        state["health_status"] = "SAFE_MODE"
        state["full_reasoning_trace"].append("self-diagnosis: SAFE_MODE active, no economic decision.")
        return state

    if last_success and (now - last_success > 300):
        state["health_status"] = "DEGRADED"
        state["full_reasoning_trace"].append("self-diagnosis: health DEGRADED (last success > 5 min).")
    else:
        state["health_status"] = "OK"
        state["full_reasoning_trace"].append("self-diagnosis: health OK.")

    return state


async def interrogate_dkg(state: AgentSanchoState) -> AgentSanchoState:
    """
    Double fetch: IntegrityAnalysis KA + ruleset, with timeout and robust parsing.
    """
    if state.get("health_status") == "SAFE_MODE":
        return state

    query_rep = state["dkg_query_reputation"]
    query_rules = state["dkg_query_rules"]

    try:
        t0 = time.time()
        raw_rep = await asyncio.wait_for(tool_query_dkg(query_rep), timeout=5.0)
        raw_rules = await asyncio.wait_for(tool_query_dkg(query_rules), timeout=5.0)
        latency = time.time() - t0
        state["dkg_query_latency"] = latency
        state["raw_dkg_response_reputation"] = raw_rep
        state["raw_dkg_response_rules"] = raw_rules
        state["full_reasoning_trace"].append(
            f"interrogate_dkg: DKG queries performed in {latency:.3f}s."
        )

        if "error" in raw_rep:
            raise RuntimeError(f"DKG reputation error: {raw_rep['error']}")
        if "error" in raw_rules:
            raise RuntimeError(f"DKG rules error: {raw_rules['error']}")

        facts = dkg_client.parse_integrity_facts(raw_rep)
        rules = dkg_client.parse_rules_facts(raw_rules)

        state["dkg_facts"] = facts
        state["rules_dkg"] = rules
        state["health_status"] = "OK"
        state["full_reasoning_trace"].append("interrogate_dkg: Facts & rules successfully extracted.")
        return state

    except Exception as e:
        state["health_status"] = "DKG_UNREACHABLE"
        state["final_decision"] = "ERROR"
        msg = f"interrogate_dkg: critical failure DKG → {str(e)}"
        state["logical_decision"] = msg
        state["full_reasoning_trace"].append(msg)
        # Circuit breaker based on FALLBACK_RULES (we may not have rules_dkg)
        update_circuit_breaker_on_failure(state, state.get("rules_dkg", FALLBACK_RULES))
        return state


async def reasoning_based_on_facts(state: AgentSanchoState) -> AgentSanchoState:
    if state.get("health_status") in ("DKG_UNREACHABLE", "SAFE_MODE"):
        return state

    facts = state.get("dkg_facts")
    rules = state.get("rules_dkg") or FALLBACK_RULES
    if not facts:
        state["final_decision"] = "ERROR"
        state["logical_decision"] = "Pas de faits DKG disponibles pour raisonner."
        state["full_reasoning_trace"].append("reasoning_based_on_facts: aucun fait → ERROR.")
        update_circuit_breaker_on_failure(state, rules)
        return state

    initial_amount = float(state["input_task"].get("amount", 0.0))
    result = reasoner.decide(facts, rules, initial_amount)

    decision: DecisionType = result["decision"]  # type: ignore
    calculated_amount = result["calculated_amount"]
    trace = result["trace"]

    state["final_decision"] = decision
    state["calculated_amount"] = calculated_amount
    state["full_reasoning_trace"].extend(trace)
    state["logical_decision"] = " || ".join(trace)

    if decision in ("PAY", "COUNTER_OFFER"):
        reset_circuit_breaker_on_success(state)
    else:
        update_circuit_breaker_on_failure(state, rules)

    return state


async def execute_decision(state: AgentSanchoState) -> AgentSanchoState:
    decision = state.get("final_decision", "ERROR")
    if decision not in ("PAY", "COUNTER_OFFER"):
        state["full_reasoning_trace"].append(
            f"execute_decision: décision={decision}, aucun XCM construit."
        )
        return state

    if state.get("health_status") == "SAFE_MODE":
        state["full_reasoning_trace"].append(
            "execute_decision: SAFE_MODE actif, XCM bloqué."
        )
        return state

    amount = state.get("calculated_amount", 0.0)
    asset_loc = NETWORK_CONFIG["asset_multilocation"]
    beneficiary = NETWORK_CONFIG["beneficiary_account"]
    dest = state["multilocation_destination"]
    mode: ExecutionMode = state.get("execution_mode", "TRANSFER_ONLY")

    # Option: contract call (via Transact) or simple transfer
    if mode == "CONTRACT_CALL":
        # Example: call GuildTCR.updateGuildMetrics
        guild_id = state["input_task"].get("guild_id", "guild:unknown")
        metrics = {
            "lastPaidAmount": amount,
            "sigma": state["dkg_facts"]["sigma"],  # type: ignore
            "integrityScore": state["dkg_facts"]["integrityScore"],  # type: ignore
        }
        scale_call_hex = scale_encoder.encode_guild_tcr_update(
            NETWORK_CONFIG["guild_tcr_address"],
            guild_id,
            metrics,
        )
        xcm_message = xcm_builder.build_contract_call(
            amount=amount,
            asset_multilocation=asset_loc,
            beneficiary=beneficiary,
            scale_call_hex=scale_call_hex,
        )
    else:
        xcm_message = xcm_builder.build_transfer_only(
            amount=amount,
            asset_multilocation=asset_loc,
            beneficiary=beneficiary,
        )

    state["xcm_call_payload"] = xcm_message

    try:
        result = await asyncio.wait_for(
            tool_execute_xcm_payment(xcm_message, dest),
            timeout=8.0,
        )
        if result.get("status") == "SUCCESS":
            state["tx_hash"] = result.get("tx_hash")
            state["health_status"] = "OK"
            state["last_success_timestamp"] = time.time()
            state["full_reasoning_trace"].append(
                f"execute_decision: send XCM, tx_hash={state['tx_hash']}."
            )
            reset_circuit_breaker_on_success(state)
        else:
            state["health_status"] = "XCM_FAILED"
            state["full_reasoning_trace"].append(
                f"execute_decision: fail XCM → {result}."
            )
            update_circuit_breaker_on_failure(
                state, state.get("rules_dkg", FALLBACK_RULES)
            )
    except asyncio.TimeoutError:
        state["health_status"] = "XCM_FAILED"
        state["full_reasoning_trace"].append(
            "execute_decision: timeout XCM."
        )
        update_circuit_breaker_on_failure(
            state, state.get("rules_dkg", FALLBACK_RULES)
        )

    return state


def _compute_reasoning_digest(trace: List[str]) -> str:
    blob = "\n".join(trace).encode("utf-8")
    return "0x" + hashlib.sha256(blob).hexdigest()


async def publish_consequence(state: AgentSanchoState) -> AgentSanchoState:
    """
   Publish two KAs:
      1) TransactionResult KA (BaratariaTransaction-like)
      2) ReasoningTrace KA (complete CoT with digest)
    """
    tache = state.get("input_task", {})
    service_id = tache.get("service_id", "ka://service/default")
    decision = state.get("final_decision", "ERROR")
    amount = state.get("calculated_amount", 0.0)
    tx_hash = state.get("tx_hash")
    facts = state.get("dkg_facts") or {}
    source_ka = facts.get("sourceKA", "ka://unknown/source")
    trace = state.get("full_reasoning_trace", [])

    reasoning_digest = _compute_reasoning_digest(trace)
    state["reasoning_digest"] = reasoning_digest

    now_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    # 1) ReasoningTrace KA
    reasoning_ka = {
        "@context": DEFAULT_JSONLD_CONTEXT,
        "@type": "BaratariaReasoningTrace",
        "identifier": f"reasoning:{tx_hash or int(time.time())}",
        "generatedAtTime": now_iso,
        "decision": decision,
        "reasoningDigest": reasoning_digest,
        "fullTrace": trace,
        "usedSourceKA": source_ka,
    }

    # 2) TransactionResult KA
    tx_ka = {
        "@context": DEFAULT_JSONLD_CONTEXT,
        "@type": "BaratariaTransaction",
        "identifier": f"tx:{tx_hash}" if tx_hash else "tx:pending-or-rejected",
        "description": "Evidence of Sancho's decision based on DKG reputation.",
        "mainEntityOfPage": service_id,
        "decision": decision,
        "price": amount,
        "priceCurrency": "TRAC",
        "prov:wasDerivedFrom": {
            "@type": "Dataset",
            "identifier": source_ka,
        },
        "prov:wasGeneratedBy": {
            "@type": "SoftwareApplication",
            "name": "Sancho Agent v3",
            "softwareVersion": "3.0",
            "reasoningDigest": reasoning_digest,
        },
        "potentialAction": {
            "@type": "PayAction",
            "actionStatus": (
                "CompletedActionStatus"
                if tx_hash and state.get("health_status") == "OK"
                else "FailedActionStatus"
            ),
            "object": {
                "@type": "BlockchainTransaction",
                "identifier": tx_hash,
            },
        },
    }

    await tool_publish_dkg(reasoning_ka)
    await tool_publish_dkg(tx_ka)

    state["full_reasoning_trace"].append(
        "publish_consequence: ReasoningTrace KA & TransactionResult KA publiés."
    )
    return state


async def state_ERROR(state: AgentSanchoState) -> AgentSanchoState:
    state["full_reasoning_trace"].append(
        f"state_ERROR: health={state.get('health_status')}, decision={state.get('final_decision')}."
    )
    return state


# ---------------------------------------------------------------------------
# 8. Simple (sequential) wiring & execution helper
# ---------------------------------------------------------------------------

async def run_sancho_once(input_task: Dict[str, Any]) -> AgentSanchoState:
    """
    Runs the Sancho v3 pipeline sequentially (without LangGraph), with a global timeout.
    """
    state: AgentSanchoState = {"input_task": input_task}
    try:
        async def _run():
            nonlocal state
            state = await schedule_verification(state)
            state = await auto_diagnostique(state)
            if state.get("health_status") == "SAFE_MODE":
                state = await state_ERROR(state)
                return state

            state = await interrogate_dkg(state)
            if state.get("health_status") == "DKG_UNREACHABLE":
                state = await state_ERROR(state)
                return state

            state = await reasoning_based_on_facts(state)
            if state.get("final_decision") in ("PAY", "COUNTER_OFFER"):
                state = await execute_decision(state)

            if state.get("final_decision") in ("PAY", "REJECT", "COUNTER_OFFER"):
                state = await publish_consequence(state)
            else:
                state = await state_ERROR(state)
            return state

        state = await asyncio.wait_for(_run(), timeout=GLOBAL_TIMEOUT_SEC)
        return state

    except asyncio.TimeoutError:
        state["health_status"] = "DEGRADED"
        state["final_decision"] = "ERROR"
        state.setdefault("full_reasoning_trace", []).append(
            "run_sancho_once: timeout global dépassé."
        )
        return state


# Example
# if __name__ == "__main__":
#     task = {
#         "service_id": "ka://service/fiable",
#         "amount": 150.0,
#         "execution_mode": "CONTRACT_CALL",  # ou "TRANSFER_ONLY"
#         "guild_id": "guild:12345",
#     }
#     final_state = asyncio.run(run_sancho_once(task))
#     print(json.dumps(final_state, indent=2, default=str))
