# BARATARIA

[![Barataria Manifesto (EN) preview](assets/barataria-manifesto-en.png)](./%20Barataria%20Manifesto%20(EN).pdf)

Origin Trail (TRAC), Neuro Web (Neuro), Polkadot (DOT)  challenge ! 

Prototype aligned with the **Barataria Manifesto**: a distributed
chivalry of trust, throneless, where reputation is a local,
probationary, and interchain flow. 

This model---without a throne---outsources integrity assessment,
prohibits centralized decision‑making, and embeds trust in the
engagement economy.\
Local policies become the only sovereigns, guided by public metrics that
are auditable and resistant to capture.\
The model is designed to be decentralized, with no central authority to
control or manipulate the system.

This repository demonstrates how to anchor the three pillars (costly
identity, mirror of the graph of consequence, pluralistic arbitration)
in an x402 payment agent supported by OriginTrail DKG and XCM.

## Intent (manifesto → code)

-   **Thronelessness**: no global score (like global Page Rank, see Law of Goodhart) to prevent Sybil attacks; local
    decisions guided by published metrics (sigma, integrityScore,
    subsidyRatio, cartelFlag).
-   **Burn‑to‑Register Sybil Tax**: identities coupled with a dynamic
    cost to prevent Sybil attacks; The issuance of identity is based on a dynamic cost function 
    designed to discourage Sybil attacks through exponential taxation adjusted according to the 
    flow of registrations. This mechanism introduces a growing economic barrier without resorting 
    to mutable centralized governance. the Trust layer remains neutral and immutable. 
-   **Graph of Consequence**: only real x402 transactions count (See our solution against intra-cluster link 
    farms between agents in our white paper. ),
    avoiding reliance on Web2 graphs possibly already corrupted by Sybil attacks and Sybil web3 corruption 
    attacks; the DKG exposes an auditable mirror (versioned KAs, SHACL
    schemas). 
-   **Arbitration Archipelago**: multiple competing arbitration guilds
    (for disputes); their integrity indices are consulted before
    payment. (The DKG automatically publishes: Proof of Yield (Tracks the source of APY), 
    Correlation Index (Detects if 50 Guilds are acting as a single Sybil cartel), 
    and Dispute History (How the Guild voted).)
-   **Cognitive Trihedron**: in homage to the novel Don Quixote, a foundational 
    work by Cervantes dealing with the modern epistemology of falsehood 
    we propose 3 distinct agents : Sancho (payment), Sanson (formalized
    analysis), Quichotte (topological monitoring). This POC implements
    Sancho and the primitives to host the other two.

## Technical Map (3 layers)

-   **Agent (Sancho)**: `tools/main_agent_sancho.py`
    -   Async pipeline: SPARQL collection, JSON‑LD parsing, symbolic
        reasoning, XCM construction.\
    -   Resilience: global timeouts, circuit breaker, safe mode, SHA‑256
        audit trace.\
    -   Toggleable mock: `DKGClientWrapper.mock_mode` and
        `tool_execute_xcm_payment` simulating HRMP/XCM sending.
-   **Knowledge (DKG)**:
    -   JSON‑LD schema and SHACL shapes: `assets/guild-schema.jsonld`,
        `assets/guild-shapes.ttl`.\
    -   Example published KA: `assets/guild-giant-asset.jsonld`, `assets/guild-windmill-asset.jsonld`
        (integrityScore, subsidyRatio, sigma, cartelStatus, audit).\
    -   Publish/query helpers: `tools/publish_knowledge.py`,
        `tools/query_debug.py`, `tools/test_connect.py`.
-   **Trust (XCM / NeuroWeb)**: `core/xcm_engine.py`
    -   XCM V3 construction (WithdrawAsset, InitiateReserveWithdraw,
        BuyExecution, DepositAsset).\
    -   Simulation of cross‑parachain x402 payment and receipt (hash,
        timestamp, destination guild).

## Pillar by Pillar (manifesto → POC)

-   **Pillar 1 --- Guarantee / Costly Identity**
    -   Manifesto: Burn‑to‑Register with adaptive Sybil tax, median
        oracle.\
    -   POC: DKG connection (`core/dkg_config.py`) and `NETWORK_CONFIG`
        preparing the call to the BurnToRegister contract on NeuroWeb.
        The private key loader applies the padding required by
        DKG/chain.
-   **Pillar 2 --- Mirror of the Graph of Consequence**
    -   Manifesto: local reputation (LocalTrust), no global PageRank,
        analytic KAs (yield, correlation, disputes).\
    -   POC: Sancho queries the DKG via SPARQL, extracts `sigma`,
        `integrityScore`, `subsidyRatio`, `cartelFlag`, then applies
        `SymbolicReasoner.decide` with ruleset (min sigma in bps,
        subsidy cap, cartel rejection). Auditable hashed trace.
-   **Pillar 3 --- Action / x402 + Arbitration Archipelago**
    -   Manifesto: payment as probative footprint; automatic refusal if
        the insurer is subsidized/correlated.\
    -   POC: `tool_execute_xcm_payment` assembles the XCM program, pays
        execution, deposits to target guild. Circuit breaker enters
        safe‑mode after `circuit_breaker_threshold` failures.

## Demonstration Path

1)  **Install environment**

    ``` bash
    python -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    ```

2)  **Publish a guild KA (initial mirror)**

    ``` bash
    python tools/publish_knowledge.py
    ```

    Publishes `assets/guild-giant-asset.jsonld` and `assets/guild-windmill-asset.jsonld` to the DKG (`epochs_num=8`)
    and returns a UAL Sancho can use.

3)  **Run Sanson (integrity analytics before payment)**

    -   Set the `guild` value in the `_demo` block at the bottom of
        `tools/sanson_agent.py` to the UAL returned in step 2 (for
        example `urn:barataria:guild:giant`).\
    -   Execute `python tools/sanson_agent.py` (or
        `SANSON_FORCE_MOCK=1 python tools/sanson_agent.py` for
        deterministic demo data).\
    -   Sanson pulls yield/cartel KAs, computes
        `subsidyRatio`/`cartelFlag`/`sigma`, and publishes
        `ka://analytics/integrity/<guild>/vX` for Sancho to consume.

4)  **Run Sancho (local decision + payment)**

    -   Fill `input_task` in `tools/main_agent_sancho.py` (service_id,
        amount, execution_mode). Sancho reads the latest integrity KA
        published by Sanson; if missing, it falls back to
        `FALLBACK_RULES` (hardcoded thresholds).\
    -   The agent:
        -   builds two SPARQL queries (reputation + ruleset) to the
            DKG,\
        -   extracts facts, applies ruleset (default `FALLBACK_RULES` if
            KA missing),\
        -   computes `decision` ∈ {REJECT, PAY, COUNTER_OFFER}, adjusts
            amount, hashes trace,\
        -   if PAY, builds and sends (or simulates) XCM and returns a
            receipt (hash, timestamp).\
    -   Safe‑mode if consecutive failures \> threshold.

## Essential Configuration

-   Variables (via `.env`):
    -   `OT_NODE_URL`: HTTP endpoint of DKG node (default
        `http://localhost:8900`).\
    -   `PRIVATE_KEY`: hex key without prefix; loader adds `0x` and
        padding.
-   Network / XCM parameters: `NETWORK_CONFIG` in
    `tools/main_agent_sancho.py` (relay chain, para_id, beneficiary,
    guild TCR).
-   Mock / prod: set `DKGClientWrapper.mock_mode=False` and replace
    `tool_execute_xcm_payment` with a Substrate connector for a live
    parachain.

## Manifesto Agent Alignment

-   **Sancho** (present): payment agent, local heuristic, trace audit.\
-   **Sanson** (present): integrity index computation, analytics KA
    publication (yield, disputes, cartel-index).\
-   **Quichotte** (to connect): topological analyses (link farms,
    cartels), alert publication (KA).\
    Both can attach to the same DKG JSON‑LD registry and feed Sancho's
    reasoner.

## Quick Resources

-   Manifesto: `Barataria Manifesto.pdf`\
-   Schema / shapes: `assets/guild-schema.jsonld`,
    `assets/guild-shapes.ttl`\
-   Agent: `tools/main_agent_sancho.py`\
-   SDK / connectors: `core/dkg_config.py`, `core/xcm_engine.py`
