# query_guild_by_contract.py
from core.dkg_config import get_dkg_client

SPARQL_TEMPLATE = """
PREFIX schema: <http://schema.org/>
PREFIX trust: <http://barataria.arch/ns#>

SELECT ?ual ?guild ?name ?integrity ?subsidy ?sigma ?cartel ?lastAudit ?auditor
WHERE {
  GRAPH ?g {
    ?guild a schema:Organization, trust:Guild ;
           trust:contractAddress "%(contract_iri)s" ;
           schema:name ?name ;
           trust:integrityScore ?integrity ;
           trust:subsidyRatio ?subsidy ;
           trust:sigma ?sigma ;
           trust:cartelStatus ?cartel ;
           trust:lastAudit ?lastAudit ;
           trust:auditedBy ?auditor .
  }
  BIND(STR(?g) AS ?ual)
}
LIMIT 1
"""

def get_guild_reputation_by_contract(contract_iri: str) -> dict | None:
    """
    contract_iri = "eip155:20430:0x4242..."
    Returns a dict ready to be consumed by the Sancho agent..
    """
    dkg = get_dkg_client()
    query = SPARQL_TEMPLATE % {"contract_iri": contract_iri}

    result = dkg.graph.query(query)
    if result.get("status") != "COMPLETED":
        raise RuntimeError(f"SPARQL query not completed: {result}")

    rows = result.get("data") or []
    if not rows:
        return None

    row = rows[0]
    return {
        "ual": row["ual"],
        "guild_iri": row["guild"],
        "name": row["name"],
        "integrityScore": float(row["integrity"]),
        "subsidyRatio": float(row["subsidy"]),
        "sigma": float(row["sigma"]),
        "cartelStatus": row["cartel"],
        "lastAudit": row["lastAudit"],
        "auditedBy": row["auditor"],
    }

if __name__ == "__main__":
    contract = "eip155:20430:0x4242424242424242424242424242424242424242"
    rep = get_guild_reputation_by_contract(contract)
    print(rep)