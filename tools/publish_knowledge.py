# publish_knowledge.py

import json
import os
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
import sys
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.dkg_config import get_dkg_client
import dkg.utils.knowledge_collection_tools as kc_tools
from dkg.modules.asset.asset import CHUNK_BYTE_SIZE

# Configuration
EPOCHS_NUM = 5
DECIMALS = int(os.getenv("BLOCKCHAIN_DECIMALS", "18"))
UNIT = Decimal(10) ** DECIMALS

def load_guild_asset(path: str) -> dict:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if "public" not in data:
        raise ValueError("The JSON file must contain a ‘public’ key.")
    return data

def estimate_cost(dkg, content: dict) -> tuple[int, int]:
    formatted = kc_tools.format_dataset(content)
    public_triples = kc_tools.generate_missing_ids_for_blank_nodes(
        formatted.get("public", [])
    )
    number_of_chunks = kc_tools.calculate_number_of_chunks(
        public_triples, CHUNK_BYTE_SIZE
    )
    dataset_size = number_of_chunks * CHUNK_BYTE_SIZE
    
    try:
        stake_ask = dkg.asset.blockchain_service.get_stake_weighted_average_ask()
        dataset_kb = dataset_size / 1024
        estimated = int(stake_ask * EPOCHS_NUM * dataset_kb)
    except Exception as e:
        print(f"[WARN] Could not fetch stake ask: {e}. Using default estimation.")
        estimated = 0
    
    return estimated, dataset_size

def publish_guild_42(asset_path: str) -> str:
    dkg = get_dkg_client()

    status = dkg.node.info
    if status.get("version") is None:
        raise RuntimeError("Unable to obtain node version – check connectivity.")

    content = load_guild_asset(asset_path)
    estimated_cost, dataset_size = estimate_cost(dkg, content)
    
    print(
        f"[INFO] Dataset size: {dataset_size} bytes. "
        f"Estimated cost: {Decimal(estimated_cost) / UNIT:.6f} NEURO/TRAC. "
        "Skipping fund verification (Safety Override Active)..."
    )

    options = {
        "epochs_num": EPOCHS_NUM,
        "minimum_number_of_finalization_confirmations": 2,
        "minimum_number_of_node_replications": 1,
        "gasLimit": 2500000,
        "max_number_of_retries": 30,
        "wait_time_between_retries": 5
        
    }

    print("------------------------------------------------")
    print("Initiating Publication of Guild Knowledge Asset...")
    print("Please wait, network propagation can take up to 150 seconds...")
    
    try:
        result = dkg.asset.create(content=content, options=options)
        
        if "UAL" in result:
            ual = result["UAL"]
            print("------------------------------------------------")
            print(f"[SUCCESS] Knowledge Asset created.")
            print(f"UAL : {ual}")
            print("------------------------------------------------")
            return ual
        else:
            print(f"[DEBUG] Raw result: {result}")
            raise RuntimeError("UAL not found in response")

    except Exception as e:
        print(f"[ERROR] Detailed error during asset creation: {e}")
        raise e

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    asset_path_calculated = os.path.join(current_dir, '..', 'assets', 'guild-windmill-asset.jsonld')
    
    try:
        ual = publish_guild_42(asset_path_calculated)
        print(f"[FINAL] UAL ready for Graph Indexing: {ual}")
    except Exception as e:
        print("[FAIL] Process aborted.")