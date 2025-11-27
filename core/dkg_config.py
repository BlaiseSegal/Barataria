# dkg_config.py
import os
from dotenv import load_dotenv
from dkg import DKG
from dkg.providers import BlockchainProvider, NodeHTTPProvider

load_dotenv()

def get_dkg_client() -> DKG:
    endpoint = os.getenv("OT_NODE_URL", "http://localhost:8900")
    blockchain_id = os.getenv("BLOCKCHAIN_ID", "otp:20430")
    blockchain_rpc = os.getenv("BLOCKCHAIN_RPC")

    # Prefer dedicated publish wallet keys when available
    private_key = os.getenv("PUBLISH_WALLET_01_PRIVATE_KEY") or os.getenv("PRIVATE_KEY")
    if not private_key:
        raise RuntimeError("Missing PRIVATE_KEY or PUBLISH_WALLET_01_PRIVATE_KEY in environment.")

    if private_key.startswith("0x"):
        clean_key = private_key[2:]
    else:
        clean_key = private_key
    if len(clean_key) < 64:
        padding_needed = 64 - len(clean_key)
        clean_key = ("0" * padding_needed) + clean_key
        print(f"⚠️  Corrected key: Zeros added ({padding_needed})")
        
    final_private_key = "0x" + clean_key
    
    os.environ["PRIVATE_KEY"] = final_private_key
    
    node_provider = NodeHTTPProvider(
        endpoint_uri=endpoint, 
        api_version="v1"
    )

    blockchain_provider = BlockchainProvider(
        blockchain_id,
        blockchain_rpc or "https://lofar-testnet.origin-trail.network",
        final_private_key,
    )

    dkg = DKG(node_provider, blockchain_provider)
    
    return dkg
