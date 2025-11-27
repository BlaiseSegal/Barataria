# publish_knowledge.py
import json
import os
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
import sys
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.dkg_config import get_dkg_client

# dkg = get_dkg_client()
# print("[DEBUG] Using wallet:", dkg.blockchain_provider.account.address)
from core.dkg_config import get_dkg_client
dkg = get_dkg_client()

print("Stake ask =", dkg.asset.blockchain_service.get_stake_weighted_average_ask())