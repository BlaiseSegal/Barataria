# core/xcm_engine.py
# ============================================================
#  Polkadot / NeuroWeb XCM Engine — x402 Payment Implementation
#  
#  - Helpers XCM (MultiLocation / MultiAsset)
#  - x402 Protocol Layer (Envelope + metaHash)
#  - EVM Call Builder (mocked)
#  - Full XCM V3 Source + Destination program
# ============================================================

import json
import time
import hashlib
from datetime import datetime
from typing import Tuple, Dict, Any


# ============================================================
# 1. Helpers XCM (MultiLocation / MultiAsset)
# ============================================================

def multilocation_parachain(para_id: int) -> Dict[str, Any]:
    """
    Target parachain MultiLocation: parents=1, interior=X1(Parachain=para_id)
    """
    return {
        "parents": 1,
        "interior": {"X1": {"Parachain": para_id}}
    }


def multilocation_account_key20(address_hex: str) -> Dict[str, Any]:
    """
    Beneficiary account using 20-byte address (EVM-compatible parachains).
    """
    return {
        "parents": 0,
        "interior": {
            "X1": {
                "AccountKey20": {
                    "network": None,
                    "key": address_hex
                }
            }
        }
    }


def concrete_here_asset(amount: int) -> Dict[str, Any]:
    """
    Fungible asset located in the RelayChain context:
      MultiLocation = {parents:1, interior:"Here"}
    """
    return {
        "id": {
            "Concrete": {
                "parents": 1,
                "interior": "Here"
            }
        },
        "fun": {"Fungible": amount}
    }


# ============================================================
# 2. Application Layer  x402 (Envelope, Hash, Call Encoding)
# ============================================================

def build_x402_envelope(
    pay_chain: str,
    pay: str,
    guild: str,
    asset_symbol: str,
    amount: float,
    dkg_claim: str,
    sigma_bps: int,
) -> Tuple[Dict[str, Any], str]:
    """
    Build the x402 envelope + calculate a SHA-256 metaHash.
    """
    envelope = {
        "protocol": "x402",
        "version": "1.0",
        "pay_chain": pay_chain,
        "pay": pay,
        "guild": guild,
        "asset": asset_symbol,
        "amount": amount,
        "dkg_claim": dkg_claim,
        "sigma_bps": sigma_bps,
        "timestamp": datetime.now(datetime.timezone.utc).isoformat() + "Z",
        "nonce": datetime.now(datetime.timezone.utc).strftime("%Y%m%d%H%M%S%f"),
    }

    blob = json.dumps(envelope, sort_keys=True, separators=(",", ":")).encode("utf-8")
    meta_hash = "0x" + hashlib.sha256(blob).hexdigest()
    return envelope, meta_hash


def _pad_32Bytes(hex_without_0x: str) -> str:
    """
    Pad on the left up to 32 bytes (64 hex characters).
    """
    return hex_without_0x.rjust(64, "0")


def mock_encode_x402_call_data(
    guild_address: str,
    raw_amount: int,
    meta_hash: str,
) -> str:
    """
    Mock EVM call encoding:
    x402Pay(address guild, uint256 amount, bytes32 metaHash)

    Structure:
    - 4 bytes selector
    - 32 bytes address
    - 32 bytes amount
    - 32 bytes metaHash
    """
    selector = "deadbeef"  # MOCK: en prod keccak("x402Pay(address,uint256,bytes32)")[:4]

    guild_clean = guild_address.lower().replace("0x", "")
    guild_padded = _pad_32Bytes(guild_clean)

    amount_hex = hex(raw_amount)[2:]
    amount_padded = _pad_32Bytes(amount_hex)

    meta_clean = meta_hash.lower().replace("0x", "")
    meta_padded = _pad_32Bytes(meta_clean)

    # EVM call : 0x + selector + padded args
    return "0x" + selector + guild_padded + amount_padded + meta_padded


# ============================================================
# 3. Principal Function  — Payment x402 via XCM V3
# ============================================================

def tool_execute_xcm_payment_x402(
    source_chain: str,
    source_pay: str,
    target_parachain_id: int,
    guild_address: str,
    amount: float,
    asset_symbol: str,
    dkg_claim: str,
    sigma_bps: int,
    x402_contract_address: str,
) -> Dict[str, Any]:
    """
    Generates a complete x402 payment:
      - x402 envelope (hashed)
      - EVM call x402Pay(...)
      - XCM V3 source programme (WithdrawAsset + InitiateReserveWithdraw)
      - XCM V3 destination programme (ReserveAssetDeposited + BuyExecution + Transact)
    """

    print("\n[Sancho-x402] 🔧 Payment construction x402...")

    # ----------------------------------------------------
    # 1. Building the x402 envelope
    # ----------------------------------------------------
    envelope, meta_hash = build_x402_envelope(
        pay_chain=source_chain,
        pay=source_pay,
        guild=guild_address,
        asset_symbol=asset_symbol,
        amount=amount,
        dkg_claim=dkg_claim,
        sigma_bps=sigma_bps,
    )

    # ----------------------------------------------------
    # 2. Convert amount → planks
    # ----------------------------------------------------
    decimals = 12 if asset_symbol == "TRAC" else 18
    raw_amount = int(amount * (10 ** decimals))

    # ----------------------------------------------------
    # 3. Encode the EVM call
    # ----------------------------------------------------
    calldata = mock_encode_x402_call_data(
        guild_address=guild_address,
        raw_amount=raw_amount,
        meta_hash=meta_hash,
    )

    # ----------------------------------------------------
    # 4. XCM Source Program (Moonbeam → NeuroWeb)
    # ----------------------------------------------------
    xcm_source = [
        { "WithdrawAsset": [ concrete_here_asset(raw_amount) ] },
        {
            "InitiateReserveWithdraw": {
                "assets": [ concrete_here_asset(raw_amount) ],
                "reserve": multilocation_parachain(target_parachain_id),
                "xcm": "Here"
            }
        }
    ]

    # ----------------------------------------------------
    # 5. XCM Destination Program (NeuroWeb)
    # ----------------------------------------------------
    buy_execution_fees = int(0.02 * raw_amount)

    xcm_destination = [
        { "ReserveAssetDeposited": [ concrete_here_asset(raw_amount) ] },
        {
            "BuyExecution": {
                "fees": concrete_here_asset(buy_execution_fees),
                "weight_limit": {
                    "Limited": {
                        "ref_time": 2_000_000_000,
                        "proof_size": 32_000
                    }
                }
            }
        },
        {
            "Transact": {
                "origin_kind": "Native",
                "require_weight_at_most": {
                    "ref_time": 1_000_000_000,
                    "proof_size": 16_000
                },
                "call": {
                    "EvmCall": {
                        "to": x402_contract_address,
                        "input": calldata
                    }
                }
            }
        }
    ]

    # ----------------------------------------------------
    # 6. HRMP simulation (POC)
    # ----------------------------------------------------
    print("[Sancho-x402] 🚀 Sending via HRMP... (simulation)")
    time.sleep(1.0)

    tx_hash = f"0x{abs(hash(datetime.now(datetime.timezone.utc).isoformat())):064x}"

    # ----------------------------------------------------
    # 7. Build receipt
    # ----------------------------------------------------
    receipt = {
        "status": "success",
        "protocol": "x402",
        "xcm_version": "V3",
        "source_parachain": source_chain,
        "destination_parachain": target_parachain_id,
        "pay": source_pay,
        "guild": guild_address,
        "asset": asset_symbol,
        "amount_raw": raw_amount,
        "x402_envelope": envelope,
        "x402_meta_hash": meta_hash,
        "evm_target_contract": x402_contract_address,
        "evm_calldata": calldata,
        "xcm_source_program": xcm_source,
        "xcm_destination_program": xcm_destination,
        "tx_hash": tx_hash,
        "timestamp": datetime.now(datetime.timezone.utc).isoformat() + "Z",
    }

    print(f"[Sancho-x402] ✅ Paiement x402 simulé. Hash={tx_hash[:10]}…")
    return receipt