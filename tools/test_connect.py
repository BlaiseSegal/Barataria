# test_connect.py
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.dkg_config import get_dkg_client
import json

print("--- DIAGNOSTIC POC BARATARIA ---")
try:
    client = get_dkg_client()
    print("📡 Client initialisé. Tentative de contact avec le Nœud...")
    
    info = client.node.info
    
    print("\n✅ SUCCÈS TOTAL ! Connexion établie.")
    print(f"   Version du Nœud : {info.get('version')}")
    print(f"   Peer ID : {info.get('id')}")
    
    print("   Réseau Blockchain configuré.")

except Exception as e:
    print("\n❌ ÉCHEC.")
    print(f"Erreur technique : {e}")