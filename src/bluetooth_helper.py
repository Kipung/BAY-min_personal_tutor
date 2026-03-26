##TEMPORARY## - emulating bluetooth via Firebase

import asyncio
import time
import firebase_admin
from firebase_admin import credentials, firestore


async def wait_for_active_user_async(poll_interval: float = 1.0) -> str:
    """Async version of wait_for_active_user — use inside async contexts."""
    if not firebase_admin._apps:
        creds = credentials.Certificate("credentials.json")
        firebase_admin.initialize_app(creds)

    db = firestore.client()
    doc_ref = db.collection("reachy-mini").document("reachy1")

    print("[state] State 1: Waiting for Bluetooth connection (active_user)...")
    while True:
        doc = doc_ref.get()
        if doc.exists:
            active_user = doc.to_dict().get("active_user")
            if active_user:
                print(f"[state] Bluetooth connected. Active user: {active_user}")
                return active_user
        await asyncio.sleep(poll_interval)
