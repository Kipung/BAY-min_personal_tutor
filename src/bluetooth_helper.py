##TEMPORARY## - emulating bluetooth via Firebase

import time
import firebase_admin
from firebase_admin import credentials, firestore


def wait_for_active_user(poll_interval: float = 1.0) -> str:
    """
    Simulates waiting for a Bluetooth connection and receiving the active user.

    Polls the Firestore document 'reachy-mini/reachy1' until the 'active_user'
    field is non-null, then returns its value.

    Args:
        poll_interval: Seconds to wait between polls (default 1.0).

    Returns:
        The active_user string once it is set.
    """
    if not firebase_admin._apps:
        creds = credentials.Certificate("credentials.json")
        firebase_admin.initialize_app(creds)

    db = firestore.client()
    doc_ref = db.collection("reachy-mini").document("reachy1")

    print("Waiting for Bluetooth connection (active_user)...")
    while True:
        doc = doc_ref.get()
        if doc.exists:
            active_user = doc.to_dict().get("active_user")
            if active_user:
                print(f"Bluetooth connected. Active user: {active_user}")
                return active_user
        time.sleep(poll_interval)
