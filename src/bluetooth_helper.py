"""
BLE replacement for the Firebase active_user polling function.

Drop-in usage (identical call signature):
    from ble_active_user import wait_for_active_user_async

    uid = await wait_for_active_user_async()

The Flutter app (screen_test_bluetooth.dart) scans, connects, and writes
the user's UID string to the first writable characteristic it finds.
This function advertises as a BLE peripheral, receives that write, and
returns the UID — mirroring the Firebase version's behavior exactly.

Platform notes:
    Linux  : Requires BlueZ >= 5.43. Run with sudo if needed.
    macOS  : Works natively via CoreBluetooth.
    Windows: Requires Windows 10 build 16299+.
"""

import asyncio
import logging

from bless import (
    BlessServer,
    BlessGATTCharacteristic,
    GATTCharacteristicProperties,
    GATTAttributePermissions,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEVICE_NAME  = "BAY-min"
SERVICE_UUID = "12345678-1234-5678-1234-56789abcdef0"
CHAR_UUID    = "12345678-1234-5678-1234-56789abcdef1"

logging.basicConfig(level=logging.WARNING)  # Suppress bless noise by default

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def wait_for_active_user_async() -> str:
    """
    BLE drop-in replacement for the Firebase polling version.

    Advertises as a BLE peripheral and blocks until the Flutter app
    connects and writes a UID. Returns the UID string.

    poll_interval is accepted for API compatibility but is unused;
    BLE uses an event instead of polling.
    """
    received_event = asyncio.Event()
    received_uid: list[str] = []  # List used as a mutable container for the callback closure

    # In on_write_request, after received_event.set(), send the ACK:
    def on_write_request(characteristic: BlessGATTCharacteristic, value, **kwargs):
        try:
            uid = bytes(value).decode("utf-8").strip()
            if uid:
                received_uid.append(uid)
                characteristic.value = "ACK".encode("utf-8")  # set the value
                server.update_value(SERVICE_UUID, CHAR_UUID)   # push notification
                received_event.set()
        except Exception as e:
            print(f"[ble] Failed to decode incoming value: {e}")

    server = BlessServer(name=DEVICE_NAME)
    server.write_request_func = on_write_request

    await server.add_new_service(SERVICE_UUID)
    await server.add_new_characteristic(
        SERVICE_UUID,
        CHAR_UUID,
        (
            GATTCharacteristicProperties.read
            | GATTCharacteristicProperties.write
            | GATTCharacteristicProperties.write_without_response
            | GATTCharacteristicProperties.notify
        ),
        None,
        (
            GATTAttributePermissions.readable
            | GATTAttributePermissions.writeable
        ),
    )

    await server.start()
    print("[state] State 1: Waiting for Bluetooth connection (active_user)...")

    try:
        await received_event.wait()
    finally:
        await server.stop()

    active_user = received_uid[0]
    print(f"[state] Bluetooth connected. Active user: {active_user}")
    return active_user