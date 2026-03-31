"""
BLE replacement for the Firebase active_user polling function.

Usage:
    from ble_active_user import wait_for_active_user_async

    uid, disconnected_event = await wait_for_active_user_async()
    # ... robot does its thing ...
    await disconnected_event.wait()
    # Flutter app disconnected; log out / clean up

The Flutter app (screen_test_bluetooth.dart) scans, connects, and writes
the user's UID string to the first writable characteristic it finds.
This function advertises as a BLE peripheral, receives that write, sends
an ACK notification back, and returns. The server keeps running in the
background; disconnected_event is set when the connection drops.

Requirements:
    pip install bless

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

DEVICE_NAME  = "PythonBLE"
SERVICE_UUID = "12345678-1234-5678-1234-56789abcdef0"
CHAR_UUID    = "12345678-1234-5678-1234-56789abcdef1"

# How often (seconds) the background task checks if the device is still connected.
CONNECTION_POLL_INTERVAL = 1.0

logging.basicConfig(level=logging.WARNING)  # Suppress bless noise by default

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def wait_for_active_user_async(
    poll_interval: float = 1.0,
) -> tuple[str, asyncio.Event]:
    """
    Advertises as a BLE peripheral and blocks until the Flutter app
    connects and writes a UID.

    Returns (active_user, disconnected_event).

    - active_user       : the UID string written by the Flutter app.
    - disconnected_event: an asyncio.Event that is set when the BLE
                          connection is broken (app closes, moves out of
                          range, user logs out, etc.). The background
                          server task stops itself automatically after
                          setting the event.

    poll_interval is accepted for API compatibility but is unused here;
    BLE notification replaces polling.
    """

    received_event     = asyncio.Event()
    disconnected_event = asyncio.Event()
    received_uid: list[str] = []  # mutable container for closure

    # We need a reference to the server inside the write callback so we
    # can push the ACK notification, but the server isn't created yet.
    # A list gives us a late-binding handle without nonlocal gymnastics.
    server_ref: list[BlessServer] = []

    # -----------------------------------------------------------------------
    # Write callback: Flutter app sent us a UID
    # -----------------------------------------------------------------------
    def on_write_request(characteristic: BlessGATTCharacteristic, value, **kwargs):
        try:
            uid = bytes(value).decode("utf-8").strip()
            if uid and not received_uid:           # only accept the first write
                received_uid.append(uid)

                # Send ACK notification so Flutter can pop the screen
                characteristic.value = "ACK".encode("utf-8")
                if server_ref:
                    server_ref[0].update_value(SERVICE_UUID, CHAR_UUID)

                received_event.set()
        except Exception as e:
            print(f"[ble] Failed to decode incoming value: {e}")

    # -----------------------------------------------------------------------
    # Build and start the GATT server
    # -----------------------------------------------------------------------
    server = BlessServer(name=DEVICE_NAME)
    server_ref.append(server)
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

    # -----------------------------------------------------------------------
    # Wait for the Flutter app to write the UID
    # -----------------------------------------------------------------------
    await received_event.wait()

    active_user = received_uid[0]
    print(f"[state] Bluetooth connected. Active user: {active_user}")

    # -----------------------------------------------------------------------
    # Background task: watch for disconnection, then clean up
    # -----------------------------------------------------------------------
    async def _watch_connection():
        # Give the stack a moment after the write before we start monitoring;
        # is_connected() may briefly return False right after the initial write.
        await asyncio.sleep(CONNECTION_POLL_INTERVAL * 2)

        while True:
            try:
                connected = server.is_connected()
            except Exception:
                connected = False

            if not connected:
                print("[state] Bluetooth disconnected. Cleaning up BLE server.")
                disconnected_event.set()
                await server.stop()
                return

            await asyncio.sleep(CONNECTION_POLL_INTERVAL)

    asyncio.create_task(_watch_connection())

    return active_user, disconnected_event