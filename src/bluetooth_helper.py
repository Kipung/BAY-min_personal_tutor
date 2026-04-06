import asyncio
import logging
import math
import random
import struct

from bless import (
    BlessServer,
    BlessGATTCharacteristic,
    GATTCharacteristicProperties,
    GATTAttributePermissions,
)
from reachy_mini import ReachyMini

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEVICE_NAME = "BAY-min"
SERVICE_UUID =          "12345678-1234-5678-1234-56789abcdef0"
CHAR_UUID =             "12345678-1234-5678-1234-56789abcdef1"
CHALLENGE_CHAR_UUID =   "12345678-1234-5678-1234-56789abcdef2"
ANTENNA_CHAR_UUID =     "12345678-1234-5678-1234-56789abcdef3"

CONNECTION_POLL_INTERVAL = 1.0
ANTENNA_POLL_INTERVAL    = 0.5

logging.basicConfig(level=logging.WARNING)  # Suppress bless noise by default

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def wait_for_active_user_async(mini: ReachyMini) -> tuple[str, asyncio.Event]:
    """
    1. Flutter connects
    2. Flutter subscribes to all notify characteristics
    3. Flutter writes "READY" to CHAR_UUID
    4. Pi receives "READY" → sends challenge
    5. Pi starts broadcasting antenna positions
    6. User completes challenge → Flutter writes UID to CHAR_UUID
    7. Pi sends ACK
    """

    ready_event = asyncio.Event()
    uid_received_event = asyncio.Event()

    received_uid: list[str] = []

    # -----------------------------------------------------------------------
    # Write callback: Flutter app sent us a UID
    # -----------------------------------------------------------------------
    def on_write_request(characteristic: BlessGATTCharacteristic, value, **kwargs):
        uuid = str(characteristic.uuid).lower()

        if uuid != CHAR_UUID.lower():
            return

        try: 
            text = bytes(value).decode("utf-8").strip()
        except UnicodeDecodeError:
            print(f"[ble] Received non-UTF8 data: {value}")
            return
        
        if text == "READY" and not ready_event.is_set():
            print("[ble] Received READY signal from Flutter app. Sending challenge and starting antenna broadcast.")
            ready_event.set()

        elif text and not received_uid:
            received_uid.append(text)
            uid_received_event.set()

    # -----------------------------------------------------------------------
    # Build and start the GATT server
    # -----------------------------------------------------------------------
    server = BlessServer(name=DEVICE_NAME)
    server.write_request_func = on_write_request

    await server.add_new_service(SERVICE_UUID)

    # Add the main characteristic for receiving the UID and sending ACK
    await server.add_new_characteristic(
        SERVICE_UUID,
        CHAR_UUID,
        (
            GATTCharacteristicProperties.write
            | GATTCharacteristicProperties.notify
        ),
        None,
        GATTAttributePermissions.writeable,
    )

    # Add a characteristic for the authentication challenge
    await server.add_new_characteristic(
        SERVICE_UUID,
        CHALLENGE_CHAR_UUID,
        GATTCharacteristicProperties.notify,
        bytearray(2),
        GATTAttributePermissions.writeable
    )

    # Add a characteristic for the antenna state
    await server.add_new_characteristic(
        SERVICE_UUID,
        ANTENNA_CHAR_UUID,
        GATTCharacteristicProperties.notify,
        bytearray([0, 0]),
        GATTAttributePermissions.readable,
    )

    await server.start()
    print("[state] State 1: Waiting for Bluetooth connection (active_user)...")

    await ready_event.wait()
    challenge = [random.random() * math.pi for _ in range(2)]
    while math.abs(challenge[0] - challenge[1]) < math.pi/6:
        challenge = [random.random() * math.pi for _ in range(2)]
    char = server.get_characteristic(SERVICE_UUID, CHALLENGE_CHAR_UUID)
    payload = bytearray(struct.pack("ff", *challenge))
    char.value = payload
    await server.notify(SERVICE_UUID, CHALLENGE_CHAR_UUID)
    
    async def _broadcast_antenna_positions():
        antenna_char = server.get_characteristic(ANTENNA_CHAR_UUID)
        while not uid_received_event.is_set():
            left, right = mini.get_present_antenna_joint_positions()
            antenna_char.value = bytearray([left, right])
            try:
                await server.notify(SERVICE_UUID, ANTENNA_CHAR_UUID)
            except Exception as e:
                print(f"[ble] Antenna notify failed: {e}")
            await asyncio.sleep(ANTENNA_POLL_INTERVAL)
    broadcast_task = asyncio.create_task(_broadcast_antenna_positions())

    await uid_received_event.wait()

    char = server.get_characteristic(SERVICE_UUID, CHAR_UUID)
    char.value = bytearray("ACK".encode("utf-8"))
    await server.notify(SERVICE_UUID, CHAR_UUID)

    active_user = received_uid[0]
    print(f"[state] Bluetooth connected with UID: {active_user}")

    # -----------------------------------------------------------------------
    # Background task: watch for disconnection, then clean up
    # -----------------------------------------------------------------------
    disconnected_event = asyncio.Event()

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