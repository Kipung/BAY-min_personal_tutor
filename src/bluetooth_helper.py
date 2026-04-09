import asyncio
import logging
import math
import random
import struct
import time

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
SERVICE_UUID =          "2906ab5c-18f2-4a89-b4a9-05a1e7f57ec5"
CHAR_UUID =             "df1b5230-7bcb-4930-8d9f-ca7865dae4c7"
CHALLENGE_CHAR_UUID =   "7e57417e-84bd-4fc8-9568-08e44d81e839"
ANTENNA_CHAR_UUID =     "37f02fcb-5045-42d9-96b8-3f893402f607"

CONNECTION_POLL_INTERVAL = 1.0
ANTENNA_POLL_INTERVAL    = 0.25

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
    uid_buffer: list[str] = []
    # -----------------------------------------------------------------------
    # Write callback: Flutter app sent us a UID
    # -----------------------------------------------------------------------
    def on_write_request(characteristic: BlessGATTCharacteristic, value, **kwargs):
        uuid = str(characteristic.uuid).lower()

        if uuid != CHAR_UUID.lower():
            return

        try: 
            text = bytes(value).decode("utf-8").strip()
            print(text)
        except UnicodeDecodeError:
            print(f"[ble] Received non-UTF8 data: {value}")
            return
        
        if text == "READY" and not ready_event.is_set():
            print("[ble] Received READY signal from Flutter app. Sending challenge and starting antenna broadcast.")
            ready_event.set()
        elif len(received_uid) == 0:
            uid_buffer.append(text)
            assembled = "".join(uid_buffer)
            if len(assembled) >= 28:
                received_uid.append(assembled)
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
    time.sleep(0.5)
    challenge = [random.random() * math.pi * -1, random.random() * math.pi]
    while abs(challenge[0] + challenge[1]) < math.pi/6:
        challenge[1] = random.random() * math.pi
    char = server.get_characteristic(CHALLENGE_CHAR_UUID)
    char.value = bytearray(struct.pack("ff", *challenge))
    server.update_value(SERVICE_UUID, CHALLENGE_CHAR_UUID)

    mini.disable_motors(ids=["left_antenna", "right_antenna"])
    async def _broadcast_antenna_positions():
        antenna_char = server.get_characteristic(ANTENNA_CHAR_UUID)
        while not uid_received_event.is_set():
            left, right = get_antenna_positions(mini)
            antenna_char.value = bytearray(struct.pack("ff", left, right))
            try:
                server.update_value(SERVICE_UUID, ANTENNA_CHAR_UUID)
            except Exception as e:
                print(f"[ble] Antenna notify failed: {e}")
            await asyncio.sleep(ANTENNA_POLL_INTERVAL)
    broadcast_task = asyncio.create_task(_broadcast_antenna_positions())

    await uid_received_event.wait()

    mini.goto_target(antennas=challenge, duration=1.0)
    mini.enable_motors(ids=["left_antenna", "right_antenna"])
    mini.goto_target(antennas=[-0.15, 0.15], duration=2.0)

    char = server.get_characteristic(CHAR_UUID)
    char.value = bytearray("ACK".encode("utf-8"))
    server.update_value(SERVICE_UUID, CHAR_UUID)

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
                connected = await server.is_connected()
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

def get_antenna_positions(mini: ReachyMini) -> tuple[float, float]:
    if mini is None:
        t = time.time()
        left  = math.pi*-0.5 - math.sin(t * 0.5) * math.pi * -0.5   # oscillates in left range
        right = math.pi*0.5 - math.sin(t * 0.3) * math.pi *  0.5    # oscillates in right range
        return left, right
    left, right = mini.get_present_antenna_joint_positions()
    return left, right