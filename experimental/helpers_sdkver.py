import time

from dynamixel_sdk import PacketHandler, PortHandler

# ============================================================
# Dynamixel SDK settings
# ============================================================

DEVICENAME = "/dev/serial0"
BAUDRATE = 1000000
PROTOCOL_VERSION = 1.0


# ============================================================
# MX-12W / Protocol 1.0 Control Table Addresses
# ============================================================

ADDR_ID = 3
ADDR_BAUD_RATE = 4

ADDR_TORQUE_ENABLE = 24
ADDR_LED = 25
ADDR_GOAL_POSITION = 30
ADDR_MOVING_SPEED = 32
ADDR_PRESENT_POSITION = 36


# ============================================================
# Constants
# ============================================================

TORQUE_ENABLE = 1
TORQUE_DISABLE = 0

LED_ON = 1
LED_OFF = 0

MIN_ANGLE = 0
MAX_ANGLE = 300

MIN_POSITION = 0
MAX_POSITION = 1023


# ============================================================
# Global SDK objects
# ============================================================

portHandler = None
packetHandler = None
connected = False


# ============================================================
# Connection functions
# ============================================================


def connect(device=DEVICENAME, baudrate=BAUDRATE, protocol=PROTOCOL_VERSION):
    """
    Open Dynamixel SDK port globally.

    Old helpers.py probably required passing port around.
    This SDK version stores the port globally.

    Example:
        connect("/dev/serial0", 1000000)
    """

    global portHandler, packetHandler, connected
    global DEVICENAME, BAUDRATE, PROTOCOL_VERSION

    DEVICENAME = device
    BAUDRATE = baudrate
    PROTOCOL_VERSION = protocol

    portHandler = PortHandler(DEVICENAME)
    packetHandler = PacketHandler(PROTOCOL_VERSION)

    if not portHandler.openPort():
        connected = False
        raise RuntimeError(f"Failed to open port: {DEVICENAME}")

    if not portHandler.setBaudRate(BAUDRATE):
        portHandler.closePort()
        connected = False
        raise RuntimeError(f"Failed to set baudrate: {BAUDRATE}")

    connected = True
    return True


def disconnect():
    """
    Close Dynamixel SDK port.
    """

    global portHandler, packetHandler, connected

    if portHandler is not None:
        portHandler.closePort()

    portHandler = None
    packetHandler = None
    connected = False


def isConnected():
    """
    Return connection status.
    """

    return connected and portHandler is not None and packetHandler is not None


def ensureConnected():
    """
    Raise error if SDK port is not open.
    """

    if not isConnected():
        raise RuntimeError("Dynamixel SDK is not connected. Call connect(...) first.")


# ============================================================
# Conversion helpers
# ============================================================


def angle_to_position(angle):
    """
    Convert physical Dynamixel angle, 0..300 degrees, to position 0..1023.

    MX-12W / AX-style Protocol 1.0 servos use:
        0 degrees   -> 0
        300 degrees -> 1023
    """

    angle = max(MIN_ANGLE, min(MAX_ANGLE, angle))
    return int((angle / MAX_ANGLE) * MAX_POSITION)


def position_to_angle(position):
    """
    Convert Dynamixel position 0..1023 to angle 0..300 degrees.
    """

    position = max(MIN_POSITION, min(MAX_POSITION, position))
    return (position / MAX_POSITION) * MAX_ANGLE


def convertAngle(angle, zeroAng=150):
    """
    Convert your logical angle to Dynamixel physical angle.

    If your old helpers.py used:
        servo_angle = zeroAng + angle

    then this keeps the same behavior.

    Example:
        angle = 0, zeroAng = 150  -> servo physical angle = 150
        angle = 30, zeroAng = 150 -> servo physical angle = 180
        angle = -30, zeroAng = 150 -> servo physical angle = 120
    """

    servo_angle = zeroAng + angle

    if servo_angle < MIN_ANGLE:
        servo_angle = MIN_ANGLE

    if servo_angle > MAX_ANGLE:
        servo_angle = MAX_ANGLE

    return servo_angle


# ============================================================
# Low-level SDK write wrappers
# ============================================================


def write1(idm, address, value):
    """
    Write 1 byte using TxOnly.
    """

    ensureConnected()

    packetHandler.write1ByteTxOnly(portHandler, int(idm), int(address), int(value))


def write2(idm, address, value):
    """
    Write 2 bytes using TxOnly.
    """

    ensureConnected()

    packetHandler.write2ByteTxOnly(portHandler, int(idm), int(address), int(value))


# ============================================================
# Servo helper functions
# ============================================================


def enableTorque(idm):
    """
    Enable torque for one servo.
    """

    write1(idm, ADDR_TORQUE_ENABLE, TORQUE_ENABLE)


def disableTorque(idm):
    """
    Disable torque for one servo.
    """

    write1(idm, ADDR_TORQUE_ENABLE, TORQUE_DISABLE)


def setLedStatus(idm, ls):
    """
    Set LED status.

    Old style:
        setLedStatus(port, idm, ls)

    New SDK style:
        setLedStatus(idm, ls)

    ls:
        1 = ON
        0 = OFF
    """

    led_value = LED_ON if int(ls) else LED_OFF
    write1(idm, ADDR_LED, led_value)


def setID(idm, newId):
    """
    Change servo ID.

    Old style:
        setID(port, idm, newId)

    New SDK style:
        setID(idm, newId)

    WARNING:
        Usually you should only have one servo connected when changing ID.
    """

    write1(idm, ADDR_ID, int(newId))


def setSpeed(idm, speed):
    """
    Set moving speed.

    Address:
        32, 2 bytes

    speed:
        0..1023 usually.
        0 can mean maximum speed depending on Dynamixel model.
    """

    speed = max(0, min(1023, int(speed)))
    write2(idm, ADDR_MOVING_SPEED, speed)


def move(idm, angle, zeroAng=150):
    """
    Move one servo.

    Old style probably:
        move(port, idm, angle, zeroAng)

    New SDK style:
        move(idm, angle, zeroAng)

    Logical angle is converted using:
        physical_angle = zeroAng + angle

    Then:
        physical_angle 0..300 deg -> position 0..1023
    """

    ensureConnected()

    servo_angle = convertAngle(angle, zeroAng)
    goal_position = angle_to_position(servo_angle)

    enableTorque(idm)

    packetHandler.write2ByteTxOnly(
        portHandler, int(idm), ADDR_GOAL_POSITION, int(goal_position)
    )


def moveRaw(idm, angle):
    """
    Move one servo using direct physical angle 0..300.

    This does NOT apply zeroAng.
    """

    ensureConnected()

    goal_position = angle_to_position(angle)

    enableTorque(idm)

    packetHandler.write2ByteTxOnly(
        portHandler, int(idm), ADDR_GOAL_POSITION, int(goal_position)
    )


def movePosition(idm, position):
    """
    Move one servo using raw Dynamixel position 0..1023.

    This does not use angle conversion.
    """

    ensureConnected()

    position = max(MIN_POSITION, min(MAX_POSITION, int(position)))

    enableTorque(idm)

    packetHandler.write2ByteTxOnly(portHandler, int(idm), ADDR_GOAL_POSITION, position)


def moveArray(angs, wait=0, zeroAng=150):
    """
    Move multiple servos.

    Old style:
        moveArray(port, angs, wait, zeroAng)

    New SDK style:
        moveArray(angs, wait, zeroAng)

    Servo IDs are assigned automatically:
        angs[0] -> ID 1
        angs[1] -> ID 2
        angs[2] -> ID 3
        ...

    Example:
        moveArray([0, 45, -30], 2, 150)

    Means:
        servo 1 -> zeroAng + 0
        servo 2 -> zeroAng + 45
        servo 3 -> zeroAng - 30
    """

    ensureConnected()

    for index, angle in enumerate(angs):
        idm = index + 1
        move(idm, angle, zeroAng)

    if wait:
        time.sleep(wait)


def moveArrayWithIds(ids, angs, wait=0, zeroAng=150):
    """
    Move multiple servos using explicit IDs.

    Example:
        moveArrayWithIds([2, 5, 7], [0, 30, -30], 2, 150)
    """

    ensureConnected()

    if len(ids) != len(angs):
        raise ValueError("ids and angs must have the same length")

    for idm, angle in zip(ids, angs):
        move(idm, angle, zeroAng)

    if wait:
        time.sleep(wait)


def ledArray(ids, ls):
    """
    Set LED status for multiple servos.

    Example:
        ledArray([1, 2, 3], 1)
    """

    ensureConnected()

    for idm in ids:
        setLedStatus(idm, ls)


def torqueArray(ids, enable=1):
    """
    Enable or disable torque for multiple servos.

    Example:
        torqueArray([1, 2, 3], 1)
        torqueArray([1, 2, 3], 0)
    """

    ensureConnected()

    for idm in ids:
        if enable:
            enableTorque(idm)
        else:
            disableTorque(idm)


# ============================================================
# Optional read helpers
# ============================================================


def readPosition(idm):
    """
    Read present position from servo.

    Returns raw position 0..1023.
    """

    ensureConnected()

    position, dxl_comm_result, dxl_error = packetHandler.read2ByteTxRx(
        portHandler, int(idm), ADDR_PRESENT_POSITION
    )

    if dxl_comm_result != 0:
        raise RuntimeError(packetHandler.getTxRxResult(dxl_comm_result))

    if dxl_error != 0:
        raise RuntimeError(packetHandler.getRxPacketError(dxl_error))

    return position


def readAngle(idm):
    """
    Read present position and convert to physical angle 0..300.
    """

    position = readPosition(idm)
    return position_to_angle(position)
