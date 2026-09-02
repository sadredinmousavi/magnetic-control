from dynamixel_sdk import PortHandler, PacketHandler

DEVICENAME = "/dev/ttyS0"
BAUDRATE = 1000000
PROTOCOL_VERSION = 1.0
DXL_ID = 4

ADDR_TORQUE_ENABLE = 24
ADDR_GOAL_POSITION = 30

TORQUE_ENABLE = 1
GOAL_POSITION_90_DEG = int((90.0 / 300.0) * 1023)  # about 306

portHandler = PortHandler(DEVICENAME)
packetHandler = PacketHandler(PROTOCOL_VERSION)

if not portHandler.openPort():
    print("Failed to open port")
    quit()

if not portHandler.setBaudRate(BAUDRATE):
    print("Failed to set baudrate")
    quit()

dxl_comm_result, dxl_error = packetHandler.write1ByteTxRx(
    portHandler, DXL_ID, ADDR_TORQUE_ENABLE, TORQUE_ENABLE
)

if dxl_comm_result != 0:
    print("Comm error:", packetHandler.getTxRxResult(dxl_comm_result))
elif dxl_error != 0:
    print("Packet error:", packetHandler.getRxPacketError(dxl_error))
else:
    print("Torque enabled")

dxl_comm_result, dxl_error = packetHandler.write2ByteTxRx(
    portHandler, DXL_ID, ADDR_GOAL_POSITION, GOAL_POSITION_90_DEG
)

if dxl_comm_result != 0:
    print("Comm error:", packetHandler.getTxRxResult(dxl_comm_result))
elif dxl_error != 0:
    print("Packet error:", packetHandler.getRxPacketError(dxl_error))
else:
    print(f"Moved ID {DXL_ID} to about 90 degrees (goal {GOAL_POSITION_90_DEG})")

portHandler.closePort()
