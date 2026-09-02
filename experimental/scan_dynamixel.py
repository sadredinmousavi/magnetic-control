from dynamixel_sdk import PortHandler, PacketHandler

DEVICENAME = "/dev/ttyS0"   # change if needed
COMMON_BAUDRATES = [9600, 57600, 115200, 1000000]
PROTOCOLS = [1.0, 2.0]
ID_RANGE = range(0, 21)   # scan IDs 0 to 20

for baud in COMMON_BAUDRATES:
    print(f"\n=== Trying baudrate: {baud} ===")
    portHandler = PortHandler(DEVICENAME)

    if not portHandler.openPort():
        print("Failed to open port")
        quit()

    if not portHandler.setBaudRate(baud):
        print(f"Failed to set baudrate {baud}")
        portHandler.closePort()
        continue

    found = False

    for protocol in PROTOCOLS:
        packetHandler = PacketHandler(protocol)
        print(f"  Protocol {protocol}")

        for dxl_id in ID_RANGE:
            model_number, comm_result, error = packetHandler.ping(portHandler, dxl_id)

            if comm_result == 0 and error == 0:
                print(f"    FOUND: ID={dxl_id}, Protocol={protocol}, Baud={baud}, Model={model_number}")
                found = True

    portHandler.closePort()

    if not found:
        print("  No motor found at this baudrate")
