"""Read MX-12W status data through a U2D2 without changing servo settings."""

import argparse

from dynamixel_sdk import COMM_SUCCESS, PacketHandler, PortHandler


PROTOCOL_VERSION = 1.0

REGISTERS = (
    ("torque_enabled", 24, 1),
    ("led", 25, 1),
    ("present_position", 36, 2),
    ("present_speed_raw", 38, 2),
    ("present_load_raw", 40, 2),
    ("input_voltage_raw", 42, 1),
    ("temperature_c", 43, 1),
    ("moving", 46, 1),
    ("status_return_level", 16, 1),
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", default="COM5", help="U2D2 port (default: COM5)")
    parser.add_argument(
        "--baud", type=int, default=1_000_000, help="Baud rate (default: 1000000)"
    )
    parser.add_argument(
        "--ids",
        type=int,
        nargs="+",
        default=list(range(1, 9)),
        help="Servo IDs to query (default: 1 2 3 4 5 6 7 8)",
    )
    return parser.parse_args()


def check_result(packet, comm_result, servo_error):
    if comm_result != COMM_SUCCESS:
        raise RuntimeError(packet.getTxRxResult(comm_result))
    if servo_error:
        raise RuntimeError(packet.getRxPacketError(servo_error))


def read_register(port, packet, servo_id, address, size):
    if size == 1:
        value, comm_result, servo_error = packet.read1ByteTxRx(
            port, servo_id, address
        )
    else:
        value, comm_result, servo_error = packet.read2ByteTxRx(
            port, servo_id, address
        )

    check_result(packet, comm_result, servo_error)
    return value


def main():
    args = parse_args()
    port = PortHandler(args.port)
    packet = PacketHandler(PROTOCOL_VERSION)

    if not port.openPort():
        raise RuntimeError(
            f"Could not open {args.port}. Close DYNAMIXEL Wizard and other serial apps."
        )

    try:
        if not port.setBaudRate(args.baud):
            raise RuntimeError(f"Could not set {args.baud} baud on {args.port}.")

        print(
            f"Reading MX-12W servos on {args.port}, {args.baud} baud, "
            f"Protocol {PROTOCOL_VERSION:.1f}"
        )

        found = 0
        for servo_id in args.ids:
            model, comm_result, servo_error = packet.ping(port, servo_id)
            if comm_result != COMM_SUCCESS:
                print(
                    f"\nID {servo_id}: no response "
                    f"({packet.getTxRxResult(comm_result)})"
                )
                continue
            if servo_error:
                print(
                    f"\nID {servo_id}: servo error "
                    f"({packet.getRxPacketError(servo_error)})"
                )
                continue

            found += 1
            print(f"\nID {servo_id}: FOUND, model number {model}")

            values = {}
            for name, address, size in REGISTERS:
                try:
                    values[name] = read_register(
                        port, packet, servo_id, address, size
                    )
                except RuntimeError as error:
                    values[name] = f"READ ERROR: {error}"

            position = values["present_position"]
            if isinstance(position, int):
                values["present_position_degrees"] = position * 360.0 / 4095.0

            voltage = values["input_voltage_raw"]
            if isinstance(voltage, int):
                values["input_voltage_v"] = voltage / 10.0

            for name, value in values.items():
                if isinstance(value, float):
                    print(f"  {name}: {value:.2f}")
                else:
                    print(f"  {name}: {value}")

        print(f"\nFinished: {found}/{len(args.ids)} requested servo(s) responded.")
        if not found:
            print("Try --baud 57600; that is the MX-12W factory default.")
    finally:
        port.closePort()


if __name__ == "__main__":
    main()
