"""Exercise a USB-serial adapter's TX pin for voltage/scope testing.

Disconnect the servo before running this test. Measure between the adapter's
TX pin and GND. Press Ctrl+C to stop safely.
"""

import argparse
import time

import serial


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", default="COM3", help="Serial port (default: COM3)")
    parser.add_argument(
        "--baud", type=int, default=1_000_000, help="Baud rate (default: 1000000)"
    )
    parser.add_argument(
        "--seconds", type=float, default=3.0, help="Seconds per test phase"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    port = serial.Serial(
        args.port,
        baudrate=args.baud,
        bytesize=serial.EIGHTBITS,
        parity=serial.PARITY_NONE,
        stopbits=serial.STOPBITS_ONE,
        timeout=1,
        write_timeout=1,
    )

    print(f"Opened {port.name} at {port.baudrate} baud (8-N-1).")
    print("Measure TX relative to GND. Press Ctrl+C to stop.")

    try:
        while True:
            print(f"TX idle HIGH for {args.seconds:g} seconds")
            time.sleep(args.seconds)

            for value, description in (
                (0x00, "mostly LOW (expect about 10% of logic voltage)"),
                (0xFF, "mostly HIGH (expect about 90% of logic voltage)"),
                (0x55, "alternating bits (expect about 50% of logic voltage)"),
            ):
                print(
                    f"Sending 0x{value:02X}, {description}, "
                    f"for {args.seconds:g} seconds"
                )
                deadline = time.monotonic() + args.seconds
                block = bytes([value]) * 4096
                while time.monotonic() < deadline:
                    port.write(block)
                port.flush()

            port.reset_input_buffer()
            message = b"RX_TX_LOOPBACK_TEST\r\n"
            port.write(message)
            port.flush()
            received = port.read(len(message))
            if received == message:
                print("Loopback PASS: RX received exactly what TX sent")
            else:
                print(
                    "Loopback not detected. To test RX, disconnect the servo "
                    "and temporarily connect TX directly to RX."
                )

    except KeyboardInterrupt:
        print("\nTest stopped.")
    finally:
        port.close()


if __name__ == "__main__":
    main()
