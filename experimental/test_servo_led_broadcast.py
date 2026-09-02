"""Blink all Dynamixel Protocol 1.0 servo LEDs using broadcast packets."""

import argparse
import time

import serial


LED_ON_PACKET = bytes.fromhex("FF FF FE 04 03 19 01 E0")
LED_OFF_PACKET = bytes.fromhex("FF FF FE 04 03 19 00 E1")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", default="COM3", help="Serial port (default: COM3)")
    parser.add_argument(
        "--baud", type=int, default=1_000_000, help="Baud rate (default: 1000000)"
    )
    parser.add_argument(
        "--cycles", type=int, default=5, help="Number of blink cycles (default: 5)"
    )
    parser.add_argument(
        "--delay", type=float, default=1.0, help="Seconds per LED state (default: 1)"
    )
    return parser.parse_args()


def send_packet(port, packet, label):
    written = port.write(packet)
    port.flush()
    print(f"{label}: wrote {written} bytes: {packet.hex(' ').upper()}")


def main():
    args = parse_args()

    print("Wiring: adapter TX -> servo DATA, adapter GND -> servo/power GND")
    print("Do not connect adapter TX to RX during this test.")
    print(f"Opening {args.port} at {args.baud} baud (8-N-1)...")

    with serial.Serial(
        args.port,
        baudrate=args.baud,
        bytesize=serial.EIGHTBITS,
        parity=serial.PARITY_NONE,
        stopbits=serial.STOPBITS_ONE,
        timeout=0.1,
        write_timeout=1,
    ) as port:
        try:
            for cycle in range(1, args.cycles + 1):
                print(f"\nCycle {cycle}/{args.cycles}")
                send_packet(port, LED_ON_PACKET, "LED ON ")
                time.sleep(args.delay)
                send_packet(port, LED_OFF_PACKET, "LED OFF")
                time.sleep(args.delay)
        finally:
            send_packet(port, LED_OFF_PACKET, "Final LED OFF")

    print("\nTest completed.")


if __name__ == "__main__":
    main()
