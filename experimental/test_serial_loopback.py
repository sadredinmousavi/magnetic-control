"""Test a USB-serial adapter by looping its TX pin back to its RX pin."""

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
        "--cycles", type=int, default=20, help="Number of test cycles (default: 20)"
    )
    return parser.parse_args()


def read_exact(port, size, timeout=1.0):
    received = bytearray()
    deadline = time.monotonic() + timeout

    while len(received) < size and time.monotonic() < deadline:
        chunk = port.read(size - len(received))
        if chunk:
            received.extend(chunk)

    return bytes(received)


def main():
    args = parse_args()
    patterns = [
        b"Hello from COM3!\r\n",
        bytes([0x00, 0xFF, 0x55, 0xAA]),
        bytes(range(256)),
    ]

    print("Disconnect the servo and connect adapter TX directly to adapter RX.")
    print(f"Opening {args.port} at {args.baud} baud (8-N-1)...")

    with serial.Serial(
        args.port,
        baudrate=args.baud,
        bytesize=serial.EIGHTBITS,
        parity=serial.PARITY_NONE,
        stopbits=serial.STOPBITS_ONE,
        timeout=0.05,
        write_timeout=1,
    ) as port:
        passed = 0
        total = args.cycles * len(patterns)

        for cycle in range(1, args.cycles + 1):
            for pattern_number, sent in enumerate(patterns, start=1):
                port.reset_input_buffer()
                port.write(sent)
                port.flush()
                received = read_exact(port, len(sent))

                if received == sent:
                    passed += 1
                    print(
                        f"Cycle {cycle:02d}, pattern {pattern_number}: "
                        f"PASS ({len(sent)} bytes)"
                    )
                else:
                    print(
                        f"Cycle {cycle:02d}, pattern {pattern_number}: FAIL\n"
                        f"  sent:     {sent.hex(' ')}\n"
                        f"  received: {received.hex(' ')}"
                    )

    print(f"\nResult: {passed}/{total} patterns passed.")
    if passed == total:
        print("LOOPBACK PASS: TX and RX work at the selected baud rate.")
    else:
        print("LOOPBACK FAIL: check the TX-RX jumper, port, driver, or baud rate.")


if __name__ == "__main__":
    main()
