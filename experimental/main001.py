import serial
from helpers import factoryReset, move, moveArray, setID, setLedStatus

# port = serial.Serial("/dev/ttyS0")
port = serial.Serial("/dev/serial0")
port.baudrate = 1000000
# port.timeout = 3


# setLedStatus(1,0)
# setLedStatus(2,0)
# setLedStatus(3,0)
# setLedStatus(4,0)


moveArray(port, [-85.28, 68.47, -85.25, 5.16, -8.61, -4.22, -1.85, 6.38], 2, 180)
moveArray(port, [61.80, -72.84, -13.21, -2.42, -1.42, -1.69, -5.84, -73.66], 2, 180)
moveArray(port, [-5.03, -8.06, -5.01, 1.94, -85.25, 68.46, -85.25, 1.35], 2, 180)
moveArray(port, [-2.22, -27.06, -68.26, 57.46, -72.60, -4.45, -11.66, -8.95], 2, 180)
moveArray(
    port, [-18.00, -18.00, -18.00, -18.00, -18.00, -18.00, -18.00, -18.00], 2, 180
)
