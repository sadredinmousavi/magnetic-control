import helpers_sdkver as h

h.connect("/dev/serial0", 1000000)

h.setLedStatus(2, 1)

h.move(5, 0, 150)

h.moveArray([0, 10, -10, 20, -20], wait=2, zeroAng=150)

h.disconnect()
