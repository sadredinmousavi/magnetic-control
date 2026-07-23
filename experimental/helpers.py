import math
import time
from time import sleep

import serial


def move(port, idm, angle):
    length = 5

    tem = divmod(angle, 22.5)
    value1 = int(tem[1] * 255 / 22.5)
    value2 = int(tem[0])

    checksum = 255 - ((idm + length + 3 + 30 + value1 + value2) % 256)

    outData = bytes([255])
    outData += bytes([255])
    outData += bytes([idm])
    outData += bytes([length])
    outData += bytes([3])
    outData += bytes([30])
    outData += bytes([value1])
    outData += bytes([value2])
    outData += bytes([checksum])

    port.write(outData)
    # ans = chr(0xFF)+chr(0xFF)+chr(idm)+chr(length)+chr(3)+chr(30)+chr(value1)+chr(value2)+chr(checksum)
    return


def setLedStatus(port, id, ls):
    #     ls=1: led ON
    #     ls=0: led OFF
    checksum = (~(id + 4 + 3 + 25 + ls)) & 0xFF
    outData = bytes([255])
    outData += bytes([255])
    outData += bytes([id])
    outData += bytes([4])
    outData += bytes([3])
    outData += bytes([25])
    outData += bytes([ls])
    outData += bytes([checksum])

    port.write(outData)
    return


def moveArray(port, angs, wait, zeroAng):
    for key, value in enumerate(angs):
        move(port, key + 1, value + zeroAng)
        sleep(0.01)
        move(port, key + 1, value + zeroAng)
        sleep(0.01)
        move(port, key + 1, value + zeroAng)
    sleep(wait)

    return


def setID(port, id, newId):
    AX_ID_LENGTH = 4
    AX_WRITE_DATA = 3
    AX_ID = 3
    checksum = (~(id + AX_ID_LENGTH + AX_WRITE_DATA + AX_ID + newId)) & 0xFF

    AX_START = 255
    outData = bytes([AX_START])
    outData += bytes([AX_START])
    outData += bytes([id])
    outData += bytes([AX_ID_LENGTH])
    outData += bytes([AX_WRITE_DATA])
    outData += bytes([AX_ID])
    outData += bytes([newId])
    outData += bytes([checksum])

    port.write(outData)
    return


def factoryReset(port, id):

    AX_RESET_LENGTH = 2
    AX_RESET = 6
    AX_START = 255

    checksum = (~(id + AX_RESET_LENGTH + AX_RESET)) & 0xFF
    outData = bytes([AX_START])
    outData += bytes([AX_START])
    outData += bytes([id])
    outData += bytes([AX_RESET_LENGTH])
    outData += bytes([AX_RESET])
    outData += bytes([checksum])

    port.write(outData)
    return


# port = serial.Serial("/dev/ttyS0")
# #port = serial.Serial("/dev/serial0")
# port.baudrate = 1000000
# # port.timeout = 3


# setLedStatus(8,0)

# setLedStatus(1,0)
# setLedStatus(2,0)
# setLedStatus(3,0)
# setLedStatus(4,0)


# moveArray([-90, -90, -90, -90, -90, -90, -90, -90],3 , 180)
# moveArray([0, 0, 0, 0, 0, 0, 0, 0],1 , 180)
# moveArray([0,0,0,0,0,0],2 , 180)

# moveArray([-33,-33,-33,-33],2 , 180)
# moveArray([-80,-15,-80,-15],2 , 180)
# moveArray([-85,+21,-85,+21],2 , 180)
# moveArray([-90,+89,-90,+89],5 , 180)
# moveArray([-85,+21,-85,+21],2 , 180)
# moveArray([-80,-15,-80,-15],2 , 180)
# moveArray([-33,-33,-33,-33],2 , 180)
# moveArray([-79,-88,-79,-88],2 , 180)
# moveArray([+32,-85,+32,-85],2 , 180)
# moveArray([+89,-90,+89,-90],5 , 180)
# moveArray([+32,-85,+32,-85],2 , 180)
# moveArray([-79,-88,-79,-88],2 , 180)
# moveArray([-33,-33,-33,-33],2 , 180)


# moveArray([-33,-33,-33,-33],5 , 180)
# moveArray([-64,-34,+8,-34],2 , 180)
# moveArray([-74,-64,+23,-4],2 , 180)
# moveArray([+85,-87,+85,-57],5 , 180)
# moveArray([+21,-82,+21,+2],2 , 180)
# # moveArray([+85,-87,+86,-59],2 , 180)
# moveArray([+25,-59,-81,-59],2 , 180)
# moveArray([-33,-33,-33,-33],2 , 180)

# moveArray([-33,-33,-33,-33],5 , 180)
# moveArray([-64,-34,+8,-34],2 , 180)
# moveArray([-74,-64,+23,-4],2 , 180)
# moveArray([-74,-73,+26,+24],2 , 180)
# moveArray([-39,-74,-39,+27],2 , 180)
# # moveArray([+3,-74,-73,+33],2 , 180)
# moveArray([+27,-39,-74,-39],2 , 180)
# moveArray([-33,-33,-33,-33],2 , 180)


# moveArray([-33.00, -33.00, -33.00, -33.00, -33.00, -33.00],2 , 180)
# moveArray([-85.00, -55.00, -15.00, -15.00, -15.00, -55.00],2 , 180)
# moveArray([-85.00, -85.00, -45.00, -45.00, -85.00, -55.00],2 , 180)
# moveArray([-55.00, -85.00, -55.00, -15.00, -15.00, -15.00],2 , 180)
# moveArray([-55.00, -85.00, -85.00, -45.00, -45.00, -85.00],2 , 180)
# moveArray([-15.00, -55.00, -85.00, -55.00, -15.00, -15.00],2 , 180)
# moveArray([-85.00, -55.00, -85.00, -85.00, -45.00, -45.00],2 , 180)
# moveArray([-15.00, -15.00, -55.00, -85.00, -55.00, -15.00],2 , 180)
# moveArray([-45.00, -85.00, -55.00, -85.00, -85.00, -45.00],2 , 180)
# moveArray([-15.00, -15.00, -15.00, -55.00, -85.00, -55.00],2 , 180)
# moveArray([-45.00, -45.00, -85.00, -55.00, -85.00, -85.00],2 , 180)
# moveArray([-55.00, -15.00, -15.00, -15.00, -55.00, -85.00],2 , 180)
# moveArray([-85.00, -45.00, -45.00, -85.00, -55.00, -85.00],2 , 180)
# moveArray([-85.00, -55.00, -15.00, -15.00, -15.00, -55.00],2 , 180)
# moveArray([-33.00, -33.00, -33.00, -33.00, -33.00, -33.00],2 , 180)


# moveArray([-33.00, -33.00, -33.00, -33.00, -33.00, -33.00],2 , 180)
# moveArray([-85.00, -33.00, -33.00, -85.00, -33.00, -33.00],2 , 180)
# moveArray([-75.00, -75.00, -33.00, -75.00, -75.00, -33.00],2 , 180)
# moveArray([-30.00, -85.00, -30.00, -30.00, -85.00, -30.00],2 , 180)
# moveArray([-30.00, -75.00, -75.00, -30.00, -75.00, -75.00],2 , 180)
# moveArray([-30.00, -30.00, -85.00, -30.00, -30.00, -85.00],2 , 180)
# moveArray([-75.00, -30.00, -75.00, -75.00, -30.00, -75.00],2 , 180)
# moveArray([-85.00, -30.00, -30.00, -85.00, -30.00, -30.00],2 , 180)
# moveArray([-33.00, -33.00, -33.00, -33.00, -33.00, -33.00],2 , 180)


##### new


# moveArray([-83,-63,-66,-83,+42,+53],2 , 180)
# moveArray([-89,+53,+53,-89,+53,+53],2 , 180)
# moveArray([-88,+52,+55,-88,+55,+52],2 , 180)
# moveArray([-89,+56,+50,-88,+49,+57],2 , 180)
# moveArray([-88,+52,+55,-88,+55,+52],2 , 180)
# moveArray([-89,+53,+53,-89,+53,+53],2 , 180)
# moveArray([-33,-33,-33,-33],3 , 180)
# moveArray([-13,-81,-82,-8,-82,-81],2 , 180)
# moveArray([+11,-84,-84,-2,-84,-84],2 , 180)
# moveArray([-6,-85,-85,+10,-85,-85],2 , 180)
# moveArray([+11,-84,-84,-2,-84,-84],2 , 180)
# moveArray([-13,-81,-82,-8,-82,-81],2 , 180)
# moveArray([-33,-33,-33,-33],3 , 180)


# moveArray([-33,-33,-33,-33, -33, -33],3 , 180)
# moveArray([-67,+62,-67,-67,-67,+62],2 , 180)
# moveArray([-84,-81,+33,-81,-84,+25],2 , 180)
# moveArray([+55,-89,+54,+54,-89,+49],2 , 180)
# moveArray([-13,-81,-82,-8,-82,-81],2 , 180)
# moveArray([+53,+53,-89,+51,+55,-89],2 , 180)
# moveArray([-78,+45,-78,-87,+12,-87],2 , 180)
# moveArray([-89,+53,+53,-89,+53,+53],2 , 180)
# moveArray([-84,-81,+33,-81,-84,+25],2 , 180)
# moveArray([+55,-89,+54,+54,-89,+49],2 , 180)
# moveArray([-13,-81,-82,-8,-82,-81],2 , 180)
# moveArray([+53,+53,-89,+51,+55,-89],2 , 180)
# moveArray([-78,+45,-78,-87,+12,-87],2 , 180)
# moveArray([-89,+53,+53,-89,+53,+53],2 , 180)
# moveArray([-33,-33,-33,-33],3 , 180)


# moveArray([-33,-33,-33,-33],3 , 180)
# moveArray([-83,-68,-59,-22,-77,-62],2 , 180)
# moveArray([-75,-80,-60,-48,-50,-67],2 , 180)
# moveArray([-54,-89,-50,-77,-35,-67],2 , 180)
# moveArray([-64,-78,-78,-61,-58,-44],2 , 180)
# moveArray([-74,-76,-75,-81,-42,-15],2 , 180)
# moveArray([-43,-38,-76,-72,-45,-40],2 , 180)
# moveArray([-36,-76,-50,-89,-54,-68],2 , 180)
# moveArray([-43,-41,-45,-72,-76,-38],2 , 180)
# moveArray([-67,-35,-77,-50,-89,-54],2 , 180)
# moveArray([-38,-44,-40,-46,-72,-76],2 , 180)
# moveArray([-50,-77,-35,-67,-54,-89],2 , 180)
# moveArray([-80,-60,-48,-50,-67,-75],2 , 180)
# moveArray([-83,-68,-59,-22,-77,-62],2 , 180)
# moveArray([-33,-33,-33,-33],3 , 180)


# moveArray([-88,+51,+55,-88,+56,+52],2 , 180)
# moveArray([-88,+52,+55,-88,+55,+52],2 , 180)
# moveArray([-88,-78,-78,-88,-30,-30],2 , 180)
# moveArray([+89,-85,-85,+90,-0,+41],2 , 180)
# moveArray([-11,-86,-78,-64,-35,-61],2 , 180)
# moveArray([-64,-78,-78,-61,-58,-44],2 , 180)
# moveArray([+43,-55,-61,+37,+43,+38],2 , 180)
# moveArray([+40,+44,+31,+59,-46,-67],2 , 180)
# moveArray([-38,-44,-40,-46,-72,-76],2 , 180)
# moveArray([-15,-59,-16,-69,-77,-86],2 , 180)
