pip install dynamixel-controller
sudo apt-get install fswebcam
pip install opencv-python

GUI005 Commands tab
-------------------
Complete Center/+X/+Y calibration (or load a matching setup from `experimental/setups`)
and select the hardware case in Calibration. Select Live camera for command execution.
In Commands, load a file from `experimental/xy_sequences`:

`[x, y] | wait_seconds | motor_zero_degrees`

Calibration rectangle dimensions are in centimetres. Select sequence units
(`mm`, `cm`, or `m`) in Commands. `Auto` respects `# targets_xy_mm`,
`# targets_xy_cm`, or legacy `# targets_xy_m` headers, and uses millimetres
when no unit header exists. The table always displays millimetres; calculations
receive metres internally. Selecting different units reloads and reprojects the file.

The image shows the desired path in cyan and selected/current target in orange.
The table displays XY in millimetres. Commands use ratio 1 and calculate fresh
motor angles for each point; the wait starts after sending that pose.
Run requires a live camera, connected controller, complete robot axes, and every
point inside the four-pin workspace. Calibration is locked while running.
Stop cancels unsent poses and interrupts waits; it does not undo the last pose.
This is timed execution, not robot-arrival feedback. Simultaneous multi-target
rows are not supported by this tab.




https://raspberrypi-guide.github.io/electronics/using-usb-webcams
https://tutorials-raspberrypi.com/raspberry-pi-security-camera-with-webcam/



```
python -m venv .venv
source .venv/bin/activate
which python

pip install -i https://mirror-pypi.runflare.com/simple dynamixel-sdk
```
