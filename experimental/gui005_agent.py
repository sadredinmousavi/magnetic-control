"""Launch the unchanged GUI005 with an optional local agent mailbox attached."""

import tkinter as tk
import argparse

from gui005 import ClosedLoopGUI
from agent_control import AgentControl


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--camera-delay', type=float, default=1.,
                        help='Estimated source-to-GUI delay in seconds; not a measured capture timestamp.')
    options = parser.parse_args()
    root = tk.Tk()
    app = ClosedLoopGUI(root)
    bridge = AgentControl(app, camera_delay_s=options.camera_delay)
    root.title(f'First Paper Closed-Loop Control — camera delay estimate {options.camera_delay:g} s')

    def close():
        bridge.close()
        app.on_close()

    root.protocol('WM_DELETE_WINDOW', close)
    print('GUI005 local control ready. The camera and serial controller connect through the usual GUI controls.', flush=True)
    root.mainloop()


if __name__ == '__main__':
    main()
