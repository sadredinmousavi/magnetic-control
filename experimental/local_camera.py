"""Background OpenCV webcam input (including an installed Iriun virtual camera)."""
import queue
import threading
import cv2


class LocalCameraClient:
    def __init__(self, frames, events, *, capture_factory=cv2.VideoCapture):
        self.frames, self.events, self.capture_factory = frames, events, capture_factory
        self.thread = None
        self.stop_event = threading.Event()
        self.connected = False

    @property
    def is_running(self):
        return self.connected and not self.stop_event.is_set()

    def start(self, index=0, *, width=None, height=None):
        if self.thread is not None and self.thread.is_alive():
            raise RuntimeError('Webcam is still running/stopping. Wait before reconnecting.')
        if not isinstance(index, int) or index < 0:
            raise ValueError('Webcam index must be a nonnegative integer.')
        if (width is None) != (height is None):
            raise ValueError('Specify both width and height, or use Auto.')
        if width is not None and any(not isinstance(value, int) or not 1 <= value <= 16384 for value in (width, height)):
            raise ValueError('Webcam dimensions must be whole numbers between 1 and 16384.')
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._read, args=(index, width, height), daemon=True)
        self.thread.start()

    def _read(self, index, width, height):
        capture = None
        try:
            capture = self.capture_factory(index, cv2.CAP_DSHOW)
            if not capture.isOpened():
                raise RuntimeError(f'Cannot open webcam index {index}. Start the Iriun desktop/phone apps or try another index.')
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if width is not None:
                capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            first = True
            last_size = None
            while not self.stop_event.is_set():
                ok, frame = capture.read()
                if not ok or frame is None:
                    raise RuntimeError('Webcam stopped delivering frames. Check Iriun and reconnect.')
                if self.stop_event.is_set():
                    break
                if first or last_size != (frame.shape[1], frame.shape[0]):
                    self.connected = True
                    actual = (frame.shape[1], frame.shape[0])
                    last_size = actual
                    message = f'Webcam {index}: {actual[0]} × {actual[1]}'
                    if width is not None:
                        message += f' (requested {width} × {height}'
                        message += '; driver returned a different size)' if actual != (width, height) else '; accepted)'
                    self.events.put(('connected', message))
                    first = False
                try:
                    self.frames.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self.frames.put_nowait(frame)
                except queue.Full:
                    pass
        except Exception as error:
            if not self.stop_event.is_set():
                self.events.put(('error', str(error)))
        finally:
            self.connected = False
            if capture is not None:
                capture.release()

    def stop(self):
        self.stop_event.set()
        self.connected = False
        if self.thread is not None:
            self.thread.join(timeout=.2)
