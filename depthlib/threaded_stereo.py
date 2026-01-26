from depthlib.StereoDepthEstimator import StereoDepthEstimator
from depthlib.input import open_capture
import threading
from queue import Queue
from typing import Optional, Tuple
import numpy as np
import cv2

class ThreadedStereoCapture:
    """Threaded stereo frame capture for improved FPS.
    
    Reads frames from both cameras in a background thread to avoid
    blocking the main processing loop.
    """
    
    def __init__(self, left_source, right_source, downscale_factor=1.0, buffer_size=2, drop_frames=True):
        self.left_source = left_source
        self.right_source = right_source
        self.downscale_factor = downscale_factor
        self.buffer_size = buffer_size
        self.drop_frames = drop_frames  # If True, drop old frames when queue full (for live cameras)
        
        self._frame_queue: Queue = Queue(maxsize=buffer_size)
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._cap_L: Optional[cv2.VideoCapture] = None
        self._cap_R: Optional[cv2.VideoCapture] = None
    
    def start(self):
        """Start the capture thread."""
        self._cap_L = open_capture(self.left_source)
        self._cap_R = open_capture(self.right_source)
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
    
    def _read_frame(self, cap: cv2.VideoCapture) -> Optional[np.ndarray]:
        ok, frame = cap.read()
        if not ok or frame is None:
            return None
        if self.downscale_factor != 1.0:
            new_size = (
                int(frame.shape[1] * self.downscale_factor),
                int(frame.shape[0] * self.downscale_factor),
            )
            frame = cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)
        return frame
    
    def _capture_loop(self):
        """Background thread loop for capturing frames."""
        while not self._stop_event.is_set():
            if self._cap_L is None or self._cap_R is None:
                self._stop_event.set()
                break
            left = self._read_frame(self._cap_L)
            right = self._read_frame(self._cap_R)
            
            if left is None or right is None:
                self._stop_event.set()
                break
            
            # Drop old frames if queue is full (for live cameras to reduce latency)
            # For video files, we want to process every frame, so don't drop
            if self.drop_frames and self._frame_queue.full():
                try:
                    self._frame_queue.get_nowait()
                except:
                    pass
            
            # Block until there's space (for video files this ensures no frames are skipped)
            self._frame_queue.put((left, right))
    
    def read(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Get the next frame pair. Returns None if stream ended."""
        if self._stop_event.is_set() and self._frame_queue.empty():
            return None
        try:
            return self._frame_queue.get(timeout=1.0)
        except:
            return None
    
    def stop(self):
        """Stop the capture thread and release resources."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._cap_L is not None:
            self._cap_L.release()
        if self._cap_R is not None:
            self._cap_R.release()