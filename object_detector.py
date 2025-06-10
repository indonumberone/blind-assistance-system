import cv2
import numpy as np
import onnxruntime as ort
from typing import List, Set, Tuple
from config import ONNX_MODEL_PATH, CONFIDENCE_THRESHOLD, INFERENCE_SIZE, CLASS_NAMES
import queue
from threading import Thread
import time

class ObjectDetector:
    """Handler Onnx object detection"""
    
    def __init__(self, model_path: str = ONNX_MODEL_PATH,src=0):
        self.model_path = model_path
        self.session = None
        self.class_names = CLASS_NAMES
        self.conf_threshold = CONFIDENCE_THRESHOLD
        self.inference_size = INFERENCE_SIZE
        self._load_model()
        self.stream = None
        self.camera_src = src
        self._setup_camera()
        self.stop = False
        self.q = queue.Queue(maxsize=2)
        
    def _setup_camera(self):
        self.stream = cv2.VideoCapture(self.camera_src)
        if not self.stream.isOpened():
            print(f"Camera at index {self.camera_src} failed, trying 0...")
            self.stream = cv2.VideoCapture(0)
            if not self.stream.isOpened():
                raise RuntimeError("No available camera.")
    def start(self):
        """Start capturing frames from the camera."""
        Thread(target=self.update_queue, daemon=True).start()
        return self
    def update_queue(self):
        """Continuously read frames from the camera and put them in the queue."""
        while not self.stop:
            if not self.q.full():
                grabbed, frame = self.stream.read()
                if not grabbed:
                    self.stop = True
                    break
                if self.q.empty():
                    self.q.put(frame)
                else:
                    try:
                        self.q.get_nowait()
                        self.q.put(frame)
                    except queue.Empty:
                        pass
            else:
                time.sleep(0.001)

        # while not self.stop:
        #     if not self.q.full():
        #         grabbed, frame = self.stream.read()
        #         if not grabbed:
        #             self.stop = True
        #             break
        #         if self.q.empty():
        #             self.q.put(frame)
        #         else:
        #             try:
        #                 self.q.get_nowait()
        #                 self.q.put(frame)
        #             except queue.Empty:
        #                 pass
        #     else:
        #         time.sleep(0.001)

                
    def read_frame(self):
        return None if self.q.empty() else self.q.get()

        # """Read a frame from the queue."""
        # if self.q.empty():
        #     return False, None
        # frame = self.q.get()
        # return True, frame
    def stop_capture(self):
        self.stop = True
        self.stream.release()
    def _load_model(self) -> None:
        """Load model onnx e"""
        try:
            print(f"Loading ONNX model from {self.model_path}...")
            self.session = ort.InferenceSession(self.model_path)
            print("Model loaded successfully.")
        except Exception as e:
            raise RuntimeError(f"Failed to load ONNX model: {e}")
    
    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        detect frame by frame
        
        Args:
            frame: Input frame from camera

        Returns:
            Preprocessed frame ready for CNN inference
            
        """
        # Resize and normalize
        input_img = cv2.resize(frame, (self.inference_size, self.inference_size))
        input_img = input_img.astype(np.float32) / 255.0
        
        # Convert HWC to CHW format and add batch dimension
        input_img = np.transpose(input_img, (2, 0, 1))
        input_img = np.expand_dims(input_img, axis=0)
        
        return input_img
    
    def detect_objects(self, frame: np.ndarray) -> Set[str]:
        """
        Detect objects in the given frame.
        
        Args:
            frame: Input frame from camera
            
        Returns:
            Set of detected object labels
        """
        if self.session is None:
            raise RuntimeError("Model not loaded")
        
        input_img = self._preprocess_frame(frame)
        
        input_name = self.session.get_inputs()[0].name
        outputs = self.session.run(None, {input_name: input_img})[0]
        
        # Process detections
        detected_labels = set()
        for det in outputs[0]:
            x1, y1, x2, y2, conf, cls_id = det
            
            if conf < self.conf_threshold:
                continue
            
            cls_id = int(cls_id)
            if cls_id < len(self.class_names):
                label = self.class_names[cls_id]
                detected_labels.add(label)
        
        return detected_labels
