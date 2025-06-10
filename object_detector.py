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
    
    def detect_objects_labels_only(self, frame: np.ndarray) -> Set[str]:
        """
        Detect objects and return only labels (optimized for performance).
        
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
        
        # Process detections - only extract labels
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

    def detect_objects(self, frame: np.ndarray) -> Tuple[Set[str], List[Tuple]]:
        """
        Detect objects in the given frame.
        
        Args:
            frame: Input frame from camera
            
        Returns:
            Tuple of (Set of detected object labels, List of bounding boxes)
            Bounding box format: (x1, y1, x2, y2, confidence, label)
        """
        if self.session is None:
            raise RuntimeError("Model not loaded")
        
        input_img = self._preprocess_frame(frame)
        
        input_name = self.session.get_inputs()[0].name
        outputs = self.session.run(None, {input_name: input_img})[0]
        
        # Get original frame dimensions
        h, w = frame.shape[:2]
        
        # Process detections
        detected_labels = set()
        bounding_boxes = []
        
        for det in outputs[0]:
            x1, y1, x2, y2, conf, cls_id = det
            
            if conf < self.conf_threshold:
                continue
            
            cls_id = int(cls_id)
            if cls_id < len(self.class_names):
                label = self.class_names[cls_id]
                detected_labels.add(label)
                
                # Scale coordinates back to original frame size
                x1 = int(x1 * w / self.inference_size)
                y1 = int(y1 * h / self.inference_size)
                x2 = int(x2 * w / self.inference_size)
                y2 = int(y2 * h / self.inference_size)
                
                bounding_boxes.append((x1, y1, x2, y2, conf, label))
        
        return detected_labels, bounding_boxes

    def detect_objects_optimized(self, frame: np.ndarray, need_bboxes: bool = True) -> Tuple[Set[str], List[Tuple]]:
        """
        Optimized detection method that skips bbox processing when not needed.
        
        Args:
            frame: Input frame from camera
            need_bboxes: Whether to calculate bounding box coordinates
            
        Returns:
            Tuple of (Set of detected object labels, List of bounding boxes or empty list)
        """
        if self.session is None:
            raise RuntimeError("Model not loaded")
        
        input_img = self._preprocess_frame(frame)
        
        input_name = self.session.get_inputs()[0].name
        outputs = self.session.run(None, {input_name: input_img})[0]
        
        # Process detections
        detected_labels = set()
        bounding_boxes = []
        
        # Get frame dimensions only if we need bboxes
        if need_bboxes:
            h, w = frame.shape[:2]
            center_x_min = int(w * 1/3)
            center_x_max = int(w * 2/3)
            center_y_min = int(h * 1/3)
            center_y_max = int(h * 2/3)

        
        
        
        for det in outputs[0]:
            x1, y1, x2, y2, conf, cls_id = det
            
            if conf < self.conf_threshold:
                continue
            
            cls_id = int(cls_id)
            if cls_id < len(self.class_names):
            
                
                # Only process bbox coordinates if needed
                if need_bboxes:
                    # Scale coordinates back to original frame size
                    x1 = int(x1 * w / self.inference_size)
                    y1 = int(y1 * h / self.inference_size)
                    x2 = int(x2 * w / self.inference_size)
                    y2 = int(y2 * h / self.inference_size)

                    center_x = (x1+x2) // 2  
                    center_y = (y1+y2) // 2  
                    
                    if (center_x_min <= center_x <= center_x_max) and (center_y_min <= center_y <= center_y_max):
                        label = self.class_names[cls_id]
                        detected_labels.add(label)
                        bounding_boxes.append((x1, y1, x2, y2, conf, label))

        
        return detected_labels, bounding_boxes

    def draw_bounding_boxes(self, frame: np.ndarray, bounding_boxes: List[Tuple]) -> np.ndarray:
        """
        Draw bounding boxes on the frame.
        
        Args:
            frame: Input frame
            bounding_boxes: List of bounding boxes (x1, y1, x2, y2, conf, label)
            
        Returns:
            Frame with bounding boxes drawn
        """
        frame_copy = frame.copy()
        
        for x1, y1, x2, y2, conf, label in bounding_boxes:
            # Draw bounding box
            cv2.rectangle(frame_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label and confidence
            label_text = f"{label}: {conf:.2f}"
            label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            
            # Draw background for text
            cv2.rectangle(frame_copy, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), (0, 255, 0), -1)
            
            # Draw text
            cv2.putText(frame_copy, label_text, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        return frame_copy

    def display_frame(self, frame: np.ndarray, window_name: str = "IuSee Preview") -> bool:
        """
        Display frame using cv2.imshow
        
        Args:
            frame: Frame to display
            window_name: Window title
            
        Returns:
            False if 'q' is pressed, True otherwise
        """
        cv2.imshow(window_name, frame)
        key = cv2.waitKey(1) & 0xFF
        return key != ord('q')
    
    def run_preview(self):
        """
        Run live preview with object detection and bounding boxes
        Press 'q' to quit
        """
        self.start()
        print("Press 'q' to quit preview")
        
        try:
            while True:
                frame = self.read_frame()
                if frame is not None:
                    # Detect objects and get bounding boxes
                    detected_objects, bounding_boxes = self.detect_objects(frame)
                    if detected_objects:
                        print(f"Detected: {detected_objects}")
                    
                    # Draw bounding boxes on frame
                    frame_with_boxes = self.draw_bounding_boxes(frame, bounding_boxes)
                    
                    # Display frame
                    if not self.display_frame(frame_with_boxes):
                        break
                else:
                    time.sleep(0.01)
        finally:
            self.stop_capture()
            cv2.destroyAllWindows()

# Example usage:
# detector = ObjectDetector()
# detector.run_preview()
