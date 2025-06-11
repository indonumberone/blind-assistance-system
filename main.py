"""Main application for IuSee object detection system."""

import cv2
import time
from typing import Set, Optional

from distance_sensor import create_distance_sensor
from object_detector import ObjectDetector
from text_to_speech import TextToSpeech
from config import (
    PROCESS_EVERY_N_FRAMES, SHOW_FPS, FPS_UPDATE_INTERVAL,
    DISTANCE_THRESHOLD_CM, DEBUG_MODE, ONNX_MODEL_PATH, SHOW_BBOXES
)


class IuSeeApp:
    """Main application class for the IuSee system."""
    
    def __init__(self):
        self.distance_sensor = create_distance_sensor()
        self.object_detector = ObjectDetector(model_path=ONNX_MODEL_PATH, src=1).start()
        time.sleep(2.0)
        self.tts = TextToSpeech()
        
        self.frame_count = 0
        self.fps_start_time = time.time()
        self.last_spoken_labels: Set[str] = set()
        
        if DEBUG_MODE:
            print("Running in DEBUG_MODE - using mock sensors")
        
    def _calculate_and_display_fps(self) -> None:
        """Calculate and display FPS if enabled."""
        if not SHOW_FPS:
            return
            
        current_time = time.time()
        if (current_time - self.fps_start_time) > FPS_UPDATE_INTERVAL:
            fps = self.frame_count / (current_time - self.fps_start_time)
            print(f"FPS: {fps:.1f}")
            self.frame_count = 0
            self.fps_start_time = current_time
    
    def _should_process_frame(self) -> bool:
        """Determine if current frame should be processed."""
        return self.frame_count % PROCESS_EVERY_N_FRAMES == 0
    
    def _handle_detections(self, detected_labels: Set[str]) -> None:
        """
        Handle detected objects by measuring distance and speaking.
        
        Args:
            detected_labels: Set of detected object labels
        """
        if not detected_labels:
            return
        
        distance_cm = self.distance_sensor.measure_distance()
        if distance_cm is None:
            print("Failed to measure distance")
            return
        
        print(f"Distance: {distance_cm} cm")
        
        
        if detected_labels != self.last_spoken_labels and distance_cm < DISTANCE_THRESHOLD_CM * 2:
            self.last_spoken_labels = detected_labels
            labels_text = ', '.join(detected_labels)
            spoken_text = f"AWASS Ada {labels_text} di depan jarak {int(distance_cm)} sentimeter"
            print(spoken_text)
            self.tts.speak(spoken_text)
        elif distance_cm < DISTANCE_THRESHOLD_CM:
            spoken_text = f"AWASS Ada sesuatu disdepan jarak {int(distance_cm)} sentimeter"
            print(spoken_text)
            self.tts.speak(spoken_text)

        
    def run(self) -> None:
        """Main application loop."""
        print("Mulai proses deteksi. Tekan Ctrl+C untuk keluar.")
        start_time = time.time()
        
        try:
            count = 0
            while True:
                frame = self.object_detector.read_frame()
                if frame is None:
                    time.sleep(0.001)  
                    print("Melewati deteksi frame")
                    continue
                time.sleep(0.001)
                self.frame_count += 1
                self._calculate_and_display_fps()
                if count > 3:
                    self.last_spoken_labels.clear()
                if self._should_process_frame():
                    if SHOW_BBOXES:
                        detected_labels, bounding_boxes = self.object_detector.detect_objects_optimized(frame, need_bboxes=SHOW_BBOXES)
                        self._handle_detections(detected_labels)
                        frame_with_boxes = self.object_detector.draw_bounding_boxes(frame, bounding_boxes)
                        cv2.imshow("IuSee Preview", frame_with_boxes)
                    else:
                        detected_labels = self.object_detector.detect_objects_labels_only(frame)
                        self._handle_detections(detected_labels)
                count += 1
                
                # Check for 'q' key press to exit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        except KeyboardInterrupt:
            print("\nDihentikan oleh user.")
        
        finally:
            self.object_detector.stop_capture()
            cv2.destroyAllWindows()
            self.distance_sensor.cleanup()
            total_time = time.time() - start_time
            print("Program selesai.")
            if total_time > 0:
                print(f"FPS rata-rata: {self.frame_count / total_time:.1f}")


def main():
    """Entry point for the application."""
    try:
        app = IuSeeApp()
        app.run()
    except Exception as e:
        print(f"Application error: {e}")
        return 1
    return 0


if __name__ == "__main__":
    exit(main())