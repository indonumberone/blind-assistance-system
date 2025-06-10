"""Main application for IuSee object detection system."""

import cv2
import time
from typing import Set, Optional

from distance_sensor import create_distance_sensor
from object_detector import ObjectDetector
from text_to_speech import TextToSpeech
from config import (
    PROCESS_EVERY_N_FRAMES, SHOW_FPS, FPS_UPDATE_INTERVAL,
    DISTANCE_THRESHOLD_CM, DEBUG_MODE
)


class IuSeeApp:
    """Main application class for the IuSee system."""
    
    def __init__(self):
        self.distance_sensor = create_distance_sensor()
        self.object_detector = ObjectDetector()
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
        
        
        if distance_cm < DISTANCE_THRESHOLD_CM:
            spoken_text = f"AWASS Ada sesuatu disdepan jarak {int(distance_cm)} sentimeter"
            self.tts.speak(spoken_text)
        elif detected_labels != self.last_spoken_labels:
            self.last_spoken_labels = detected_labels
            labels_text = ', '.join(detected_labels)
            spoken_text = f"AWASS Ada {labels_text} di depan jarak {int(distance_cm)} sentimeter"
            self.tts.speak(spoken_text)
        
    def run(self) -> None:
        """Main application loop."""
        cap = cv2.VideoCapture(1)
        if not cap.isOpened():
            raise RuntimeError("Error: Cannot open webcam")
        
        print("Mulai proses deteksi. Tekan Ctrl+C untuk keluar.")
        start_time = time.time()
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Gagal membaca frame.")
                    break
                
                self.frame_count += 1
                self._calculate_and_display_fps()
                
                # Skip processing if not the right frame
                if not self._should_process_frame():
                    continue
                
                # Detect objects
                detected_labels = self.object_detector.detect_objects(frame)
                
                # Handle detections
                self._handle_detections(detected_labels)
                
        except KeyboardInterrupt:
            print("\nDihentikan oleh user.")
        
        finally:
            cap.release()
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