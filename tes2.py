import cv2
import onnxruntime as ort
import numpy as np
import time
from threading import Thread
import queue

# Load ONNX model
print("Loading ONNX model...")
session = ort.InferenceSession("best.onnx")

# Class names for detection
CLASS_NAMES = ["Tempat Sampah", "kursi", "lampu lalu lintas", "lubang-jalan", "mobil",
               "motor", "person", "pohon", "tangga", "zebracross"]

# Confidence threshold
CONF_THRESH = 0.4

# Show FPS on screen
SHOW_FPS = True

# Skip frame processing for performance
PROCESS_EVERY_N_FRAMES = 2

# -----------------------------
# Video Stream with Threading
# -----------------------------
class VideoStream:
    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src)
        self.stopped = False
        self.q = queue.Queue(maxsize=2)

    def start(self):
        Thread(target=self.update, daemon=True).start()
        return self

    def update(self):
        while not self.stopped:
            if not self.q.full():
                grabbed, frame = self.stream.read()
                if not grabbed:
                    self.stopped = True
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

    def read(self):
        return None if self.q.empty() else self.q.get()

    def stop(self):
        self.stopped = True
        self.stream.release()


# -----------------------------
# Main Detection Loop
# -----------------------------
def main():
    print("Connecting to video stream...")
    vs = VideoStream(1).start()
    time.sleep(2.0)

    frame = vs.read()
    if frame is None:
        print("Error: Could not read from video source")
        return

    height, width = frame.shape[:2]
    print(f"Stream dimensions: {width}x{height}")

    out = cv2.VideoWriter("output.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 20, (width, height))

    print("Processing video. Press 'q' to exit.")

    frame_count = 0
    start_time = time.time()
    fps_start_time = start_time
    fps = 0

    try:
        while True:
            frame = vs.read()
            if frame is None:
                continue

            frame_count += 1

            # FPS calculation
            if SHOW_FPS and (time.time() - fps_start_time) > 1:
                fps = frame_count / (time.time() - fps_start_time)
                frame_count = 0
                fps_start_time = time.time()

            if frame_count % PROCESS_EVERY_N_FRAMES != 0:
                if SHOW_FPS:
                    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow("Object Detection", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue

            # Resize for inference
            h0, w0 = frame.shape[:2]
            inference_size = 640
            input_img = cv2.resize(frame, (inference_size, inference_size))
            input_img = input_img.astype(np.float32) / 255.0
            input_img = np.transpose(input_img, (2, 0, 1))
            input_img = np.expand_dims(input_img, axis=0)

            # Inference
            outputs = session.run(None, {session.get_inputs()[0].name: input_img})[0]

            # Draw boxes
            for det in outputs[0]:
                x1, y1, x2, y2, conf, cls_id = det
                if conf < CONF_THRESH:
                    continue

                x1 = int(x1 / inference_size * w0)
                y1 = int(y1 / inference_size * h0)
                x2 = int(x2 / inference_size * w0)
                y2 = int(y2 / inference_size * h0)
                cls_id = int(cls_id)
                label = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else str(cls_id)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f'{label} {conf:.2f}', (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

                print(f"Detected: {label} with confidence {conf:.2f} at [{x1}, {y1}, {x2}, {y2}]")

            if SHOW_FPS:
                cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            out.write(frame)
            # cv2.imshow("Object Detection", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        vs.stop()
        out.release()
        cv2.destroyAllWindows()
        print(f"Output saved to output.mp4")
        print(f"Average FPS: {frame_count/(time.time()-start_time):.1f}")

if __name__ == "__main__":
    main()