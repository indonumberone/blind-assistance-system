import cv2
import onnxruntime as ort
import numpy as np
import time
import random
# import RPi.GPIO as GPIO  # Disabled for debug mode
import subprocess

# === DEBUG MODE - GPIO DISABLED ===
DEBUG_MODE = True
print("🔧 DEBUG MODE AKTIF - GPIO SENSOR DINONAKTIFKAN")

def ukur_jarak_simulasi():
    """Simulasi sensor jarak untuk debug mode"""
    # Simulasi jarak random antara 50-500 cm
    jarak_simulasi = random.randint(50, 500)
    time.sleep(0.01)  # Simulasi delay sensor
    return jarak_simulasi

# === MODEL DETEKSI ONNX SETUP ===
print("📡 Loading ONNX model...")
try:
    session = ort.InferenceSession("best.onnx")
    print("✅ Model berhasil dimuat")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    print("📝 Menggunakan simulasi deteksi untuk demo")
    session = None

CLASS_NAMES = ["Tempat Sampah", "kursi", "lampu lalu lintas", "lubang-jalan", "mobil",
               "motor", "person", "pohon", "tangga", "zebracross"]

# Warna untuk setiap class (BGR format untuk OpenCV)
CLASS_COLORS = [
    (0, 255, 0),    # Tempat Sampah - Hijau
    (255, 0, 0),    # kursi - Biru
    (0, 255, 255),  # lampu lalu lintas - Kuning
    (128, 0, 128),  # lubang-jalan - Ungu
    (255, 255, 0),  # mobil - Cyan
    (0, 128, 255),  # motor - Orange
    (255, 0, 255),  # person - Magenta
    (0, 128, 0),    # pohon - Hijau gelap
    (128, 128, 0),  # tangga - Olive
    (255, 255, 255) # zebracross - Putih
]

CONF_THRESH = 0.4
PROCESS_EVERY_N_FRAMES = 2
SHOW_FPS = True

def speak(text):
    print(f"🔊 Speaking: {text}")
    if not DEBUG_MODE:
        try:
            subprocess.run(["espeak", "-v", "id+m3", "-s", "150", text])
        except FileNotFoundError:
            print("❌ eSpeak tidak ditemukan.")
    else:
        print("🔧 Debug mode - Audio output disabled")

def draw_detection_info(frame, detected_labels, jarak_cm, fps):
    """Menggambar informasi deteksi pada frame"""
    height, width = frame.shape[:2]
    
    # Background untuk info panel
    cv2.rectangle(frame, (10, 10), (width-10, 120), (0, 0, 0), -1)
    cv2.rectangle(frame, (10, 10), (width-10, 120), (255, 255, 255), 2)
    
    # Judul
    cv2.putText(frame, "SISTEM DETEKSI OBJEK & JARAK", 
                (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    # FPS
    cv2.putText(frame, f"FPS: {fps:.1f}", 
                (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    # Jarak
    if jarak_cm:
        color = (0, 0, 255) if jarak_cm < 100 else (0, 255, 255) if jarak_cm < 300 else (0, 255, 0)
        cv2.putText(frame, f"Jarak: {jarak_cm} cm", 
                    (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    # Objek terdeteksi
    if detected_labels:
        objects_text = f"Objek: {', '.join(list(detected_labels)[:3])}"  # Maksimal 3 objek
        if len(detected_labels) > 3:
            objects_text += f" +{len(detected_labels)-3} lagi"
        cv2.putText(frame, objects_text, 
                    (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Status warning
        if jarak_cm and jarak_cm < 200:
            cv2.putText(frame, "⚠️ OBJEK DEKAT!", 
                        (20, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    else:
        cv2.putText(frame, "Objek: Tidak ada deteksi", 
                    (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)

def draw_bounding_boxes(frame, outputs, h0, w0, inference_size):
    """Menggambar bounding box untuk objek yang terdeteksi"""
    detected_labels = set()
    
    if outputs is None:
        return detected_labels
    
    for i, det in enumerate(outputs[0]):
        x1, y1, x2, y2, conf, cls_id = det
        if conf < CONF_THRESH:
            continue
            
        cls_id = int(cls_id)
        if cls_id >= len(CLASS_NAMES):
            continue
            
        label = CLASS_NAMES[cls_id]
        detected_labels.add(label)
        
        # Konversi koordinat dari inference size ke ukuran frame asli
        x1 = int(x1 * w0 / inference_size)
        y1 = int(y1 * h0 / inference_size)
        x2 = int(x2 * w0 / inference_size)
        y2 = int(y2 * h0 / inference_size)
        
        # Pilih warna berdasarkan class
        color = CLASS_COLORS[cls_id % len(CLASS_COLORS)]
        
        # Gambar bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Background untuk label
        label_text = f"{label}: {conf:.2f}"
        (text_width, text_height), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1-25), (x1 + text_width + 10, y1), color, -1)
        
        # Text label
        cv2.putText(frame, label_text, (x1 + 5, y1 - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    return detected_labels

def simulate_detection():
    """Simulasi deteksi untuk testing tanpa model"""
    # Simulasi beberapa deteksi random
    simulated_objects = random.choices(CLASS_NAMES, k=random.randint(0, 3))
    return set(simulated_objects)

def main():
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("❌ Error: Cannot open webcam")
        print("🔧 Menggunakan webcam virtual untuk demo...")
        # Untuk demo, kita akan membuat frame dummy
        cap = None
    
    print("🚀 Mulai proses deteksi. Tekan 'q' untuk keluar.")
    frame_count = 0
    fps_start_time = time.time()
    start_time = time.time()
    last_spoken_labels = set()
    current_fps = 0

    try:
        while True:
            # Baca frame dari webcam atau buat frame dummy
            if cap:
                ret, frame = cap.read()
                if not ret:
                    print("❌ Gagal membaca frame.")
                    break
            else:
                # Frame dummy untuk demo
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                frame[:] = (50, 50, 50)  # Background abu-abu
                cv2.putText(frame, "DEMO MODE - NO WEBCAM", (200, 240), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            frame_count += 1

            # Hitung FPS
            if SHOW_FPS and (time.time() - fps_start_time) > 1:
                current_fps = frame_count / (time.time() - fps_start_time)
                print(f"📊 FPS: {current_fps:.1f}")
                frame_count = 0
                fps_start_time = time.time()

            if frame_count % PROCESS_EVERY_N_FRAMES != 0:
                # Tampilkan frame dengan info sebelumnya
                draw_detection_info(frame, last_spoken_labels, None, current_fps)
                cv2.imshow('Object Detection Debug', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue

            detected_labels = set()
            
            # Proses deteksi objek
            if session:
                try:
                    h0, w0 = frame.shape[:2]
                    inference_size = 640
                    input_img = cv2.resize(frame, (inference_size, inference_size))
                    input_img = input_img.astype(np.float32) / 255.0
                    input_img = np.transpose(input_img, (2, 0, 1))
                    input_img = np.expand_dims(input_img, axis=0)

                    outputs = session.run(None, {session.get_inputs()[0].name: input_img})[0]
                    detected_labels = draw_bounding_boxes(frame, outputs, h0, w0, inference_size)
                    
                except Exception as e:
                    print(f"❌ Error dalam deteksi: {e}")
                    detected_labels = simulate_detection()
            else:
                # Simulasi deteksi
                detected_labels = simulate_detection()
                if detected_labels:
                    print(f"🎯 Simulasi deteksi: {detected_labels}")

            # Ukur jarak (simulasi atau real)
            jarak_cm = None
            if detected_labels:
                if DEBUG_MODE:
                    jarak_cm = ukur_jarak_simulasi()
                    print(f"📏 Jarak simulasi: {jarak_cm} cm")
                
                # Logic untuk speech
                if jarak_cm and (jarak_cm < 400 or detected_labels != last_spoken_labels):
                    last_spoken_labels = detected_labels.copy()
                    spoken_text = f"Terdeteksi: {', '.join(detected_labels)}, dalam jarak {int(jarak_cm)} sentimeter"
                    speak(spoken_text)
                    
                    # Log ke console
                    print(f"🎯 Deteksi: {detected_labels}")
                    print(f"📏 Jarak: {jarak_cm} cm")
                    print(f"🔊 Audio: {spoken_text}")
                    print("-" * 50)

            # Gambar informasi pada frame
            draw_detection_info(frame, detected_labels, jarak_cm, current_fps)
            
            # Tampilkan frame
            cv2.imshow('Object Detection Debug', frame)
            
            # Keluar jika 'q' ditekan
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("\n🛑 Dihentikan oleh pengguna.")

    finally:
        if cap:
            cap.release()
        cv2.destroyAllWindows()
        
        if not DEBUG_MODE:
            GPIO.cleanup()
            
        total_time = time.time() - start_time
        print("✅ Program selesai.")
        print(f"📊 FPS rata-rata: {frame_count / total_time:.1f}")
        print(f"⏱️  Total waktu: {total_time:.1f} detik")

if __name__ == "__main__":
    main()