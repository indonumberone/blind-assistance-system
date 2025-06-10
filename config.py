# System Configuration
DEBUG_MODE = True  # Set to False when running on Raspberry Pi
MOCK_DISTANCE_CM = 200  # Constant distance for debugging

# GPIO Configuration
GPIO_TRIG = 14
GPIO_ECHO = 15

# Distance Sensor Configuration
DISTANCE_TIMEOUT = 0.04  # 40ms timeout
DISTANCE_THRESHOLD_CM = 400  # Alert when objects are closer than this

# Object Detection Configuration
ONNX_MODEL_PATH = "best.onnx"
CONFIDENCE_THRESHOLD = 0.4
INFERENCE_SIZE = 640
PROCESS_EVERY_N_FRAMES = 2

# Class names for object detection
CLASS_NAMES = [
    "Tempat Sampah", "kursi", "lampu lalu lintas", "lubang-jalan", "mobil",
    "motor", "person", "pohon", "tangga", "zebracross"
]

# Text-to-Speech Configuration
ESPEAK_VOICE = "id+m3"
ESPEAK_SPEED = "150"

# Display Configuration
SHOW_FPS = True
FPS_UPDATE_INTERVAL = 1.0  # seconds
