import subprocess
from threading import Thread, Event
from queue import Queue, Empty
from config import ESPEAK_VOICE, ESPEAK_SPEED


class TextToSpeech:
    def __init__(self, voice: str = ESPEAK_VOICE, speed: str = ESPEAK_SPEED):
        self.voice = voice
        self.speed = speed
        self.queue = Queue(maxsize=1)
        self._stop_event = Event()
        self._check_espeak_availability()

        # Worker thread to speak texts in queue
        self.worker_thread = Thread(target=self._process_queue)
        self.worker_thread.daemon = True
        self.worker_thread.start()
    
    def _check_espeak_availability(self) -> None:
        try:
            subprocess.run(["espeak", "--version"], capture_output=True, check=True)
        except (FileNotFoundError, subprocess.CalledProcessError):
            print("Warning: eSpeak not found. Speech functionality disabled.")
    
    def _speak_sync(self, text: str) -> None:
        try:
            subprocess.run([
                "espeak", "-v", self.voice, "-s", self.speed, text
            ], check=True)
        except FileNotFoundError:
            print("eSpeak not found.")
        except subprocess.CalledProcessError as e:
            print(f"eSpeak error: {e}")

    def _process_queue(self):
        while not self._stop_event.is_set():
            try:
                text = self.queue.get(timeout=0.1)
                self._speak_sync(text)
                self.queue.task_done()
            except Empty:
                continue

    def speak(self, text: str) -> bool:
        if not text.strip():
            return False
        self.queue.put(text)
        return True

    def stop(self):
        """Stop the TTS worker thread gracefully."""
        self._stop_event.set()
        self.worker_thread.join()
