import subprocess
from typing import Optional
from config import ESPEAK_VOICE, ESPEAK_SPEED


class TextToSpeech:
    """Handles text-to-speech operations using eSpeak."""
    
    def __init__(self, voice: str = ESPEAK_VOICE, speed: str = ESPEAK_SPEED):
        self.voice = voice
        self.speed = speed
        self._check_espeak_availability()
    
    def _check_espeak_availability(self) -> None:
        """Check if eSpeak is available on the system."""
        try:
            subprocess.run(["espeak", "--version"], 
                         capture_output=True, check=True)
        except (FileNotFoundError, subprocess.CalledProcessError):
            print("Warning: eSpeak not found. Speech functionality disabled.")
    
    def speak(self, text: str) -> bool:
        """
        Convert text to speech.
        
        Args:
            text: Text to be spoken
            
        Returns:
            True if speech was successful, False otherwise
        """
        if not text.strip():
            return False
        
        print(f"Speaking: {text}")
        
        try:
            subprocess.run([
                "espeak", "-v", self.voice, "-s", self.speed, text
            ], check=True)
            return True
        except FileNotFoundError:
            print("eSpeak not found. Cannot speak.")
            return False
        except subprocess.CalledProcessError as e:
            print(f"eSpeak error: {e}")
            return False
