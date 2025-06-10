import time
from typing import Optional
from config import GPIO_TRIG, GPIO_ECHO, DISTANCE_TIMEOUT, DEBUG_MODE, MOCK_DISTANCE_CM

# Only import RPi.GPIO when not in debug mode
if not DEBUG_MODE:
    import RPi.GPIO as GPIO


class MockDistanceSensor:
    """Mock distance sensor for debugging on non-Raspberry Pi systems."""
    
    def __init__(self, mock_distance: float = MOCK_DISTANCE_CM):
        self.mock_distance = mock_distance
        print(f"Using mock distance sensor with constant distance: {mock_distance} cm")
    
    def measure_distance(self) -> Optional[float]:
        """Return a constant mock distance."""
        return self.mock_distance
    
    def cleanup(self) -> None:
        """Mock cleanup - does nothing."""
        pass


class DistanceSensor:
    """Handles ultrasonic distance sensor operations."""
    
    def __init__(self, trig_pin: int = GPIO_TRIG, echo_pin: int = GPIO_ECHO):
        if DEBUG_MODE:
            raise RuntimeError("Real DistanceSensor should not be used in DEBUG_MODE")
        
        self.trig_pin = trig_pin
        self.echo_pin = echo_pin
        self._setup_gpio()
    
    def _setup_gpio(self) -> None:
        """Initialize GPIO pins for the sensor."""
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.trig_pin, GPIO.OUT)
        GPIO.setup(self.echo_pin, GPIO.IN)
    
    def measure_distance(self) -> Optional[float]:
        """
        Measure distance using ultrasonic sensor.
        
        Returns:
            Distance in centimeters, or None if measurement failed.
        """
        GPIO.output(self.trig_pin, False)
        time.sleep(0.05)
        
        GPIO.output(self.trig_pin, True)
        time.sleep(0.00001)  # 10 microseconds
        GPIO.output(self.trig_pin, False)
        
        pulse_start, pulse_end = 0, 0
        timeout = time.time() + DISTANCE_TIMEOUT
        
        while GPIO.input(self.echo_pin) == 0:
            pulse_start = time.time()
            if pulse_start > timeout:
                return None
        
        while GPIO.input(self.echo_pin) == 1:
            pulse_end = time.time()
            if pulse_end > timeout:
                return None
        
        pulse_duration = pulse_end - pulse_start
        distance = pulse_duration * 17150  
        return round(distance, 2)
    
    def cleanup(self) -> None:
        """Clean up GPIO resources."""
        GPIO.cleanup()


def create_distance_sensor() -> Optional[object]:
    """Factory function to create appropriate distance sensor based on debug mode."""
    if DEBUG_MODE:
        return MockDistanceSensor()
    else:
        return DistanceSensor()
