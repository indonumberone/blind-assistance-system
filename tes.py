import time
from distance_sensor import create_distance_sensor

def test_distance_sensor():
    """Test the distance sensor functionality."""
    print("Starting distance sensor test...")
    print("Press Ctrl+C to stop")
    
    # Create sensor instance
    sensor = create_distance_sensor()
    
    if sensor is None:
        print("Failed to create distance sensor!")
        return
    
    try:
        while True:
            # Measure distance
            distance = sensor.measure_distance()
            
            if distance is not None:
                print(f"Distance: {distance:.2f} cm")
                
                # Optional: Add distance warnings
                if distance < 10:
                    print("  -> Object very close!")
                elif distance < 30:
                    print("  -> Object nearby")
            else:
                print("Failed to measure distance (timeout or error)")
            
            # Wait before next measurement
            time.sleep(0.5)
            
    except KeyboardInterrupt:
        print("\nTest stopped by user")
    except Exception as e:
        print(f"Error during test: {e}")
    finally:
        # Clean up resources
        sensor.cleanup()
        print("Sensor cleanup completed")

if __name__ == "__main__":
    test_distance_sensor()
