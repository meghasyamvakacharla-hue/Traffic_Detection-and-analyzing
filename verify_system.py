from src.processor import TrafficAnalyzer
import cv2
import os

def test_system():
    if not os.path.exists("sample_traffic.mp4"):
        print("Sample video not found. Skipping test.")
        return

    print("Initializing TrafficAnalyzer...")
    analyzer = TrafficAnalyzer()
    
    # Define dummy ROI and Line (based on sample video dimensions approx 1280x720)
    roi = [(200, 200), (600, 200), (600, 600), (200, 600)]
    line = [(150, 500), (950, 500)]
    
    analyzer.set_roi(roi)
    analyzer.set_stop_line(line)
    
    print("Processing video...")
    video_path = "sample_traffic.mp4"
    
    count = 0
    try:
        for frame, stats in analyzer.process_video(video_path):
            count += 1
            if count % 10 == 0:
                print(f"Frame {count}: {stats}")
            if count >= 30: # Test first 30 frames
                break
        print("Test passed successfully.")
    except Exception as e:
        print(f"Test failed with error: {e}")

if __name__ == "__main__":
    test_system()
