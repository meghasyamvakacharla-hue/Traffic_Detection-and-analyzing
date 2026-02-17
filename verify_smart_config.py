from src.processor import TrafficAnalyzer
import cv2
import os

def check_smart_config():
    if not os.path.exists("sample_traffic.mp4"):
        print("Sample video not found. Generating one...")
        import create_sample_video
        create_sample_video.generate_sample_video()

    video_path = "sample_traffic.mp4"
    print("Testing Smart Auto-Detect Logic...")
    
    # 1. Try standard detection (YOLO) - Expecting None for sample video
    print("Step 1: Attempting YOLO detection...")
    temp_analyzer = TrafficAnalyzer(use_color_detection=False)
    detected_roi, detected_line = temp_analyzer.analyze_traffic_pattern(video_path)
    
    if detected_roi:
        print("Surprise! YOLO detected the sample video objects.")
    else:
        print("YOLO returned None (as expected for synthetic video).")
        
        # 2. Fallback to color detection
        print("Step 2: Fallback to Color detection...")
        temp_analyzer = TrafficAnalyzer(use_color_detection=True)
        detected_roi, detected_line = temp_analyzer.analyze_traffic_pattern(video_path)
        
    if detected_roi and detected_line:
        print("\nSUCCESS: Smart detection returned values.")
        print(f"Detected ROI: {detected_roi}")
        print(f"Detected Stop Line: {detected_line}")
    else:
        print("\nFAILURE: Smart detection returned None.")
        exit(1)

if __name__ == "__main__":
    check_smart_config()
