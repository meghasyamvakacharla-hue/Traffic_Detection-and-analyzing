from src.processor import TrafficAnalyzer
import cv2
import os

def check_auto_config():
    if not os.path.exists("sample_traffic.mp4"):
        print("Sample video not found. Generating one...")
        import create_sample_video
        create_sample_video.generate_sample_video()

    print("Initializing Analyzer...")
    analyzer = TrafficAnalyzer(use_color_detection=True)
    
    print("Running Auto-Configuration Analysis...")
    roi, line = analyzer.analyze_traffic_pattern("sample_traffic.mp4")
    
    if roi and line:
        print("\nSUCCESS: Auto-configuration returned values.")
        print(f"Detected ROI: {roi}")
        print(f"Detected Stop Line: {line}")
        
        # Validation checks
        # ROI should be a list of 4 points
        if len(roi) != 4:
            print("FAILURE: ROI does not have 4 points.")
            exit(1)
            
        # Stop line should be a list of 2 points
        if len(line) != 2:
            print("FAILURE: Stop line does not have 2 points.")
            exit(1)
            
        print("Verification Passed.")
    else:
        print("\nFAILURE: Auto-configuration returned None.")
        exit(1)

if __name__ == "__main__":
    check_auto_config()
