from src.processor import TrafficAnalyzer
import cv2
import os
import pandas as pd

def verify_full_flow():
    if not os.path.exists("sample_traffic.mp4"):
        print("Sample video not found. Generating one...")
        # Fallback to generate if missing (unlikely since we made it)
        import create_sample_video
        create_sample_video.create_sample_video()

    print("Initializing Analyzer...")
    analyzer = TrafficAnalyzer()
    
    # Simulating the UI inputs
    roi = [(200, 200), (600, 200), (600, 600), (200, 600)]
    line = [(150, 500), (950, 500)]
    
    analyzer.set_roi(roi)
    analyzer.set_stop_line(line)
    
    print("Starting processing loop...")
    video_path = "sample_traffic.mp4"
    
    data_log = []
    
    try:
        frame_count = 0
        for frame, stats in analyzer.process_video(video_path):
            frame_count += 1
            # Check if stats contain all keys
            required_keys = ["Queue Length", "Signal", "Violations", "Queue Density"]
            for k in required_keys:
                if k not in stats:
                    raise ValueError(f"Missing key in stats: {k}")
            
            data_log.append(stats)
            
            if frame_count % 30 == 0:
                print(f"Processed {frame_count} frames. Current Stats: {stats}")
                
        print("Processing complete.")
        
        # Verify DataFrame creation works (as used in app)
        df = pd.DataFrame(data_log)
        print("DataFrame created successfully.")
        print(df.head())
        print(f"Total Frames: {len(df)}")
        
    except Exception as e:
        print(f"Verification FAILED: {e}")
        exit(1)

if __name__ == "__main__":
    verify_full_flow()
