import cv2
import numpy as np

def generate_sample_video(filename='sample_traffic.mp4', duration=10, fps=30):
    width, height = 1280, 720
    # Use 'mp4v' or 'h264' if available. 'avc1' sometimes works better on Windows.
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter(filename, fourcc, fps, (width, height))

    # Vehicles: list of [x, y, w, h, color, speed]
    vehicles = []
    
    # Spawn lanes
    lanes = [200, 500, 800, 1100]
    
    for i in range(duration * fps):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Draw road
        cv2.rectangle(frame, (0, 0), (width, height), (100, 100, 100), -1)
        for x in lanes:
            cv2.line(frame, (x, 0), (x, height), (255, 255, 255), 2)
            
        # Draw Stop Line
        cv2.line(frame, (0, 200), (width, 200), (0, 255, 0), 5) # Green line
        cv2.putText(frame, "Signal: GREEN", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Spawn new vehicles randomly
        if i % 30 == 0: # Every 1 second
            lane = lanes[np.random.randint(0, len(lanes))] - 100 # Center in lane
            color = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
            # x, y, w, h, color, speed
            vehicles.append([lane, 720, 60, 100, color, np.random.randint(3, 8)])

        # Update and Draw Vehicles
        for v in vehicles:
            v[1] -= v[5] # Move up
            cv2.rectangle(frame, (v[0], v[1]), (v[0]+v[2], v[1]+v[3]), v[4], -1)
        
        # Remove off-screen vehicles
        vehicles = [v for v in vehicles if v[1] > -150]
        
        out.write(frame)
        
    out.release()
    return True

if __name__ == "__main__":
    generate_sample_video()
    print("Sample video created.")
