import cv2
import numpy as np

def create_sample_video(output_path='sample_traffic.mp4', duration=10, fps=30, width=1280, height=720):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Simple car simulation
    class Car:
        def __init__(self, x, y, speed, color):
            self.x = x
            self.y = y
            self.speed = speed
            self.color = color
            self.width = 60
            self.height = 100

        def move(self):
            self.y += self.speed
            if self.y > height:
                self.y = -self.height

        def draw(self, frame):
            cv2.rectangle(frame, (int(self.x), int(self.y)), (int(self.x + self.width), int(self.y + self.height)), self.color, -1)

    cars = []
    import random
    for i in range(5): # 5 lanes
        lane_x = 200 + i * 150
        for j in range(3): # 3 cars per lane initially
            y = random.randint(-500, height)
            speed = random.randint(5, 15)
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            cars.append(Car(lane_x, y, speed, color))

    for _ in range(duration * fps):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Draw road
        cv2.rectangle(frame, (150, 0), (950, height), (100, 100, 100), -1)
        # Draw lanes
        for i in range(1, 5):
            x = 200 + i * 150 - 25 # line position
            cv2.line(frame, (x, 0), (x, height), (255, 255, 255), 2)
            
        # Draw cars
        for car in cars:
            car.move()
            car.draw(frame)
            
        out.write(frame)
        
    out.release()
    print(f"Sample video saved to {output_path}")

if __name__ == "__main__":
    create_sample_video()
