Title: AI-Powered Traffic Queue Analysis and Rule Violation Detection using Video Analytics
Problem Overview
Urban intersections across India face severe congestion, frequent traffic violations, and
unsafe driving behavior. Most existing traffic signal systems rely on:
Fixed-time signaling
Manual monitoring
Limited adaptability to real-time traffic conditions
With the widespread availability of CCTV cameras, there is a strong opportunity
to use computer vision and video analytics to automatically analyze traffic flow and support
data-driven traffic management and enforcement.
Your challenge is to build
a vision-based traffic analysis system that processes traffic camera footage and extracts
actionable traffic intelligence such as queue length, traffic density, and rule violations.
What You Need to Build
Participants must design and implement a system that:
Processes traffic video footage from signalized intersections
Detects and tracks multiple vehicles over time
Estimates queue length and queue density before traffic signals
Identifies traffic rule violations such as red-light jumping and rash driving
Presents insights via a web-based dashboard
The focus is on computer vision accuracy, tracking consistency, and explainable analytics,
not on building a full production-grade system.
Core Technical Expectations (High Level)
Your solution should conceptually and/or practically address:
Vehicle Detection
Detection of multiple vehicle categories (cars, bikes, buses, trucks, autos, etc.)
Multi-Object Tracking
Consistent ID assignment across frames
Vehicle trajectory tracking
Queue Analytics
Queue length estimation (on basis of vehicle count)
Queue density estimation (vehicles per unit area)
Violation Detection
Red signal jump detection
Rash driving detection using motion or trajectory heuristics
Visualization & Reporting
Annotated video output
Real-time or near real-time metrics on a dashboard
Technical Constraints
1.Input: Must be pre-recorded traffic video footage (Indian road traffic is strongly
preferred)
2.Computer Vision: Use of Computer Vision is mandatory (deep learning–based or hybrid
approaches allowed)
3.Tracking: Multi-object tracking is required, frame-wise detection alone is insufficient
4.Deployment: Must include a web interface (minimum expectation: Streamlit)
5.Design: System should be modular and explainable
What NOT to Focus On
1.No hardware integration required
2.No real-time camera feed required
3.No traffic signal control logic required
4.No mobile app or frontend-heavy UI required
This is an analytics and perception problem, not an IoT deployment challenge.