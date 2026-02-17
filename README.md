# AI-Powered Traffic Analysis System

This system uses Computer Vision (YOLOv8) to analyze traffic video footage for:
- Vehicle Detection & Tracking
- Queue Length & Density Estimation
- Rule Violation Detection (Red light jumping, Rash driving)

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the application:
   ```bash
   streamlit run app.py
   ```

3. Upload a traffic video to start analysis.

## Usage
- Use the sidebar to upload a video.
- **Auto-Adjust**: The app automatically analyzes the video on upload to set the detection area. You can also click "Auto-Adjust Settings" to re-calibrate.
- Click "Start Analysis".
- Monitor real-time statistics on the dashboard.

## Features
- **Automatic ROI Detection**: Analyzes traffic patterns to set detection zones automatically.
- **Optimized Performance**: efficiently handles large video files.
- **Real-time Analysis**: Detects vehicles, estimates queue length, and identifies rule violations.
