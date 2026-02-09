import streamlit as st
import tempfile
import cv2
import numpy as np
import os
import pandas as pd
from src.processor import TrafficAnalyzer

st.set_page_config(page_title="AI Traffic Analysis", layout="wide", page_icon="🚦")

# Custom CSS for the dashboard to match the screenshot style
st.markdown("""
<style>
    .stApp {
        background-color: #0e1117;
    }
    .metric-card {
        background-color: #1e2130;
        border: 1px solid #2b2f42;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 20px;
    }
    .metric-value {
        font-size: 3rem;
        font-weight: bold;
        color: #ffffff;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #a0a0a0;
        text-transform: uppercase;
        margin-top: 5px;
    }
    .status-low { color: #00ff00; }
    .status-medium { color: #ffff00; }
    .status-high { color: #ff0000; }
</style>
""", unsafe_allow_html=True)

sidebar = st.sidebar
sidebar.header("Configuration")

# Video Source
sidebar.subheader("Input Source")
uploaded_file = sidebar.file_uploader("Upload Traffic Video", type=["mp4", "avi", "mov"])
use_sample = sidebar.checkbox("Use Sample Video (sample_traffic.mp4)")
use_color_mode = sidebar.checkbox("Use Color Detection (for synthetic video)", value=True)

# ROI Configuration
sidebar.subheader("Queue Detection Area (Yellow Box)")
q_x1 = sidebar.slider("Queue X1", 0, 1920, 639)
q_y1 = sidebar.slider("Queue Y1", 0, 1080, 501)
q_x2 = sidebar.slider("Queue X2", 0, 1920, 782)
q_y2 = sidebar.slider("Queue Y2", 0, 1080, 505)

queue_roi = [(q_x1, q_y1), (q_x2, q_y1), (q_x2, q_y2), (q_x1, q_y2)]

# Line Configuration
sidebar.subheader("Stop Line (Signal Line)")
l_y = sidebar.slider("Stop Line Y Position", 0, 1080, 168)
l_x1 = sidebar.slider("Stop Line X1", 0, 1920, 313)
l_x2 = sidebar.slider("Stop Line X2", 0, 1920, 950)

stop_line = [(l_x1, l_y), (l_x2, l_y)]

# Helper to determine Status
def get_status(density):
    if density < 0.3:
        return "Low", "status-low"
    elif density < 0.7:
        return "Medium", "status-medium"
    else:
        return "High", "status-high"

# Layout: Top Metrics Row
metrics_placeholder = st.empty()

# Layout: Video Row with two columns
st.markdown("### Video Analysis")
col1, col2 = st.columns(2)

with col1:
    st.markdown("**Processed Feed (with detection)**")
    video_placeholder = st.empty()

with col2:
    st.markdown("**Original Video**")
    original_video_placeholder = st.empty()

# Layout: Graphs Row
st.markdown("### Analysis Trends")
graphs_placeholder = st.empty()


start_button = sidebar.button("Start Analysis")

if start_button:
    video_path = None
    if use_sample:
        if os.path.exists("sample_traffic.mp4"):
            video_path = "sample_traffic.mp4"
        else:
            st.error("Sample video not found. Please run 'create_sample_video.py' or upload a file.")
    elif uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_file.read())
        video_path = tfile.name
    
    if video_path:
        # Show original video in the right column
        original_video_placeholder.video(video_path)
        
        analyzer = TrafficAnalyzer(use_color_detection=use_color_mode)
        analyzer.set_roi(queue_roi)
        analyzer.set_stop_line(stop_line)
        
        # Data storage for graphs
        graph_data = {"Queue Length": [], "Queue Density": []}
        
        try:
            for processed_frame, stats in analyzer.process_video(video_path):
                # Update Metrics using HTML/CSS for custom look
                q_len = stats['Queue Length']
                density = stats['Queue Density']
                viol = stats['Violations']
                status_text, status_class = get_status(density)
                
                metrics_html = f"""
                <div style="display: flex; justify-content: space-between; gap: 20px;">
                    <div class="metric-card" style="flex: 1;">
                        <div class="metric-value">{q_len}</div>
                        <div class="metric-label">Total Vehicles</div>
                    </div>
                    <div class="metric-card" style="flex: 1;">
                        <div class="metric-value">{density:.4f}</div>
                        <div class="metric-label">Density</div>
                    </div>
                    <div class="metric-card" style="flex: 1;">
                        <div class="metric-value {status_class}">{status_text}</div>
                        <div class="metric-label">Status</div>
                    </div>
                    <div class="metric-card" style="flex: 1;">
                        <div class="metric-value" style="color: #ff4b4b;">{viol}</div>
                        <div class="metric-label">Violations</div>
                    </div>
                </div>
                """
                metrics_placeholder.markdown(metrics_html, unsafe_allow_html=True)
                
                # Update Video
                frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                video_placeholder.image(frame_rgb, use_container_width=True, channels="RGB")
                
                # Update Graphs
                graph_data["Queue Length"].append(q_len)
                graph_data["Queue Density"].append(density)
                df = pd.DataFrame(graph_data)
                graphs_placeholder.line_chart(df)
                
        except Exception as e:
            st.error(f"An error occurred: {e}")
            
    else:
        st.warning("Please upload a video or select the sample video.")
else:
    # Render empty metrics state for initial view
    metrics_html = f"""
    <div style="display: flex; justify-content: space-between; gap: 20px;">
        <div class="metric-card" style="flex: 1;">
            <div class="metric-value">0</div>
            <div class="metric-label">Total Vehicles</div>
        </div>
        <div class="metric-card" style="flex: 1;">
            <div class="metric-value">0.0000</div>
            <div class="metric-label">Density</div>
        </div>
        <div class="metric-card" style="flex: 1;">
            <div class="metric-value status-low">Low</div>
            <div class="metric-label">Status</div>
        </div>
        <div class="metric-card" style="flex: 1;">
            <div class="metric-value">0</div>
            <div class="metric-label">Violations</div>
        </div>
    </div>
    """
    metrics_placeholder.markdown(metrics_html, unsafe_allow_html=True)
