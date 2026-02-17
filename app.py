import streamlit as st
import tempfile
import cv2
import numpy as np
import os
import pandas as pd
from src.processor import TrafficAnalyzer

st.set_page_config(page_title="AI Traffic Analysis", layout="wide", page_icon="🚦")

# --- Custom CSS ---
st.markdown("""
<style>
    .stApp { background-color: #0e1117; }
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

# --- Helper Functions ---
@st.cache_resource
def load_model():
    return TrafficAnalyzer()

def get_status(density):
    if density < 0.3:
        return "Low", "status-low"
    elif density < 0.7:
        return "Medium", "status-medium"
    else:
        return "High", "status-high"

def render_metrics(q_len, density, violations):
    status_text, status_class = get_status(density)
    return f"""
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
            <div class="metric-value" style="color: #ff4b4b;">{violations}</div>
            <div class="metric-label">Violations</div>
        </div>
    </div>
    """

# --- Sidebar Configuration ---
sidebar = st.sidebar
sidebar.header("Configuration")

# Video Source Selection
st.sidebar.subheader("Input Source")
uploaded_file = st.sidebar.file_uploader("Upload Traffic Video", type=["mp4", "avi", "mov"])

# Initialize variables
video_path = None
queue_roi = []
stop_line = []
width = 1920
height = 1080
fps = 30

if uploaded_file:
    # Save temp file
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    uploaded_file.seek(0)
    tfile.write(uploaded_file.read())
    tfile.close()
    video_path = tfile.name

    # Get video properties
    cap = cv2.VideoCapture(video_path)
    if cap.isOpened():
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        ret, frame = cap.read()
        cap.release()
        
        if ret:
            st.sidebar.markdown("### ROI Configuration")
            
            # Queue ROI Sliders
            with st.sidebar.expander("Queue Detection Area", expanded=True):
                q_x1 = st.slider("Queue X1", 0, width, int(width*0.3))
                q_y1 = st.slider("Queue Y1", 0, height, int(height*0.4))
                q_x2 = st.slider("Queue X2", 0, width, int(width*0.7))
                q_y2 = st.slider("Queue Y2", 0, height, int(height*0.6))
                queue_roi = [(q_x1, q_y1), (q_x2, q_y1), (q_x2, q_y2), (q_x1, q_y2)]
            
            # Stop Line Sliders
            with st.sidebar.expander("Stop Line", expanded=True):
                l_y = st.slider("Stop Line Y Position", 0, height, int(height*0.8))
                l_x1 = st.slider("Stop Line X1", 0, width, int(width*0.2))
                l_x2 = st.slider("Stop Line X2", 0, width, int(width*0.8))
                stop_line = [(l_x1, l_y), (l_x2, l_y)]
                
            # Preview
            preview_frame = frame.copy()
            pts = np.array(queue_roi, np.int32)
            cv2.polylines(preview_frame, [pts], True, (0, 255, 255), 2)
            cv2.line(preview_frame, stop_line[0], stop_line[1], (0, 255, 0), 2)
            st.sidebar.image(preview_frame, channels="BGR", caption="ROI Preview", use_container_width=True)


# --- Main Layout ---
metrics_placeholder = st.empty()

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### Video Analysis")
    video_placeholder = st.empty()

with col2:
    st.markdown("### Analysis Trends")
    graphs_placeholder = st.empty()


if 'is_running' not in st.session_state:
    st.session_state['is_running'] = False

def start_analysis():
    st.session_state['is_running'] = True

def stop_analysis():
    st.session_state['is_running'] = False

if not st.session_state['is_running']:
    st.sidebar.button("Start Analysis", on_click=start_analysis, type="primary")
else:
    st.sidebar.button("Stop Analysis", on_click=stop_analysis, type="secondary")

# --- Application Logic ---
if st.session_state['is_running']:
    if not video_path:
         st.error("Please upload a video first.")
         st.session_state['is_running'] = False

    
    if video_path:
        # Load fresh analyzer instance each run but model is cached inside logic if updated
        # However, for now, we instantiate directly.
        # To truly cache model loading, we need to modify processor.py or wrap it.
        # But 'load_model' (TrafficAnalyzer()) might not work if TrafficAnalyzer isn't designed for reuse.
        
        # Let's instantiate fresh for safety, relying on internal improvements if any.
        # Use default YOLO model (no color mode)
        analyzer = TrafficAnalyzer(use_color_detection=False)
        analyzer.set_roi(queue_roi)
        analyzer.set_stop_line(stop_line)
        
        graph_data = {"Queue Length": [], "Queue Density": []}
        
        try:
            # FRAME SKIPPING: Skip 2 frames (process every 3rd) to speed up playback
            for processed_frame, stats in analyzer.process_video(video_path, skip_frames=2):
                if not st.session_state['is_running']:
                    break
                    
                # Update Metrics
                metrics_html = render_metrics(
                    stats['Queue Length'], 
                    stats['Queue Density'], 
                    stats['Violations']
                )
                metrics_placeholder.markdown(metrics_html, unsafe_allow_html=True)
                
                # Update Video
                frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                video_placeholder.image(frame_rgb, use_container_width=True, channels="RGB")
                
                # Update Graphs
                graph_data["Queue Length"].append(stats['Queue Length'])
                graph_data["Queue Density"].append(stats['Queue Density'])
                df = pd.DataFrame(graph_data)
                graphs_placeholder.line_chart(df)
                
        except Exception as e:
            st.error(f"An error occurred during analysis: {e}")
        finally:
            # Clean up temp file only if we are done or it's a new run
            # Note: Deleting while running might cause issues if open.
            # Here we might choose NOT to delete immediately to allow seek/replay
            pass

            # If finished naturally, stop running state? 
            # Maybe keep it True so user can see final results? 
            # Or reset to allow restart? Usually reset.
            # st.session_state['is_running'] = False 
            # But if we reset here, UI might flicker or clear. Better let user stop or restart manually.
    else:
        st.warning("Please upload a video to begin.")
        st.session_state['is_running'] = False
else:
    # Initial State
    metrics_placeholder.markdown(render_metrics(0, 0.0, 0), unsafe_allow_html=True)
