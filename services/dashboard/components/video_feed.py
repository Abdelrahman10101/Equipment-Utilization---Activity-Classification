"""
Dashboard - Video Feed Component

Displays the annotated video feed from the CV processor.
"""
import base64
import logging

import cv2
import numpy as np
import streamlit as st

logger = logging.getLogger("VideoFeed")


def render_video_feed(frame_data: dict):
    """
    Render the annotated video frame in Streamlit.

    Args:
        frame_data: Dict with 'frame_data' (base64 JPEG) and metadata
    """
    if not frame_data or "frame_data" not in frame_data:
        st.info("⏳ Waiting for video frames...")
        return

    try:
        # Decode base64 frame
        frame_b64 = frame_data["frame_data"]
        frame_bytes = base64.b64decode(frame_b64)
        frame_array = np.frombuffer(frame_bytes, dtype=np.uint8)
        frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)

        if frame is not None:
            # Convert BGR to RGB for Streamlit
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            st.image(
                frame_rgb,
                caption=f"Frame #{frame_data.get('frame_id', '?')} | "
                        f"Time: {frame_data.get('timestamp', 'N/A')} | "
                        f"Video: {frame_data.get('video_name', 'N/A')}",
                use_column_width=True,
            )
        else:
            st.warning("Failed to decode frame")

    except Exception as e:
        logger.error(f"Error rendering frame: {e}")
        st.error(f"Frame rendering error: {e}")


def render_placeholder():
    """Show placeholder when no frames are available."""
    st.markdown(
        """
        <div style="
            background: linear-gradient(135deg, #1e1e2e 0%, #2d2d44 100%);
            border: 2px dashed #555;
            border-radius: 12px;
            padding: 60px 20px;
            text-align: center;
            color: #888;
        ">
            <h3>📹 Video Feed</h3>
            <p>Waiting for processed video frames...</p>
            <p style="font-size: 0.9em; color: #666;">
                Make sure the CV Processor is running and processing video.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
