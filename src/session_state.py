import streamlit as st
import json
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SessionState:
    """Manage Streamlit session state"""
    @staticmethod
    def init_session_state():
        """Initialize session state variables if they don't exist"""
        if 'profile_completed' not in st.session_state:
            st.session_state.profile_completed = False
        if 'user_info' not in st.session_state:
            st.session_state.user_info = None
        if 'progress_data' not in st.session_state:
            st.session_state.progress_data = []
        if 'workout_history' not in st.session_state:
            st.session_state.workout_history = []

    @staticmethod
    def save_progress_data():
        """Save progress data to local storage"""
        try:
            with open('progress_data.json', 'w') as f:
                json.dump(st.session_state.progress_data, f)
        except Exception as e:
            logger.error(f"Error saving progress data: {e}")

    @staticmethod
    def load_progress_data():
        """Load progress data from local storage"""
        try:
            with open('progress_data.json', 'r') as f:
                st.session_state.progress_data = json.load(f)
        except FileNotFoundError:
            st.session_state.progress_data = []
        except Exception as e:
            logger.error(f"Error loading progress data: {e}")
            st.session_state.progress_data = []