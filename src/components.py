import streamlit as st
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class UIComponents:
    """UI Component handlers"""
    @staticmethod
    def setup_page():
        """Configure page settings and styling"""
        st.set_page_config(
            page_title="Fit AI",
            page_icon="💪",
            layout="wide",
            initial_sidebar_state="collapsed"
        )
        