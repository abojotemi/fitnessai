import streamlit as st

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
        