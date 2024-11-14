# ui_components.py
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
        
        st.markdown("""
            <style>
            .stButton>button {
                background-color: #0066cc;
                color: white;
                border-radius: 5px;
                padding: 10px 20px;
                transition: all 0.3s ease;
            }
            .stButton>button:hover {
                background-color: #0052a3;
                transform: translateY(-2px);
            }
            .metric-card {
                background-color: #f0f2f6;
                padding: 20px;
                border-radius: 10px;
                text-align: center;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            .graph-container {
                background: white;
                padding: 20px;
                border-radius: 10px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            </style>
        """, unsafe_allow_html=True)