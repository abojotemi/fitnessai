from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from datetime import datetime, timedelta
import time
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from llm import LLMHandler
import streamlit as st
import logging
from analytics_tab import log_diet_analysis

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Message:
    role: str
    content: str
    image: Optional[Any] = None
    timestamp: datetime = None

@dataclass
class AnalyticsData:
    total_analyses: int = 0
    successful_analyses: int = 0
    failed_analyses: int = 0
    avg_response_time: float = 0.0
    daily_usage: Dict[str, int] = None
    

class DietAnalyzer:
    def __init__(self):
        self.llm = LLMHandler()
        self._initialize_session_state()
        
    def _initialize_session_state(self):
        """Initialize or reset session state variables with analytics"""
        if "current_analysis" not in st.session_state:
            st.session_state.current_analysis = {
                'image': None,
                'response': None,
                'timestamp': None
            }
        if "diet_processing" not in st.session_state:
            st.session_state.diet_processing = False
        if "analysis_history" not in st.session_state:
            st.session_state.analysis_history = []
        if "analytics" not in st.session_state:
            st.session_state.diet_analytics = AnalyticsData(
                daily_usage={}
            )
        if "response_times" not in st.session_state:
            st.session_state.response_times = []
        if "uploaded_image" not in st.session_state:
            st.session_state.uploaded_image = None

    def display_current_analysis(self):
        """Display only the current image and analysis"""
        try:
            current = st.session_state.current_analysis
            
            if current['image']:
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.image(current['image'], width=300, caption="Uploaded Food Image")
                with col2:
                    if current['timestamp']:
                        st.caption(f"Uploaded: {current['timestamp'].strftime('%Y-%m-%d %H:%M')}")
            
            if current['response']:
                st.markdown("### Analysis Result")
                st.markdown(current['response'])
                
        except Exception as e:
            logger.error(f"Error in display_current_analysis: {str(e)}")
            st.error("Error displaying analysis")

    def _display_analysis_history(self):
        """Display historical analysis with enhanced filtering and visualization"""
        if not st.session_state.analysis_history:
            st.info("No analysis history yet. Start by analyzing some food images!")
            return
            
        st.subheader("📊 Analysis History")
        
        # Add date filter
        date_range = st.date_input(
            "Filter by date range",
            value=(
                datetime.now() - timedelta(days=7),
                datetime.now()
            ),
            key="history_date_filter"
        )
        
        # Filter entries by date
        filtered_entries = [
            entry for entry in st.session_state.analysis_history
            if date_range[0] <= entry['timestamp'].date() <= date_range[1]
        ]
        
        if not filtered_entries:
            st.warning("No entries found in selected date range.")
            return
        
        # Display entries in grid layout
        cols = st.columns(2)
        for idx, entry in enumerate(filtered_entries):
            with cols[idx % 2]:
                with st.expander(f"{entry['timestamp'].strftime('%Y-%m-%d %H:%M')}"):
                    st.image(entry['image'], width=200)
                    st.markdown(f"**Analysis:** {entry['analysis']}")
                    st.caption(f"Response Time: {entry['response_time']:.2f}s")
                    
                    # Add action buttons
                    col1, col2 = st.columns(2)
                    if col1.button(f"📋 Copy Analysis", key=f"copy_{idx}"):
                        st.write("Analysis copied to clipboard!")
                    if col2.button(f"⭐ Save as Favorite", key=f"fav_{idx}"):
                        st.write("Added to favorites!")
                    
                    st.divider()
                    
    def display_analytics(self):
        """Display usage analytics with interactive visualizations"""
        analytics = st.session_state.diet_analytics
        
        # Create metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Analyses", analytics.total_analyses)
        with col2:
            success_rate = (analytics.successful_analyses / analytics.total_analyses * 100) \
                if analytics.total_analyses > 0 else 0
            st.metric("Success Rate", f"{success_rate:.1f}%")
        with col3:
            st.metric("Avg Response Time", f"{analytics.avg_response_time:.2f}s")
        with col4:
            st.metric("Failed Analyses", analytics.failed_analyses)

        # Daily usage trend
        if analytics.daily_usage:
            dates = list(analytics.daily_usage.keys())
            counts = list(analytics.daily_usage.values())
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=dates,
                y=counts,
                mode='lines+markers',
                name='Daily Usage',
                line=dict(color='#FF4B4B')
            ))
            fig.update_layout(
                title='Daily Usage Trend',
                xaxis_title='Date',
                yaxis_title='Number of Analyses',
                height=300,
                margin=dict(l=0, r=0, t=40, b=0)
            )
            st.plotly_chart(fig, use_container_width=True)

        # Response time distribution
        if st.session_state.response_times:
            fig = px.histogram(
                x=st.session_state.response_times,
                nbins=20,
                title='Response Time Distribution',
                labels={'x': 'Response Time (s)', 'y': 'Count'},
                color_discrete_sequence=['#FF4B4B']
            )
            fig.update_layout(
                height=300,
                margin=dict(l=0, r=0, t=40, b=0)
            )
            st.plotly_chart(fig, use_container_width=True)
            
    def _handle_analysis(self, user_info: Dict[str, Any]):
        """Handle the analysis process with proper error handling and analytics tracking"""
        try:
            if st.session_state.uploaded_image is None:
                st.warning("⚠️ Please upload an image first.")
                return
                
            if not st.session_state.diet_processing:
                st.session_state.diet_processing = True
                
                # Process image and get analysis
                analysis = self.process_image(st.session_state.uploaded_image, user_info)
                
                if analysis:
                    # Update current analysis
                    st.session_state.current_analysis = {
                        'image': st.session_state.uploaded_image,
                        'response': analysis,
                        'timestamp': datetime.now()
                    }
                
                st.session_state.diet_processing = False
                # Clear the uploaded image after successful analysis
                st.session_state.uploaded_image = None
                st.rerun()
                
        except Exception as e:
            logger.error(f"Error in handle_analysis: {str(e)}")
            st.error("An error occurred while analyzing the image. Please try again.")
            st.session_state.diet_processing = False
            st.session_state.diet_analytics.failed_analyses += 1

    def _handle_clear_chat(self):
        """Handle clearing the current analysis"""
        try:
            # Reset current analysis
            st.session_state.current_analysis = {
                'image': None,
                'response': None,
                'timestamp': None
            }
            
            # Reset processing state
            st.session_state.diet_processing = False
            
            # Clear uploaded image
            st.session_state.uploaded_image = None
            
            st.rerun()
            
        except Exception as e:
            logger.error(f"Error in handle_clear_chat: {str(e)}")
            st.error("An error occurred while clearing the analysis. Please try again.")
            
    def process_image(self, image: Any, user_info: Dict[str, Any]) -> Optional[str]:
        """Process uploaded food image with performance tracking"""
        try:
            start_time = time.time()
            with st.spinner("🔍 Analyzing your food..."):
                analysis = self.llm.analyze_diet(image, user_info)
                
                # Update analytics
                end_time = time.time()
                response_time = end_time - start_time
                st.session_state.response_times.append(response_time)
                
                current_date = datetime.now().strftime('%Y-%m-%d')
                st.session_state.diet_analytics.daily_usage[current_date] = \
                    st.session_state.diet_analytics.daily_usage.get(current_date, 0) + 1
                
                st.session_state.diet_analytics.total_analyses += 1
                st.session_state.diet_analytics.successful_analyses += 1
                st.session_state.diet_analytics.avg_response_time = \
                    sum(st.session_state.response_times) / len(st.session_state.response_times)
                
                # Store analysis in history
                st.session_state.analysis_history.append({
                    'analysis': analysis,
                    'image': image,
                    'timestamp': datetime.now(),
                    'response_time': response_time,
                    'user_info': user_info
                })
                
                processing_time = time.time() - start_time
                log_diet_analysis(
                    success=True,
                    processing_time=processing_time,
                    user_id=user_info.name
                )
                
                return analysis
                
        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            st.session_state.diet_analytics.failed_analyses += 1
            st.error("Unable to process the image. Please try again.")
            return None

    def display(self, user_info: Dict[str, Any]):
        """Enhanced main render method with analytics and improved UI"""
        st.header("🥗 Diet Analyzer")
        
        # Create tabs for different views
        tab1, tab2, tab3 = st.tabs(["Analysis", "History", "Analytics"])
        
        with tab1:
            # Add quick tips
            with st.expander("📝 Tips for better analysis"):
                st.markdown("""
                - Ensure good lighting when taking food photos
                - Include all items in the frame
                - Take photos from above for best results
                - Include portion sizes if possible
                """)
            
            # Current analysis display
            
            # Upload section
            st.markdown("### Upload Food Image")
            upload_col1, upload_col2 = st.columns([2, 1])
            
            with upload_col1:
                st.session_state.uploaded_image = st.file_uploader(
                    "Choose an image of your food",
                    type=['jpg', 'jpeg', 'png', 'webp'],
                    key="diet_uploader",
                    help="For best results, ensure the food is clearly visible and well-lit"
                )
            
            with upload_col2:
                st.markdown("#### Supported formats:")
                st.markdown("- JPG/JPEG\n- PNG\n- WebP")

            # Action buttons with improved layout
            button_col1, button_col2, button_col3 = st.columns([1, 1, 2])
            
            analyze_button = button_col1.button(
                "🔍 Analyze",
                key="analyze_diet",
                disabled=st.session_state.diet_processing,
                use_container_width=True
            )
            
            clear_button = button_col2.button(
                "🗑️ Clear",
                key="clear_diet",
                use_container_width=True
            )
            if analyze_button:
                self._handle_analysis(user_info)
            
            if clear_button:
                self._handle_clear_chat()
                
            self.display_current_analysis()
            
        
        with tab2:
            self._display_analysis_history()
            
        with tab3:
            self.display_analytics()