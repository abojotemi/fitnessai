from datetime import datetime, timedelta
import tempfile
import time
import streamlit as st
import os
import logging
from dotenv import load_dotenv
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image
import io
import pandas as pd
from collections import defaultdict
from llm import LLMHandler
from utils import TTSHandler
from analytics_tab import log_video_analysis

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
load_dotenv()

class VideoAnalyzer:
    def __init__(self):
        self._initialize_session_state()
        self.llm = LLMHandler()
        self.tts_handler = TTSHandler()

    def _initialize_session_state(self):
        sessions = {
            'present_video': None,
            'current_answer': None,
            'total_processed': 0,
            'processing': False,
            'answer_history': [],
            'video_history': [],
            'audio_answer': None,
            'path': '',
            'last_video_id': None,
            'favorites': set(),
            'video_stats': {
                'daily_uploads': 0,
                'weekly_uploads': 0,
                'total_duration': 0
            },
            'tags': [],
            # New analytics tracking
            'video_analytics': {
                'processing_times': [],
                'daily_usage': defaultdict(int),
                'error_counts': defaultdict(int),
                'user_engagement': {
                    'total_users': 0,
                    'active_users': 0,
                    'returning_users': 0
                },
                'model_performance': {
                    'success_rate': 100.0,
                    'avg_processing_time': 0.0,
                    'total_errors': 0
                },
                'usage_by_country': defaultdict(int),
                'popular_workout_types': defaultdict(int)
            }
        }

        for key, val in sessions.items():
            if key not in st.session_state:
                st.session_state[key] = val

    def _update_analytics(self, success=True, processing_time=None, error_type=None, user_info=None):
        analytics = st.session_state.video_analytics
        
        # Update processing metrics
        if processing_time:
            analytics['processing_times'].append(processing_time)
            analytics['model_performance']['avg_processing_time'] = sum(analytics['processing_times']) / len(analytics['processing_times'])

        # Update error tracking
        if not success:
            analytics['model_performance']['total_errors'] += 1
            if error_type:
                analytics['error_counts'][error_type] += 1

        # Update success rate
        total_attempts = len(analytics['processing_times'])
        if total_attempts > 0:
            success_rate = ((total_attempts - analytics['model_performance']['total_errors']) / total_attempts) * 100
            analytics['model_performance']['success_rate'] = round(success_rate, 2)

        # Update geographic usage
        if user_info and hasattr(user_info, 'country'):
            analytics['usage_by_country'][user_info.country] += 1

        # Update daily usage
        current_date = datetime.now().strftime('%Y-%m-%d')
        analytics['daily_usage'][current_date] += 1


    def _create_activity_chart(self):
        # Create sample data for the week
        dates = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        
        # Get current day index (0 = Monday, 6 = Sunday)
        current_day = datetime.now().weekday()
        
        # Create values array with actual uploads for current day and zeros for other days
        values = [0] * 7
        values[current_day] = st.session_state.video_stats['daily_uploads']

        # Create the figure dictionary
        fig_dict = {
            'data': [{
                'type': 'bar',
                'x': dates,
                'y': values,
                'marker': {'color': '#FF4B4B'}
            }],
            'layout': {
                'title': 'Weekly Activity',
                'xaxis': {'title': 'Day'},
                'yaxis': {'title': 'Videos Analyzed'},
                'height': 300,
                'margin': {'l': 20, 'r': 20, 't': 40, 'b': 20},
                'paper_bgcolor': 'rgba(0,0,0,0)',
                'plot_bgcolor': 'rgba(0,0,0,0)'
            }
        }
        
        return fig_dict
    def _create_analytics_charts(self):
        analytics = st.session_state.video_analytics
        
        # Processing Time Trend
        if analytics['processing_times']:
            fig_processing = go.Figure()
            fig_processing.add_trace(go.Scatter(
                y=analytics['processing_times'],
                mode='lines+markers',
                name='Processing Time'
            ))
            fig_processing.update_layout(
                title='Processing Time Trend',
                yaxis_title='Time (seconds)',
                height=300
            )
        else:
            fig_processing = None

        # Success Rate Gauge
        fig_success = go.Figure(go.Indicator(
            mode="gauge+number",
            value=analytics['model_performance']['success_rate'],
            title={'text': "Success Rate"},
            domain={'x': [0, 1], 'y': [0, 1]},
            gauge={'axis': {'range': [0, 100]},
                  'bar': {'color': "#FF4B4B"}}
        ))
        fig_success.update_layout(height=250)

        # Geographic Usage
        if analytics['usage_by_country']:
            country_data = pd.DataFrame(
                list(analytics['usage_by_country'].items()),
                columns=['Country', 'Usage Count']
            )
            fig_geo = px.bar(country_data, x='Country', y='Usage Count',
                           title='Usage by Country')
            fig_geo.update_layout(height=300)
        else:
            fig_geo = None

        return fig_processing, fig_success, fig_geo

    def _add_tags(self, video_content):
        """
        Extract workout types from video analysis content.
        Args:
            video_content (str): The analysis text from the LLM
        Returns:
            list: List of identified workout tags
        """
        # Common workout types to look for in the analysis
        common_workouts = [
            'cardio', 'strength', 'yoga', 'pilates', 'stretch', 
            'HIIT', 'bodyweight', 'weightlifting', 'crossfit',
            'powerlifting', 'calisthenics', 'resistance', 'conditioning',
            'functional', 'flexibility'
        ]
        
        tags = []
        # Convert content to lowercase for case-insensitive matching
        content_lower = video_content.lower()
        
        # Check for each workout type in the content
        for workout in common_workouts:
            if workout.lower() in content_lower:
                # Add the workout type in its original format (not lowercase)
                
                tags.append(workout)
                
        # Add intensity tags if certain keywords are found
        if any(word in content_lower for word in ['high intensity', 'intense', 'advanced']):
            tags.append('High Intensity')
        elif any(word in content_lower for word in ['moderate', 'intermediate']):
            tags.append('Moderate Intensity')
        elif any(word in content_lower for word in ['low intensity', 'beginner', 'gentle']):
            tags.append('Low Intensity')
        
        # Deduplicate tags while preserving order
        return list(dict.fromkeys(tags))

    def display(self, user_info):
        st.title("🏋️ Workout Video Analyzer")
        
        tabs = st.tabs(["Main Dashboard", "Analytics Dashboard"])
        
        with tabs[0]:
            # Original dashboard content
            col1, col2, col3 = st.columns([1, 2, 1])
            
            with col1:
                st.markdown("### 📊 Analytics")
                with st.container():
                    st.markdown('<div class="metrics-card">', unsafe_allow_html=True)
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.metric("Videos Analyzed", st.session_state.total_processed)
                    with col_b:
                        st.metric("Today's Uploads", st.session_state.video_stats['daily_uploads'])
                    st.markdown('</div>', unsafe_allow_html=True)
                
                st.plotly_chart(self._create_activity_chart(), use_container_width=True)
                
                with st.expander("ℹ️ Help", expanded=False):
                    st.markdown("""
                    ### Quick Guide
                    1. 📤 Upload your workout video
                    2. ⏳ Wait for AI analysis
                    3. 📝 View detailed feedback
                    4. ⭐ Save favorites for reference
                    
                    ### Best Practices
                    - Ensure good lighting
                    - Keep camera stable
                    - Film from multiple angles
                    - Wear contrasting clothes
                    """)

            with col2:
                st.markdown("## 🎥 Video Analysis")
                uploaded_file = st.file_uploader(
                    "Upload your workout video",
                    type=['mp4', 'mpeg', 'webm', 'avi', 'mpg', 'wmv', '3gpp', 'x-flv'],
                    key="video_upload",
                    help="Supported formats: MP4, AVI, WebM, etc."
                )

                if uploaded_file:
                    video_id = f"{uploaded_file.name}_{datetime.now().timestamp()}"
                    
                    if st.session_state.last_video_id != video_id:
                        st.session_state.current_answer = None
                        st.session_state.processing = False
                        st.session_state.last_video_id = video_id
                        
                        temp_dir = "./vid_files"
                        if not os.path.exists(temp_dir):
                            os.makedirs(temp_dir)
                        st.session_state.path = os.path.join(temp_dir, uploaded_file.name)
                        with open(st.session_state.path, "wb") as f:
                            f.write(uploaded_file.getvalue())
                        
                        st.session_state.present_video = uploaded_file
                        st.session_state.video_stats['daily_uploads'] += 1

                if st.session_state.present_video:
                    st.video(st.session_state.present_video)
                    
                    col_x, col_y = st.columns([3, 1])
                    with col_x:
                        analyze_button = st.button("🔍 Analyze Video", key="analyze_button", use_container_width=True)
                    with col_y:
                        if st.button("🔄 Reset", key="reset_button"):
                            st.session_state.current_answer = None
                            st.session_state.present_video = None
                            st.rerun()
                    
                    if analyze_button:
                        st.session_state.processing = True
                        st.session_state.current_answer = None
                    
                    if st.session_state.processing:
                        with st.spinner('🤖 AI is analyzing your workout...'):
                            start_time = time.time()
                            try:
                                response = self.llm.video_analyzer_llm(st.session_state.path, user_info)
                                if response:
                                    st.session_state.current_answer = response
                                    st.session_state.total_processed += 1
                                    
                                    tags = self._add_tags(response)
                                    for tag in tags:
                                        st.session_state.video_analytics['popular_workout_types'][tag] += 1
                                    
                                    video_entry = {
                                        'id': st.session_state.last_video_id,
                                        'title': uploaded_file.name,
                                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                        'content': st.session_state.current_answer,
                                        'tags': tags
                                    }
                                    st.session_state.video_history.append(video_entry)
                                    
                                    processing_time = time.time() - start_time
                                    self._update_analytics(
                                        success=True,
                                        processing_time=processing_time,
                                        user_info=user_info
                                    )
                                    
                                    log_video_analysis(True, processing_time, video_type='workout')
                                    
                            except Exception as e:
                                st.error(f"❌ Analysis failed: {str(e)}")
                                self._update_analytics(
                                    success=False,
                                    error_type=type(e).__name__,
                                    user_info=user_info
                                )
                                log_video_analysis(False, processing_time, video_type='workout', error=str(e))
                            finally:
                                st.session_state.processing = False
                    
                    if st.session_state.current_answer:
                        st.markdown("### 📝 Analysis Results")
                        st.markdown('<div class="success-message">', unsafe_allow_html=True)
                        st.markdown(st.session_state.current_answer)
                        st.markdown('</div>', unsafe_allow_html=True)
            with col3:
                st.markdown("### 📚 History")
                
                filter_options = ['All']
                if st.session_state.video_history:
                    all_tags = set()
                    for video in st.session_state.video_history:
                        all_tags.update(video.get('tags', []))
                    filter_options.extend(sorted(all_tags))
                
                selected_filter = st.selectbox("Filter by workout type:", filter_options)
                
                for video in reversed(st.session_state.video_history):
                    if selected_filter == 'All' or selected_filter in video.get('tags', []):
                        with st.expander(f"📺 {video['title'][:20]}...", expanded=False):
                            st.write(f"📅 {video['timestamp']}")
                            if video.get('tags'):
                                st.markdown(" ".join([f"`{tag}`" for tag in video['tags']]))
                            
                            if video['id'] in st.session_state.favorites:
                                if st.button("❤️ Favorited", key=f"fav_{video['id']}"):
                                    st.session_state.favorites.remove(video['id'])
                            else:
                                if st.button("🤍 Add to Favorites", key=f"fav_{video['id']}"):
                                    st.session_state.favorites.add(video['id'])
                            
                            st.divider()
                            st.markdown(video['content'])

        with tabs[1]:
            st.markdown("## 📊 Usage Analytics")
            
            # Model Performance Metrics
            st.markdown("### 🎯 Model Performance")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Success Rate", f"{st.session_state.video_analytics['model_performance']['success_rate']}%")
            with col2:
                avg_time = st.session_state.video_analytics['model_performance']['avg_processing_time']
                st.metric("Avg Processing Time", f"{avg_time:.2f}s")
            with col3:
                st.metric("Total Errors", st.session_state.video_analytics['model_performance']['total_errors'])

            # Processing Time Trend
            st.markdown("### ⏱️ Processing Time Trends")
            fig_processing, fig_success, fig_geo = self._create_analytics_charts()
            if fig_processing:
                st.plotly_chart(fig_processing, use_container_width=True)

            # Success Rate Gauge
            st.markdown("### 📈 Success Rate")
            st.plotly_chart(fig_success, use_container_width=True)

            # Geographic Usage
            if fig_geo:
                st.markdown("### 🌎 Geographic Usage")
                st.plotly_chart(fig_geo, use_container_width=True)

            # Popular Workout Types
            st.markdown("### 💪 Popular Workout Types")
            workout_data = pd.DataFrame(
                list(st.session_state.video_analytics['popular_workout_types'].items()),
                columns=['Workout Type', 'Count']
            ).sort_values('Count', ascending=False)
            
            if not workout_data.empty:
                fig_workouts = px.bar(
                    workout_data,
                    x='Workout Type',
                    y='Count',
                    title='Most Popular Workout Types'
                )
                st.plotly_chart(fig_workouts, use_container_width=True)

            # Error Analysis
            if st.session_state.video_analytics['error_counts']:
                st.markdown("### ❌ Error Analysis")
                error_data = pd.DataFrame(
                    list(st.session_state.video_analytics['error_counts'].items()),
                    columns=['Error Type', 'Count']
                ).sort_values('Count', ascending=False)
                
                fig_errors = px.pie(
                    error_data,
                    values='Count',
                    names='Error Type',
                    title='Error Distribution'
                )
                st.plotly_chart(fig_errors, use_container_width=True)