from pathlib import Path
import time
import streamlit as st
from analytics_tab import log_response_time, log_tts_request, log_user_interaction
from llm import LLMHandler
from utils import TTSHandler
import logging
import pandas as pd
import plotly.express as px
from datetime import datetime


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class Workout:
    def __init__(self): 
        # Initialize session state variables if they don't exist
        if 'workout_plan' not in st.session_state:
            st.session_state.workout_plan = None
        if 'workout_summary' not in st.session_state:
            st.session_state.workout_summary = None
        if 'audio_html' not in st.session_state:
            st.session_state.audio_html = None
        if 'cache_path' not in st.session_state:
            st.session_state.cache_path = None
        if 'workout_analytics' not in st.session_state:
            st.session_state.workout_analytics = {
                'total_plans_generated': 0,
                'equipment_usage': {},
                'duration_preference': {},
                'focus_areas': {},
                'generation_times': []
            }
    def _update_analytics(self, workout_preferences, processing_time):
        """Update workout generation analytics."""
        analytics = st.session_state.workout_analytics
        
        # Increment total plans generated
        analytics['total_plans_generated'] += 1
        
        # Track equipment usage
        equipment = workout_preferences['equipment']
        analytics['equipment_usage'][equipment] = analytics['equipment_usage'].get(equipment, 0) + 1
        
        # Track duration preference
        duration = str(workout_preferences['duration']) + ' mins'
        analytics['duration_preference'][duration] = analytics['duration_preference'].get(duration, 0) + 1
        
        # Track focus areas
        for area in workout_preferences['focus_areas'].split(', '):
            analytics['focus_areas'][area] = analytics['focus_areas'].get(area, 0) + 1
        
        # Track generation times
        analytics['generation_times'].append({
            'timestamp': datetime.now(),
            'processing_time': processing_time
        })
        
    def _display_analytics(self):
        """Display workout generation analytics."""
        st.sidebar.header("🔍 Workout Generation Analytics")
        
        analytics = st.session_state.workout_analytics
        
        # Total Plans Generated
        st.sidebar.metric("Total Plans", analytics['total_plans_generated'])
        
        # Equipment Usage Pie Chart
        if analytics['equipment_usage']:
            st.sidebar.subheader("Equipment Usage")
            equipment_df = pd.DataFrame.from_dict(
                analytics['equipment_usage'], 
                orient='index', 
                columns=['Count']
            ).reset_index()
            equipment_df.columns = ['Equipment', 'Count']
            fig_equipment = px.pie(
                equipment_df, 
                values='Count', 
                names='Equipment', 
                title='Equipment Preferences'
            )
            st.sidebar.plotly_chart(fig_equipment, use_container_width=True)
        
        # Performance Tracking
        if analytics['generation_times']:
            st.sidebar.subheader("Performance Tracking")
            times_df = pd.DataFrame(analytics['generation_times'])
            times_df['processing_time_ms'] = times_df['processing_time'] * 1000
            
            # Line chart of processing times
            fig_performance = px.line(
                times_df, 
                x='timestamp', 
                y='processing_time_ms', 
                title='Workout Plan Generation Time'
            )
            st.sidebar.plotly_chart(fig_performance, use_container_width=True)


    def display(self):
        """Render workout tab content with enhanced functionality and error handling."""
        st.header("🏋️ Generate Your Personalized Workout Routine")
        
        # Add a motivational quote
        motivational_quotes = [
            "Every workout is a step towards a better you!",
            "Strength doesn't come from what you can do, it comes from overcoming what you thought you couldn't.",
            "Your body can stand almost anything. It's your mind that you have to convince.",
            "Success is walking from failure to failure with no loss of enthusiasm."
        ]
        st.markdown(f"*💬 Quote of the Day: {motivational_quotes[hash(str(datetime.now().date())) % len(motivational_quotes)]}*")
        
        # Equipment selection with icons
        equipment_options = {
            "No Equipment (Bodyweight) 🤸‍♀️": "No Equipment",
            "Basic Home Equipment 🏠": "Basic Home",
            "Full Gym Access 💪": "Full Gym",
            "Resistance Bands Only 🔗": "Resistance Bands",
            "Dumbbells Only 🏋️": "Dumbbells"
        }
        selected_equipment = st.selectbox(
            "What equipment do you have access to?",
            options=list(equipment_options.keys())
        )
        selected_equipment = equipment_options[selected_equipment]
        
        log_user_interaction('workout_generation', {'equipment': selected_equipment})
        
        # Workout duration preference
        duration_options = {
            "30 mins ⏱️": 30,
            "45 mins 🕒": 45,
            "60 mins ⌛": 60,
            "90 mins 🕓": 90
        }
        selected_duration = st.select_slider(
            "Preferred workout duration",
            options=list(duration_options.keys()),
            value="45 mins 🕒"
        )
        selected_duration = duration_options[selected_duration]

        # Workout frequency
        frequency_options = {
            "2-3 times/week 🔄": "2-3x",
            "3-4 times/week 📆": "3-4x",
            "4-5 times/week 💥": "4-5x",
            "6+ times/week 🚀": "6+x"
        }
        selected_frequency = st.select_slider(
            "Workout frequency",
            options=list(frequency_options.keys()),
            value="3-4 times/week 📆"
        )
        selected_frequency = frequency_options[selected_frequency]

        # Workout focus areas with multi-select
        focus_areas = st.multiselect(
            "Select focus areas",
            options=[
                "Strength Training 💪", 
                "Cardio 🏃", 
                "Flexibility 🧘", 
                "Core Strength 🌟", 
                "Weight Loss 🔥", 
                "Muscle Gain 💥", 
                "Endurance 🚴"
            ],
            default=["Strength Training 💪", "Cardio 🏃"]
        )
        # Remove emojis for backend processing
        focus_areas = [area.split()[0] for area in focus_areas]

        # Existing injuries or limitations
        limitations = st.text_area(
            "Any injuries or limitations we should consider? (Optional)",
            help="Enter any injuries, medical conditions, or limitations that might affect your workout"
        )

        if st.button("Generate Workout Routine 🚀", type="primary"):
            try:
                start_time = time.time()
                with st.spinner("Creating your personalized fitness journey..."):
                    # Prepare workout preferences
                    workout_preferences = {
                        "equipment": selected_equipment,
                        "duration": selected_duration,
                        "frequency": selected_frequency,
                        "focus_areas": ", ".join(focus_areas),
                        "limitations": limitations.strip() if limitations else "None"
                    }

                    # Initialize LLM handler
                    llm = LLMHandler()

                    # Generate the workout plan
                    fitness_plan = llm.generate_fitness_plan(
                        user_profile=st.session_state.user_info,
                        workout_preferences=workout_preferences
                    )

                    if fitness_plan:
                        processing_time = time.time() - start_time
                        
                        # Update analytics
                        self._update_analytics(workout_preferences, processing_time)
                        
                        log_response_time('workout_generation', processing_time)
                        log_user_interaction('workout_plan_generated', {
                                'duration': selected_duration,
                                'focus_areas': focus_areas
                            })

                        # Create tabs for different views of the workout plan
                        st.session_state.workout_plan = fitness_plan.content
                        st.session_state.workout_summary = llm.summarizer(fitness_plan.content)
                        if st.session_state.workout_summary:
                            start_time = time.time()
                            tts_handler = TTSHandler()
                            st.session_state.audio_html = tts_handler.text_to_speech(st.session_state.workout_summary)
                            st.session_state.cache_path = tts_handler._get_cache_path(
                                tts_handler._clean_text(st.session_state.workout_summary)
                            )
                            processing_time = time.time() - start_time
                            log_tts_request(
                                text_length=len(st.session_state.workout_summary),
                                processing_time=processing_time
                                )
                
                # Create workout plan tabs
                if st.session_state.workout_plan:
                    st.session_state.plan_tabs = st.tabs(["Complete Plan 📋", "Quick Summary 🎯", "Audio Guide 🎧"])
                    with st.session_state.plan_tabs[0]:
                        st.markdown("### 📋 Your Complete Workout Plan")
                        st.markdown(st.session_state.workout_plan)

                    with st.session_state.plan_tabs[1]:
                        st.markdown("### 🎯 Quick Summary")
                        if st.session_state.workout_summary:
                            st.markdown(st.session_state.workout_summary)

                    with st.session_state.plan_tabs[2]:
                        st.markdown("### 🎧 Audio Guide")
                        if st.session_state.audio_html:
                            st.markdown(st.session_state.audio_html, unsafe_allow_html=True)
                            
                            # Add download button for audio
                            if st.session_state.cache_path and Path(st.session_state.cache_path).exists():
                                with open(st.session_state.cache_path, 'rb') as f:
                                    audio_bytes = f.read()
                                    st.download_button(
                                        label="📥 Download Audio Guide",
                                        data=audio_bytes,
                                        file_name="workout_guide.mp3",
                                        mime="audio/mp3"
                                    )
                
                # Display analytics
                self._display_analytics()

            except Exception as e:
                logger.error(f"Error generating workout plan: {str(e)}")
                st.error("An error occurred while generating your workout plan. Please try again.")