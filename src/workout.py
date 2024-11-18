from pathlib import Path
import time
import streamlit as st
from analytics_tab import log_response_time, log_tts_request, log_user_interaction
from llm import LLMHandler
from utils import TTSHandler
import logging


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
    def display(self):
            """Render workout tab content with enhanced functionality and error handling."""
            st.header("Generate Your Workout Routine")
            # Equipment selection
            equipment_options = [
                "No Equipment (Bodyweight)",
                "Basic Home Equipment",
                "Full Gym Access",
                "Resistance Bands Only",
                "Dumbbells Only"
            ]
            selected_equipment = st.selectbox(
                "What equipment do you have access to?",
                options=equipment_options
            )
            log_user_interaction('workout_generation', {'equipment': selected_equipment})
            # Workout duration preference
            duration_options = {
                "30 mins": 30,
                "45 mins": 45,
                "60 mins": 60,
                "90 mins": 90
            }
            selected_duration = st.select_slider(
                "Preferred workout duration",
                options=list(duration_options.keys()),
                value="45 mins"
            )

            # Workout frequency
            frequency_options = {
                "2-3 times per week": "2-3x",
                "3-4 times per week": "3-4x",
                "4-5 times per week": "4-5x",
                "6+ times per week": "6+x"
            }
            selected_frequency = st.select_slider(
                "Preferred workout frequency",
                options=list(frequency_options.keys()),
                value="3-4 times per week"
            )

            # Workout focus areas (multiple selection)
            focus_areas = st.multiselect(
                "Select focus areas",
                options=["Strength Training", "Cardio", "Flexibility", "Core Strength", 
                        "Weight Loss", "Muscle Gain", "Endurance"],
                default=["Strength Training", "Cardio"]
            )

            # Existing injuries or limitations
            limitations = st.text_area(
                "Any injuries or limitations we should consider? (Optional)",
                help="Enter any injuries, medical conditions, or limitations that might affect your workout"
            )

            if st.button("Generate Workout Routine", type="primary"):
                try:
                    start_time = time.time()
                    with st.spinner("Creating your personalized fitness journey..."):
                        # Prepare workout preferences
                        workout_preferences = {
                            "equipment": selected_equipment,
                            "duration": duration_options[selected_duration],
                            "frequency": frequency_options[selected_frequency],
                            "focus_areas": ", ".join(focus_areas),  # Convert list to string
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
                            log_response_time('workout_generation', processing_time)
                            log_user_interaction('workout_plan_generated', {
                                    'duration': duration_options[selected_duration],
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
                    if st.session_state.workout_plan:
                            st.session_state.plan_tabs = st.tabs(["Complete Plan", "Quick Summary", "Audio Guide"])
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
                except Exception as e:
                    logger.error(f"Error generating workout plan: {str(e)}")
                    st.error("An error occurred while generating your workout plan. Please try again.")