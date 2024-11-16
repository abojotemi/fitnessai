from pathlib import Path
import time
import streamlit as st
from streamlit_option_menu import option_menu
import pycountry
from pydantic import ValidationError
import logging
from config import AppConfig, UserInfo
from session_state import SessionState
from components import UIComponents
from utils import TTSHandler, get_audio_duration, text_to_speech, speech_to_text
from diet_analysis import DietAnalyzer
from video_analysis import display_video_tab
from progress_journal import initialize_progress_journal
from llm import LLMHandler
from analytics_tab import display_analytics, log_user_interaction, log_tts_request, log_stt_request, log_response_time
from food_generator import FoodImageGenerator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FitnessCoachApp:
    """Main application class"""
    def __init__(self):
        self.config = AppConfig()
        SessionState.init_session_state()
        SessionState.load_progress_data()
        self.ui = UIComponents()
        self.diet_analyzer = DietAnalyzer()
        self.food_generator = FoodImageGenerator()

    def start_application(self):
        """Main application entry point"""
        self.ui.setup_page()
        st.title("🏋️‍♂️ Fitness AI - Your Personal Fitness Coach")
        
        options = ["Profile", "Generate Workout", "Diet analyzer", "Food Generator", "Questions", "RAG Video","Progress Journal", "Analytics"]
        selected = option_menu(
            menu_title=None,
            options=options,
            icons=['person', 'book', 'egg-fried', 'pencil', 'patch-question-fill', 'youtube', 'journal', 'graph-up-arrow'],
            default_index=0,
            orientation="horizontal",
        )

        if selected == 'Profile':
            self.display_profile_section()
            
        if st.session_state.profile_completed:
            if selected == options[1]:
                self.display_workout_section()
            if selected == options[2]:
                self.display_diet_section()
            if selected == options[3]:
                self.display_pose_section()
            if selected == options[4]:
                self.display_questions_section()
            if selected == options[5]:
                self.display_video_section()
            if selected == options[6]:
                self.display_progress_journal_section()
            if selected == options[7]:
                display_analytics()
        else:
            for option in options[1:]:
                if selected == option:
                    st.info("Please complete your profile first.")
    def display_profile_section(self):
            """Render profile tab content"""
            st.header("Your Profile")
            
            try:
                with st.form("profile_form"):
                    name = st.text_input(
                        "Name",
                        value=st.session_state.user_info.name if st.session_state.user_info else "",
                        help="Enter your full name (3-30 characters)"
                    )
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        age = st.number_input(
                            "Age",
                            min_value=self.config.MIN_AGE,
                            value=st.session_state.user_info.age if st.session_state.user_info else self.config.DEFAULT_AGE
                        )
                        weight = st.number_input(
                            "Weight (kg)",
                            min_value=self.config.MIN_WEIGHT,
                            value=st.session_state.user_info.weight if st.session_state.user_info else self.config.DEFAULT_WEIGHT
                        )
                    
                    with col2:
                        sex = st.selectbox(
                            "Sex",
                            options=["Male", "Female", "Other"],
                            index=["Male", "Female", "Other"].index(st.session_state.user_info.sex) if st.session_state.user_info else 0
                        )
                        height = st.number_input(
                            "Height (cm)",
                            min_value=self.config.MIN_HEIGHT,
                            value=st.session_state.user_info.height if st.session_state.user_info else self.config.DEFAULT_HEIGHT
                        )
                    
                    countries = [country.name for country in pycountry.countries]
                    country = st.selectbox(
                        "Country",
                        options=countries,
                        index=countries.index(st.session_state.user_info.country) if st.session_state.user_info else 234
                    )
                    
                    goals = st.text_area(
                        "Fitness Goals",
                        value=st.session_state.user_info.goals if st.session_state.user_info else "",
                        help="Describe your fitness goals (minimum 5 characters)",
                        placeholder="Gaining Muscle"
                    )
                    
                    submit = st.form_submit_button("Save Profile")
                    
                    if submit:
                        try:
                            info = UserInfo(
                                name=name,
                                age=age,
                                sex=sex,
                                weight=weight,
                                height=height,
                                goals=goals,
                                country=country
                            )
                            st.session_state.user_info = info
                            st.session_state.profile_completed = True
                            log_user_interaction('profile_update', {
                                    'age': age,
                                    'country': country
                                })
                            st.success("Profile saved successfully! 🎉")
                            st.balloons()
                        except ValidationError as e:
                            st.error(f"Please check your inputs: {str(e)}")
                        except Exception as e:
                            logger.error(f"Error saving profile: {e}")
                            st.error("An unexpected error occurred. Please try again.")
            
            except Exception as e:
                logger.error(f"Error rendering profile tab: {e}")
                st.error("An error occurred while loading the profile form. Please refresh the page.")


    def display_workout_section(self):
        """Render workout tab content with enhanced functionality and error handling."""
        st.header("Generate Your Workout Routine")
        # Initialize session state variables if they don't exist
        if 'workout_plan' not in st.session_state:
            st.session_state.workout_plan = None
        if 'workout_summary' not in st.session_state:
            st.session_state.workout_summary = None
        if 'audio_html' not in st.session_state:
            st.session_state.audio_html = None
        if 'cache_path' not in st.session_state:
            st.session_state.cache_path = None

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
                        plan_tabs = st.tabs(["Complete Plan", "Quick Summary", "Audio Guide"])
                        with plan_tabs[0]:
                            st.markdown("### 📋 Your Complete Workout Plan")
                            st.markdown(st.session_state.workout_plan)

                        with plan_tabs[1]:
                            st.markdown("### 🎯 Quick Summary")
                            if st.session_state.workout_summary:
                                st.markdown(st.session_state.workout_summary)


                        with plan_tabs[2]:
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

    def display_diet_section(self):
        """Render workout tab content"""
        if st.session_state.user_info:
            log_user_interaction('diet_analysis_start', {
                    'user_weight': st.session_state.user_info.weight
                })
            self.diet_analyzer.display_diet_analyzer(st.session_state.user_info)
        else:
            st.error("Please complete your profile first to use the Diet Analyzer.")
            logger.error("Please complete your profile first to use the Diet Analyzer.")
    
    def display_pose_section(self):
        if st.session_state.user_info:
            log_user_interaction('food_generation_start', {
                'feature': 'text_to_image'
            })
            self.food_generator.display_generator()
        else:
            st.error("Please complete your profile first to use the Posture Generator.")
            logger.error("Please complete your profile first to use the Posture Generator.")
            
    def display_questions_section(self):
        """Render questions tab with enhanced features"""
        st.header("Ask Your Fitness Questions 💪")
        
        try:
            # Text input for questions
            col1, col2 = st.columns([3, 1])
            with col1:
                question = st.text_area(
                    "Type your fitness question here",
                    help="Ask anything about workouts, nutrition, or general fitness advice",
                    placeholder="e.g., How can I improve my squat form?"
                )
            
            with col2:
                st.write("Or")
                audio_file = st.file_uploader(
                    "Upload audio question",
                    type=("mp3", "wav", "m4a"),
                    help="Record and upload your question as audio"
                )

            if audio_file:
                st.audio(audio_file)
                if st.button("Transcribe Audio"):
                    with st.spinner("Transcribing your question..."):
                        start_time = time.time()
                        question = speech_to_text(audio_file)
                        processing_time = time.time() - start_time
                        log_stt_request(
                            audio_duration=get_audio_duration(audio_file),  # You'll need to implement this
                            processing_time=processing_time
                        )
                        st.session_state.transcribed_question = question
                        st.write("Transcribed question:", question)

            # Process question
            if st.button("Get Answer", type="primary") and (question or st.session_state.get('transcribed_question')):
                start_time = time.time()
                with st.spinner("Analyzing your question..."):
                    # Initialize LLM handler
                    llm = LLMHandler()
                    
                    # Get the final question text
                    final_question = question or st.session_state.get('transcribed_question')
                    
                    # Generate answer
                    response = llm.answer_question(final_question, st.session_state.user_info)
                    
                    if response:
                        processing_time = time.time() - start_time
                        log_response_time('question_answering', processing_time)
                        # Display answer
                        st.markdown(response)
                        
                        # Generate and display follow-up questions
                        follow_ups = llm.get_follow_up_questions(final_question, response)
                        if follow_ups:
                            st.markdown("### Related Questions You Might Want to Ask:")
                            for i, q in enumerate(follow_ups, 1):
                                if st.button(f"🔄 {q}", key=f"follow_up_{i}"):
                                    st.session_state.transcribed_question = q
                        
                        # Audio option
                        if st.button("🔊 Listen to the answer"):
                            audio_html = text_to_speech(response)
                            st.markdown(audio_html, unsafe_allow_html=True)

        except Exception as e:
            logger.error(f"Error in questions section: {str(e)}")
            st.error("An error occurred while processing your question. Please try again.")

    def display_video_section(self):
        display_video_tab()
    def display_progress_journal_section(self):
        """Render progress journal tab"""
        st.header("Progress Journal 📔")
        
        if not st.session_state.get('progress_journal_ui'):
            st.session_state.progress_journal_ui = initialize_progress_journal()
        
        # Create tabs for adding entries and viewing history
        journal_sections = st.tabs(["Add Entry", "View Progress"])
        
        with journal_sections[0]:
            st.session_state.progress_journal_ui.display_entry_form()
        
        with journal_sections[1]:
            st.session_state.progress_journal_ui.display_progress_view()

if __name__ == "__main__":
    try:
        app = FitnessCoachApp()
        app.start_application()
    except Exception as e:
        logger.error(f"Application error: {e}")
        st.error("An unexpected error occurred. Please refresh the page.")