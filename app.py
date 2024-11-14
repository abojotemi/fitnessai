# app.py
import streamlit as st
import plotly.express as px
from streamlit_option_menu import option_menu
from datetime import datetime
import pandas as pd
import pycountry
from pydantic import ValidationError
import base64
import logging
from config import AppConfig, UserInfo
from session_state import SessionState
from components import UIComponents
from utils import text_to_speech, speech_to_text
from chat_logic import DietAnalyzer
from progress_journal import initialize_progress_journal
from llm import LLMHandler

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

    def start_application(self):
        """Main application entry point"""
        self.ui.setup_page()
        st.title("🏋️‍♂️ Fit AI - Your Personal Fitness Coach")
        
        options = ["Profile", "Generate Workout", "Diet analyzer", "Questions", "Progress Journal", "Analytics"]
        selected = option_menu(
            menu_title=None,
            options=options,
            icons=['person', 'book', 'egg-fried', 'patch-question-fill', 'journal', 'graph-up-arrow'],
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
                self.display_questions_section()
            if selected == options[4]:
                self.display_progress_journal_section()
            if selected == options[5]:
                self.display_analytics_section()
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
                            min_value=self.config.DEFAULT_AGE,
                            value=st.session_state.user_info.age if st.session_state.user_info else self.config.DEFAULT_AGE
                        )
                        weight = st.number_input(
                            "Weight (kg)",
                            min_value=self.config.DEFAULT_WEIGHT,
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
                            min_value=self.config.DEFAULT_HEIGHT,
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
                        help="Describe your fitness goals (minimum 5 characters)"
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
                        # Create tabs for different views of the workout plan
                        plan_tabs = st.tabs(["Complete Plan", "Quick Summary", "Audio Guide"])

                        with plan_tabs[0]:
                            st.markdown("### 📋 Your Complete Workout Plan")
                            st.markdown(fitness_plan.content)

                        with plan_tabs[1]:
                            st.markdown("### 🎯 Quick Summary")
                            summary = llm.summarizer(fitness_plan.content)
                            if summary:
                                st.markdown(summary)

                        with plan_tabs[2]:
                            st.markdown("### 🎧 Audio Guide")
                            if summary:
                                audio_html = text_to_speech(summary)
                                st.markdown(audio_html, unsafe_allow_html=True)

                        # Add save/export options
                        col1, col2 = st.columns(2)
                        with col1:
                            if st.button("Save to Progress Journal"):
                                self._save_workout_plan(fitness_plan.content)
                                st.success("Workout plan saved to your progress journal!")

                        with col2:
                            # Create download button
                            if fitness_plan.content:
                                self._create_download_button(fitness_plan.content)

            except Exception as e:
                logger.error(f"Error generating workout plan: {str(e)}")
                st.error("An error occurred while generating your workout plan. Please try again.")

    def _save_workout_plan(self, plan_content: str):
        """Save workout plan to progress journal."""
        if 'workout_history' not in st.session_state:
            st.session_state.workout_history = []
        
        workout_entry = {
            'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'plan': plan_content
        }
        st.session_state.workout_history.append(workout_entry)

    def _create_download_button(self, plan_content: str):
        """Create download button for workout plan."""
        try:
            # Convert plan to MD format
            markdown_content = f"""# Your Personal Workout Plan
            Generated on: {datetime.now().strftime('%Y-%m-%d')}
            
            {plan_content}
            """
            
            # Create download button
            b64 = base64.b64encode(markdown_content.encode()).decode()
            href = f'<a href="data:text/markdown;base64,{b64}" download="workout_plan.md" class="downloadButton">📥 Download Workout Plan</a>'
            st.markdown(href, unsafe_allow_html=True)
        except Exception as e:
            logger.error(f"Error creating download button: {str(e)}")
            st.warning("Unable to create download button. Please try again.")
            # Generate workout routine
    def display_diet_section(self):
        """Render workout tab content"""
        if st.session_state.user_info:
            self.diet_analyzer.display_diet_analyzer(st.session_state.user_info)
        else:
            st.error("Please complete your profile first to use the Diet Analyzer.")
            logger.error("Please complete your profile first to use the Diet Analyzer.")
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
                        question = speech_to_text(audio_file)
                        st.session_state.transcribed_question = question
                        st.write("Transcribed question:", question)

            # Process question
            if st.button("Get Answer", type="primary") and (question or st.session_state.get('transcribed_question')):
                with st.spinner("Analyzing your question..."):
                    # Initialize LLM handler
                    llm = LLMHandler()
                    
                    # Get the final question text
                    final_question = question or st.session_state.get('transcribed_question')
                    
                    # Generate answer
                    response = llm.answer_question(final_question, st.session_state.user_info)
                    
                    if response:
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
    
    def display_analytics_section(self):
        """Render analytics dashboard tab content"""
        st.header("Analytics Dashboard")
        
        try:
            if st.session_state.progress_data:
                progress_dataframe = pd.DataFrame(st.session_state.progress_data)
                
                # Weight Progress Chart
                st.subheader("Weight Progress")
                fig_weight = px.line(
                    progress_dataframe,
                    x="date",
                    y="weight",
                    title="Weight Tracking",
                    labels={"weight": "Weight (kg)", "date": "Date"}
                )
                st.plotly_chart(fig_weight, use_container_width=True)
                
                # Mood and Intensity Distribution
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Energy Level Distribution")
                    fig_mood = px.pie(
                        progress_dataframe,
                        names="mood",
                        title="Energy Levels"
                    )
                    st.plotly_chart(fig_mood, use_container_width=True)
                
                with col2:
                    st.subheader("Workout Intensity Distribution")
                    fig_intensity = px.pie(
                        progress_dataframe,
                        names="intensity",
                        title="Workout Intensities"
                    )
                    st.plotly_chart(fig_intensity, use_container_width=True)
                
                # Export Data Option
                if st.button("Download Progress Report"):
                    try:
                        csv = progress_dataframe.to_csv(index=False)
                        b64 = base64.b64encode(csv.encode()).decode()
                        href = f'<a href="data:file/csv;base64,{b64}" download="fitness_progress.csv">Download CSV File</a>'
                        st.markdown(href, unsafe_allow_html=True)
                    except Exception as e:
                        logger.error(f"Error exporting data: {e}")
                        st.error("Unable to generate report. Please try again.")
                
            else:
                st.info("No progress data available yet. Start tracking your progress to see analytics!")
        
        except Exception as e:
            logger.error(f"Error in analytics tab: {e}")
            st.error("An error occurred while loading analytics. Please try again.")
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