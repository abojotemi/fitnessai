import streamlit as st
import plotly.express as px
from datetime import datetime
import pandas as pd
import pycountry
import base64
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field, ValidationError
from dataclasses import dataclass
import logging
import json
from chat_logic import DietAnalyzer
from progress_journal import initialize_progress_journal
from llm import LLMHandler
from gtts import gTTS
import re
import io
import assemblyai as aai
import os
from dotenv import load_dotenv
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
@dataclass
class AppConfig:
    VALID_IMAGE_TYPES: tuple = ("png", "jpg", "jpeg", "gif", "webp")
    MAX_IMAGE_SIZE: int = 10 * 1024 * 1024 
    DEFAULT_WEIGHT: float = 30.0
    DEFAULT_HEIGHT: float = 120.0
    DEFAULT_AGE: int = 10

class UserInfo(BaseModel):
    """User profile data model with validation"""
    name: str = Field(min_length=3, max_length=30)
    age: int = Field(gt=10)
    sex: str = Field(min_length=3)
    weight: float = Field(ge=30)
    height: float = Field(ge=120)
    goals: str = Field(min_length=5)
    country: str = Field(min_length=3)

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

class FitnessCoachApp:
    """Main application class"""
    def __init__(self):
        self.config = AppConfig()
        SessionState.init_session_state()
        SessionState.load_progress_data()
        self.ui = UIComponents()
        self.diet_analyzer = DietAnalyzer() 
    def remove_markdown_formatting(self, text):
        """Remove common Markdown symbols using regular expressions"""
        # Remove headings (e.g., # Heading)
        text = re.sub(r'#', '', text)
        # Remove emphasis (e.g., **bold** or *italic*)
        text = re.sub(r'\*{1,2}', '', text)
        # Remove underscores (e.g., _italic_)
        text = re.sub(r'_', '', text)
        # Remove links (e.g., [text](url))
        text = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', text)
        # Remove extra spaces
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def speech_to_text(self, audio_file):

        aai.settings.api_key = os.getenv("ASSEMBLYAI_API_KEY")
        transcriber = aai.Transcriber()

        transcript = transcriber.transcribe(audio_file)
        return transcript.text
    def text_to_speech(self,text):
        """Convert text to speech and return the audio player HTML"""
        # Create gTTS object
        tts = gTTS(text=self.remove_markdown_formatting(text), lang='en')
        
        # Create a BytesIO object and write the audio to it
        audio_fp = io.BytesIO()
        tts.write_to_fp(audio_fp)
        
        # Rewind the BytesIO object to the beginning
        audio_fp.seek(0)
        
        # Encode the audio content to base64
        audio_str = base64.b64encode(audio_fp.read()).decode()
        
        # Generate HTML audio player
        audio_html = f'''
            <audio controls autoplay>
                <source src="data:audio/mp3;base64,{audio_str}" type="audio/mp3">
            </audio>
        '''
        
        return audio_html
        
    def start_application(self):
        """Main application entry point"""
        self.ui.setup_page()
        st.title("🏋️‍♂️ Fit AI - Your Personal Fitness Coach")
        
        # Create tabs
        tabs = st.tabs(["Profile", "Generate Workout routine", "Diet analyzer", "Questions", "Progress Journal", "Analytics"])
        
        # Render appropriate tab content
        with tabs[0]:
            self.display_profile_section()
            
        if st.session_state.profile_completed:
            with tabs[1]:
                self.display_workout_section()
            with tabs[2]:
                self.display_diet_section()
            with tabs[3]:
                self.display_questions_section()
            with tabs[4]:
                self.display_progress_journal_section()
            with tabs[5]:
                self.display_analytics_section()
        else:
            for tab in tabs[1:]:
                with tab:
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
                                audio_html = self.text_to_speech(summary)
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
        """Render questions tab with audio transcription and LLM answers"""
        st.header("Ask your questions here")
        
        try:
            audio_file = st.file_uploader("Upload your questions here (Audio file)", type=("mp3", "wav", "m4a"))
            
            if audio_file:
                st.audio(audio_file)
                
                # First step: Transcribe audio
                if st.button("Generate transcript"):
                    with st.spinner("Transcribing your questions..."):
                        transcribed_speech = self.speech_to_text(audio_file)
                        # Store transcribed text in session state
                        st.session_state.transcribed_speech = transcribed_speech
                
                # Show text area if we have transcribed text
                if 'transcribed_speech' in st.session_state:
                    transcribed_speech = st.text_area(
                        "Edit your question",
                        value=st.session_state.transcribed_speech,
                        height=200
                    )
                    
                    # Second step: Generate answer
                    if st.button("Answer question"):
                        llm = LLMHandler()
                        with st.spinner("Answering the question..."):
                            response = llm.answer_question(transcribed_speech, st.session_state.user_info)
                            st.markdown(response)
                            
        except Exception as e:
            logger.error(f"Error in questions tab: {str(e)}")
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