from pathlib import Path
import streamlit as st
from streamlit_option_menu import option_menu
import logging
from config import AppConfig
from question import Question
from session_state import SessionState
from diet_analysis import DietAnalyzer
from video_analysis import VideoAnalyzer
from progress_journal import initialize_progress_journal
from food_generator import FoodImageGenerator
from workout import Workout
from profile_tab import Profile

# Configure logging with more detailed format
st.set_page_config(
            page_title="Fit AI",
            page_icon="💪",
            layout="wide",
            initial_sidebar_state="expanded"
        )

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class FitnessCoachApp:
    """
    Main application class for the Fitness AI Coach.
    Handles initialization of components and routing between different sections.
    """
    
    # Class constants
    APP_TITLE = "🏋️‍♂️ Fitness AI - Your Personal Fitness Coach"
    NAVIGATION_OPTIONS = [
        "Profile", "Generate Workout", "Diet analyzer", 
        "Food Generator", "Questions", "Video Analyzer",
        "Progress Journal"
    ]
    NAVIGATION_ICONS = [
        'person', 'book', 'egg-fried', 'pencil', 
        'patch-question-fill', 'youtube', 'journal', 
        
    ]
    
    def __init__(self):
        """Initialize application components and state"""
        try:
            # Initialize core components
            self._initialize_components()
            self.config = AppConfig()
            SessionState.init_session_state()
            SessionState.load_progress_data()
            
            logger.info("Successfully initialized FitnessCoachApp")
        except Exception as e:
            logger.error(f"Failed to initialize FitnessCoachApp: {e}")
            raise

    def _initialize_components(self) -> None:
        """Initialize all application components with error handling"""
        try:
            self.diet_analyzer = DietAnalyzer()
            self.food_generator = FoodImageGenerator()
            self.profile = Profile()
            self.workout = Workout()
            self.question = Question()
            self.video_analyzer = VideoAnalyzer()
            logger.info("Successfully initialized all components")
        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
            raise

    def start_application(self) -> None:
        """Main application entry point and routing logic"""
        try:
            st.title(self.APP_TITLE)
            
            # Create horizontal navigation menu
            selected = option_menu(
                menu_title=None,
                options=self.NAVIGATION_OPTIONS,
                icons=self.NAVIGATION_ICONS,
                default_index=0,
                orientation="horizontal",
            )
            
            # Route to selected section
            self._route_to_section(selected)
            
        except Exception as e:
            logger.error(f"Error in application execution: {e}")
            st.error("An unexpected error occurred. Please refresh the page.")

    def _route_to_section(self, selected: str) -> None:
        """Route to appropriate section based on selection"""
        try:
            # Handle Profile section separately as it doesn't require profile completion
            if selected == 'Profile':
                self.display_profile_section()
                return

            # Check profile completion for other sections
            if not st.session_state.profile_completed:
                st.info("Please complete your profile first.")
                return

            # Route to appropriate section
            section_map = {
                'Generate Workout': self.display_workout_section,
                'Diet analyzer': self.display_diet_section,
                'Food Generator': self.display_pose_section,
                'Questions': self.display_questions_section,
                'Video Analyzer': self.display_video_section,
                'Progress Journal': self.display_progress_journal_section,
            }

            if selected in section_map:
                section_map[selected]()
                logger.info(f"Successfully routed to {selected} section")

        except Exception as e:
            logger.error(f"Error routing to section {selected}: {e}")
            st.error(f"Error loading {selected} section. Please try again.")

    def display_profile_section(self) -> None:
        """Display profile section with error handling"""
        try:
            self.profile.display(self.config)
            logger.info("Successfully displayed profile section")
        except Exception as e:
            logger.error(f"Error displaying profile section: {e}")
            st.error("Unable to load profile section. Please try again.")

    def display_workout_section(self) -> None:
        """Display workout section with error handling"""
        try:
            self.workout.display()
            logger.info("Successfully displayed workout section")
        except Exception as e:
            logger.error(f"Error displaying workout section: {e}")
            st.error("Unable to load workout section. Please try again.")

    def display_diet_section(self) -> None:
        """Display diet analyzer section with error handling"""
        try:
            if st.session_state.user_info:
                self.diet_analyzer.display(st.session_state.user_info)
                logger.info("Successfully displayed diet analyzer section")
            else:
                raise ValueError("User info not found in session state")
        except Exception as e:
            logger.error(f"Error displaying diet analyzer section: {e}")
            st.error("Unable to load diet analyzer. Please ensure your profile is complete.")

    def display_pose_section(self) -> None:
        """Display food generator section with error handling"""
        try:
            if st.session_state.user_info:
                self.food_generator.display()
                logger.info("Successfully displayed food generator section")
            else:
                raise ValueError("User info not found in session state")
        except Exception as e:
            logger.error(f"Error displaying food generator section: {e}")
            st.error("Unable to load food generator. Please ensure your profile is complete.")

    def display_questions_section(self) -> None:
        """Display questions section with error handling"""
        try:
            self.question.display()
            logger.info("Successfully displayed questions section")
        except Exception as e:
            logger.error(f"Error displaying questions section: {e}")
            st.error("Unable to load questions section. Please try again.")

    def display_video_section(self) -> None:
        """Display video analyzer section with error handling"""
        try:
            self.video_analyzer.display(st.session_state.user_info)
            logger.info("Successfully displayed video analyzer section")
        except Exception as e:
            logger.error(f"Error displaying video analyzer section: {e}")
            st.error("Unable to load video analyzer. Please try again.")

    def display_progress_journal_section(self) -> None:
        """Display progress journal section with error handling"""
        try:
            st.header("Progress Journal 📔")
            
            # Initialize progress journal UI if not already done
            if not st.session_state.get('progress_journal_ui'):
                st.session_state.progress_journal_ui = initialize_progress_journal()
                logger.info("Initialized progress journal UI")
            
            # Create tabs for journal sections
            journal_sections = st.tabs(["Add Entry", "View Progress"])
            
            with journal_sections[0]:
                st.session_state.progress_journal_ui.display_entry_form()
            
            with journal_sections[1]:
                st.session_state.progress_journal_ui.display_progress_view()
                
            logger.info("Successfully displayed progress journal section")
        except Exception as e:
            logger.error(f"Error displaying progress journal section: {e}")
            st.error("Unable to load progress journal. Please try again.")

def main():
    """Main entry point with error handling"""
    try:
        app = FitnessCoachApp()
        app.start_application()
    except Exception as e:
        logger.error(f"Critical application error: {e}")
        st.error("""
            An unexpected error occurred while starting the application. 
            Please refresh the page or contact support if the issue persists.
        """)

if __name__ == "__main__":
    main()