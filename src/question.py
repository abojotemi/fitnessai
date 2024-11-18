import logging
import time
from datetime import datetime
import streamlit as st
from video_generator import VideoGenerator
from analytics_tab import log_response_time
from llm import LLMHandler
from utils import speech_to_text
import os

# Configure logging with more detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('app.log')
    ]
)
logger = logging.getLogger(__name__)

class Question:
    """Handles question answering functionality with speech support"""
    
    SUPPORTED_AUDIO_FORMATS = ["mp3", "wav", "m4a", "aac"]
    MAX_FOLLOW_UP_QUESTIONS = 3
    
    def __init__(self):
        """Initialize Question handler"""
        logger.info("Initializing Question handler")
        self.llm = LLMHandler()
        self.video_generator = VideoGenerator()
        self._initialize_session_state()
        logger.debug("Question handler initialized successfully")
    
    def _initialize_session_state(self) -> None:
        """Initialize session state variables"""
        default_states = {
            'transcribed_question': None,
            'question_history': [],
            'current_question': '',
            'audio_file': None,
            'last_response': None,
            'video_path': None,
            'is_video_generating': False, 
            'pending_video_generation': False,
            'show_video_button': True,
        }
        
        for key, default_value in default_states.items():
            if key not in st.session_state:
                st.session_state[key] = default_value
                logger.debug(f"Initialized session state: {key} = {default_value}")
    
    def display(self) -> None:
        """Render questions tab with enhanced features"""
        st.header("Ask Your Fitness Questions 💪")
        
        try:
            self._render_input_section()
            self._handle_audio_transcription()
            self._process_question()
            
            # Create video section
            c1,c2 = st.columns(2)
            with c1:
                self._display_video_section()
            with c2:
                self._display_question_history()
            
        except Exception as e:
            logger.error(f"Error in questions section: {e}", exc_info=True)
            st.error("An error occurred while processing your question. Please try again.")
    
    def _display_video_section(self) -> None:
        """Handle video display and generation status"""
        try:
            # Check for pending video generation first
            if st.session_state.pending_video_generation:
                self._handle_pending_video_generation()
                return

            # Video generation button
            if (st.session_state.last_response and 
                not st.session_state.is_video_generating and 
                st.session_state.show_video_button):
                if st.button("🎥 Generate Video", key="generate_video"):
                    st.session_state.pending_video_generation = True
                    st.session_state.show_video_button = False
                    st.rerun()  # Use rerun instead of st.rerun()

            # Video display - Add debug logging and error handling
            if st.session_state.video_path:
                logger.info(f"Attempting to display video from path: {st.session_state.video_path}")
                try:
                    st.markdown("### Generated Video")
                    st.divider()
                    # Convert path to string and ensure it exists
                    video_path = str(st.session_state.video_path)
                    if not os.path.exists(video_path):
                        logger.error(f"Video file not found at path: {video_path}")
                        st.error("Video file not found. Please try generating again.")
                        return
                    # Display video with error catching
                    st.video(video_path)
                    logger.info("Video displayed successfully")
                except Exception as e:
                    logger.error(f"Error displaying video: {e}", exc_info=True)
                    st.error("Error displaying video. Please try generating again.")
        except Exception as e:
            logger.error(f"Error in video section: {e}", exc_info=True)
            st.error("An error occurred in the video section. Please try again.")

    def _handle_pending_video_generation(self) -> None:
        """Handle pending video generation"""
        try:
            logger.info("Starting video generation process")
            st.session_state.is_video_generating = True
            
            with st.spinner("Generating Video..."):
                video_path = self.video_generator.generate_video(st.session_state.last_response)
                
                if video_path:
                    logger.info(f"Video generated successfully: {video_path}")
                    st.session_state.video_path = video_path
                    # Force streamlit to recognize the state change
                    st.session_state["_video_display_key"] = time.time()
                else:
                    logger.error("Video generation failed")
                    st.error("Unable to generate a video.")
                    st.session_state.show_video_button = True
                
        except Exception as e:
            logger.error(f"Error in video generation: {e}", exc_info=True)
            st.error("Failed to generate video. Please try again.")
            st.session_state.show_video_button = True
        
        finally:
            st.session_state.is_video_generating = False
            st.session_state.pending_video_generation = False
            st.rerun()  # Force a rerun after video generation    
        
    def _generate_video_callback(self, response: str) -> None:
        """Callback function for video generation button"""
        try:
            logger.info("Starting video generation process")
            st.session_state.is_video_generating = True
            st.session_state.show_video_button = False
            
            video_path = self.video_generator.generate_video(response)
            
            if video_path:
                logger.info(f"Video generated successfully: {video_path}")
                st.session_state.video_path = video_path
                st.session_state.is_video_generating = False
            else:
                logger.error("Video generation failed")
                st.error("Unable to generate a video.")
                st.session_state.show_video_button = True
                st.session_state.is_video_generating = False
            
            st.rerun()
                
        except Exception as e:
            logger.error(f"Error in video generation: {e}", exc_info=True)
            st.error("Failed to generate video. Please try again.")
            st.session_state.show_video_button = True
            st.session_state.is_video_generating = False
            st.rerun()
    
    def _render_input_section(self) -> None:
        """Render the question input section"""
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.session_state.current_question = st.text_area(
                "Type your fitness question here",
                value=st.session_state.current_question,
                help="Ask anything about workouts, nutrition, or general fitness advice",
                placeholder="e.g., How can I improve my squat form?",
                key="question_input"
            )
        
        with col2:
            st.write("Or")
            st.session_state.audio_file = st.file_uploader(
                "Upload audio question",
                type=self.SUPPORTED_AUDIO_FORMATS,
                help="Record and upload your question as audio",
                key="audio_upload"
            )
    
    def _handle_audio_transcription(self) -> None:
        """Handle audio file transcription"""
        if st.session_state.audio_file:
            st.audio(st.session_state.audio_file)
            
            if st.button("🎤 Transcribe Audio", key="transcribe_button"):
                logger.info("Starting audio transcription")
                start_time = time.time()
                
                try:
                    with st.spinner("Transcribing your question..."):
                        question = speech_to_text(st.session_state.audio_file)
                        processing_time = time.time() - start_time
                        
                        if question:
                            st.session_state.transcribed_question = question
                            st.session_state.current_question = question
                            st.write("Transcribed question:", question)
                            log_response_time('audio_transcription', processing_time)
                            logger.info(f"Audio transcription completed in {processing_time:.2f} seconds")
                        else:
                            st.error("Unable to transcribe audio. Please try again.")
                            
                except Exception as e:
                    logger.error(f"Error in audio transcription: {e}", exc_info=True)
                    st.error("Failed to transcribe audio. Please try again or type your question.")
    
    def _process_question(self) -> None:
        """Process and answer the question"""
        question = (st.session_state.current_question or 
                   st.session_state.transcribed_question)
        
        if st.button("💡 Get Answer", type="primary", key="answer_button") and question:
            logger.info(f"Processing question: {question[:100]}...")
            start_time = time.time()
            
            try:
                with st.spinner("Analyzing your question..."):
                    response = self.llm.answer_question(question, st.session_state.user_info)
                    
                    if response:
                        processing_time = time.time() - start_time
                        self._handle_successful_response(question, response, processing_time)
                        # Reset video-related states for new question
                        st.session_state.video_path = None
                        st.session_state.show_video_button = True
                        st.session_state.is_video_generating = False
                        st.session_state.pending_video_generation = False
                    else:
                        st.error("Unable to generate an answer. Please try rephrasing your question.")
                        
            except Exception as e:
                logger.error(f"Error processing question: {e}", exc_info=True)
                st.error("Failed to process your question. Please try again.")
    
    def _handle_successful_response(self, question: str, response: str, processing_time: float) -> None:
        """Handle successful question response"""
        try:
            # Log metrics
            log_response_time('question_answering', processing_time)
            logger.info(f"Question answered in {processing_time:.2f} seconds")
            
            # Store in history
            self._update_question_history(question, response)
            
            # Display response
            st.markdown("### Answer:")
            st.markdown(response)
            st.session_state.last_response = response
            
        except Exception as e:
            logger.error(f"Error handling response: {e}", exc_info=True)
            st.error("Error displaying response. Please try asking your question again.")
    
    def _update_question_history(self, question: str, response: str) -> None:
        """Update question history with new Q&A pair"""
        history_entry = {
            'question': question,
            'response': response,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        st.session_state.question_history.append(history_entry)
        logger.debug("Added new entry to question history")

    def _display_question_history(self) -> None:
        """Display question history with filtering options"""
        if st.session_state.question_history:
            st.markdown("### Previous Questions")
            
            # Reverse history to show most recent first
            for entry in reversed(st.session_state.question_history[-5:]):  # Show last 5 questions
                with st.expander(f"Q: {entry['question'][:100]}... ({entry['timestamp']})"):
                    st.markdown(f"**Question:** {entry['question']}")
                    st.markdown(f"**Answer:** {entry['response']}")