import logging
import time
from typing import List, Optional, Dict, Any
from datetime import datetime
import streamlit as st

from analytics_tab import log_response_time
from llm import LLMHandler
from utils import speech_to_text, text_to_speech

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
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
        self._initialize_session_state()
    
    def _initialize_session_state(self) -> None:
        """Initialize session state variables"""
        default_states = {
            'transcribed_question': None,
            'question_history': [],
            'current_question': '',
            'audio_file': None,
            'last_response': None
        }
        
        for key, default_value in default_states.items():
            if key not in st.session_state:
                st.session_state[key] = default_value
                logger.debug(f"Initialized session state: {key}")
    
    def display(self) -> None:
        """Render questions tab with enhanced features"""
        st.header("Ask Your Fitness Questions 💪")
        
        try:
            self._render_input_section()
            self._handle_audio_transcription()
            self._process_question()
            self._display_question_history()
            
        except Exception as e:
            logger.error(f"Error in questions section: {e}", exc_info=True)
            st.error("An error occurred while processing your question. Please try again.")
    
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
                            st.session_state.current_question = question  # Update text area
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
                    else:
                        st.error("Unable to generate an answer. Please try rephrasing your question.")
                        
            except Exception as e:
                logger.error(f"Error processing question: {e}", exc_info=True)
                st.error("Failed to process your question. Please try again.")
    
    def _handle_successful_response(self, question: str, response: str, processing_time: float) -> None:
        """Handle successful question response and related features
        
        Args:
            question: Original question text
            response: Generated answer
            processing_time: Time taken to process the question
        """
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
            
            # Audio response option
            if st.button("🔊 Listen to Answer", key="audio_response"):
                try:
                    with st.spinner("Generating audio..."):
                        audio_html = text_to_speech(response)
                        st.markdown(audio_html, unsafe_allow_html=True)
                except Exception as e:
                    logger.error(f"Error generating audio response: {e}", exc_info=True)
                    st.error("Unable to generate audio. Please try again.")
            
            # Generate and display follow-up questions
            self._display_follow_up_questions(question, response)
            
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
    
    def _display_follow_up_questions(self, original_question: str, response: str) -> None:
        """Generate and display follow-up questions
        
        Args:
            original_question: The user's original question
            response: The generated response
        """
        try:
            follow_ups = self.llm.get_follow_up_questions(original_question, response)
            
            if follow_ups:
                st.markdown("### Related Questions You Might Want to Ask:")
                
                for i, question in enumerate(follow_ups[:self.MAX_FOLLOW_UP_QUESTIONS], 1):
                    if st.button(f"🔄 {question}", key=f"follow_up_{i}"):
                        st.session_state.current_question = question
                        st.rerun()
                        
        except Exception as e:
            logger.error(f"Error generating follow-up questions: {e}", exc_info=True)
            # Silently fail for follow-up questions as they're not critical
    
    def _display_question_history(self) -> None:
        """Display question history with filtering options"""
        if st.session_state.question_history:
            st.markdown("### Previous Questions")
            
            # Reverse history to show most recent first
            for entry in reversed(st.session_state.question_history[-5:]):  # Show last 5 questions
                with st.expander(f"Q: {entry['question'][:100]}... ({entry['timestamp']})"):
                    st.markdown(f"**Question:** {entry['question']}")
                    st.markdown(f"**Answer:** {entry['response']}")
                    if st.button("🔄 Ask Again", key=f"replay_{entry['timestamp']}"):
                        st.session_state.current_question = entry['question']
                        st.rerun()