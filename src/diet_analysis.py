import time
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from image_processing import predict_food
from llm import LLMHandler
import streamlit as st
import logging
from datetime import datetime

# Enhanced logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class Message:
    """Dataclass to store chat messages with optional image attachment"""
    role: str
    content: str
    image: Optional[Any] = None
    timestamp: datetime = datetime.now()

class DietAnalyzer:
    """Diet analysis component for FitnessCoachApp with improved error handling and UX"""
    
    SUPPORTED_IMAGE_FORMATS = ['jpg', 'jpeg', 'png', 'webp']
    CONFIDENCE_THRESHOLD = 50.0
    
    def __init__(self):
        """Initialize DietAnalyzer with LLM handler and session state"""
        logger.info("Initializing DietAnalyzer")
        self.llm = LLMHandler()
        self._initialize_session_state()
    
    def _initialize_session_state(self) -> None:
        """Initialize or reset session state variables with default values"""
        default_states = {
            "diet_messages": [Message(
                role="assistant",
                content="Hello! I'm your Diet Analyzer. Upload food images and I'll provide nutritional insights. 📸"
            )],
            "diet_processing": False,
            "analysis_history": [],
            "uploaded_image": None
        }
        
        for key, default_value in default_states.items():
            if key not in st.session_state:
                st.session_state[key] = default_value
                logger.debug(f"Initialized session state: {key}")

    def display_chat_messages(self) -> None:
        """Display chat messages with improved styling and error handling"""
        try:
            for message in st.session_state.diet_messages:
                with st.chat_message(message.role):
                    if message.image is not None:
                        try:
                            st.image(
                                message.image,
                                width=300,
                                caption="Uploaded Food Image",
                                use_container_width=False
                            )
                        except Exception as e:
                            logger.error(f"Error displaying image: {e}", exc_info=True)
                            st.error("Unable to display image. Please try uploading again.")
                    
                    if message.content:
                        st.markdown(message.content)
                        
        except Exception as e:
            logger.error(f"Error in display_chat_messages: {e}", exc_info=True)
            st.error("Error displaying chat history. Please refresh the page.")

    def typing_animation(self, response_text: str, speed: float = 0.03) -> str:
        """Create typing animation with configurable speed and error handling
        
        Args:
            response_text: Text to animate
            speed: Delay between words in seconds
            
        Returns:
            The complete response text
        """
        try:
            placeholder = st.empty()
            displayed_message = ""
            
            for word in response_text.split():
                displayed_message += word + " "
                time.sleep(speed)
                placeholder.markdown(displayed_message + "▌")
            
            placeholder.markdown(response_text)
            return response_text
            
        except Exception as e:
            logger.error(f"Error in typing animation: {e}", exc_info=True)
            st.markdown(response_text)  # Fallback to instant display
            return response_text

    def process_image(self, image: Any, user_info: Dict[str, Any]) -> Optional[str]:
        """Process uploaded food image with comprehensive error handling
        
        Args:
            image: Uploaded image file
            user_info: User profile information
            
        Returns:
            Analysis text if successful, None otherwise
        """
        start_time = time.time()
        logger.info("Starting image processing")
        
        try:
            with st.spinner("🔍 Analyzing your food..."):
                prediction = predict_food(image)
                
                if not prediction:
                    logger.warning("Food prediction failed")
                    st.warning("Unable to identify the food in the image. Please try another photo.")
                    return None
                
                confidence = prediction.get('score', 0) * 100
                food_label = prediction.get('label', 'Unknown')
                
                logger.info(f"Food identified: {food_label} with confidence {confidence:.1f}%")
                st.info(f"Identified as {food_label} (Confidence: {confidence:.1f}%)")
                
                if confidence < self.CONFIDENCE_THRESHOLD:
                    st.warning("Low confidence in food identification. Results may not be accurate.")
                
                analysis = self.llm.analyze_diet(prediction, user_info)
                
                self._update_analysis_history(food_label, confidence, analysis)
                
                processing_time = time.time() - start_time
                logger.info(f"Image processing completed in {processing_time:.2f} seconds")
                
                return analysis
                
        except Exception as e:
            logger.error(f"Error processing image: {e}", exc_info=True)
            st.error("Unable to process the image. Please try again.")
            return None

    def _update_analysis_history(self, food: str, confidence: float, analysis: str) -> None:
        """Update the analysis history with new entry"""
        history_entry = {
            'food': food,
            'confidence': confidence,
            'analysis': analysis,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        st.session_state.analysis_history.append(history_entry)
        logger.debug(f"Added new entry to analysis history: {food}")

    def display(self, user_info: Dict[str, Any]) -> None:
        """Enhanced main render method with improved UI/UX"""
        st.header("🥗 Diet Analyzer")
        
        tab1, tab2 = st.tabs(["Analysis", "History"])
        
        with tab1:
            self.display_chat_messages()
            self._render_upload_section()
            self._render_action_buttons(user_info)
        
        with tab2:
            self._display_analysis_history()

    def _render_upload_section(self) -> None:
        """Render the image upload section"""
        st.markdown("### Upload Food Image")
        st.markdown(f"📸 Supported formats: {', '.join(self.SUPPORTED_IMAGE_FORMATS).upper()}")
        
        st.session_state.uploaded_image = st.file_uploader(
            "Choose an image of your food",
            type=self.SUPPORTED_IMAGE_FORMATS,
            key="diet_uploader",
            help="For best results, ensure the food is clearly visible and well-lit"
        )

    def _render_action_buttons(self, user_info: Dict[str, Any]) -> None:
        """Render analyze and clear buttons"""
        col1, col2, col3 = st.columns([1, 1, 2])
        
        analyze_button = col1.button(
            "🔍 Analyze",
            key="analyze_diet",
            disabled=st.session_state.diet_processing,
            help="Click to analyze the uploaded image"
        )
        
        clear_button = col2.button(
            "🗑️ Clear Chat",
            key="clear_diet",
            help="Clear chat history"
        )
        
        if analyze_button:
            self._handle_analysis(user_info)
        
        if clear_button:
            self._handle_clear_chat()

    def _handle_analysis(self, user_info: Dict[str, Any]) -> None:
        """Handle the analysis process with proper error handling"""
        if st.session_state.uploaded_image is None:
            st.warning("⚠️ Please upload an image first.")
            return
        
        try:
            if not st.session_state.diet_processing:
                st.session_state.diet_processing = True
                logger.info("Starting new analysis")
                
                st.session_state.diet_messages.append(Message(
                    role='user',
                    content='Uploaded an image for analysis',
                    image=st.session_state.uploaded_image
                ))
                
                analysis = self.process_image(st.session_state.uploaded_image, user_info)
                
                if analysis:
                    with st.chat_message("assistant"):
                        final_response = self.typing_animation(analysis)
                        st.session_state.diet_messages.append(Message(
                            role='assistant',
                            content=final_response
                        ))
                
                st.session_state.diet_processing = False
                st.rerun()
                
        except Exception as e:
            logger.error(f"Error in analysis handling: {e}", exc_info=True)
            st.error("An error occurred during analysis. Please try again.")
            st.session_state.diet_processing = False

    # def _handle_clear_chat(self) -> None:
    #     """Handle clearing the chat history"""
    #     logger.info("Clearing chat history")
    #     self._initialize_session_state()
    #     st.rerun()
    def _handle_clear_chat(self) -> None:
        """Handle clearing the chat history and resetting all related states"""
        logger.info("Clearing chat history and resetting states")
        st.session_state.diet_messages = [Message(
            role="assistant",
            content="Hello! I'm your Diet Analyzer. Upload food images and I'll provide nutritional insights. 📸"
        )]
        st.session_state.diet_processing = False
        st.session_state.uploaded_image = None
        logger.debug("Chat history and states cleared")
        st.rerun()

    def _display_analysis_history(self) -> None:
        """Display historical analysis with filtering and sorting options"""
        if not st.session_state.analysis_history:
            st.info("No analysis history yet. Start by analyzing some food images!")
            return
        
        st.subheader("📊 Analysis History")
        
        sort_options = {
            "Most Recent": lambda x: x['timestamp'],
            "Highest Confidence": lambda x: -x['confidence'],
            "Lowest Confidence": lambda x: x['confidence']
        }
        
        sort_by = st.selectbox(
            "Sort by:",
            list(sort_options.keys())
        )
        
        history = sorted(
            st.session_state.analysis_history,
            key=sort_options[sort_by]
        )
        
        for entry in history:
            with st.expander(f"{entry['food']} - {entry['timestamp']}"):
                st.markdown(f"**Confidence:** {entry['confidence']:.1f}%")
                st.markdown(f"**Analysis:** {entry['analysis']}")