import time
from typing import Optional, Dict, Any
from dataclasses import dataclass
from chat import predict_food
from llm import LLMHandler
import streamlit as st
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Message:
    role: str
    content: str
    image: Optional[Any] = None

class DietAnalyzer:
    """Diet analysis component for FitnessCoachApp with improved error handling and UX"""
    
    def __init__(self):
        self.llm = LLMHandler()
        self._initialize_session_state()
        
    def _initialize_session_state(self):
        """Initialize or reset session state variables"""
        if "diet_messages" not in st.session_state:
            st.session_state.diet_messages = [Message(
                role="assistant",
                content="Hello! I'm your Diet Analyzer. Upload food images and I'll provide nutritional insights. 📸"
            )]
        if "diet_processing" not in st.session_state:
            st.session_state.diet_processing = False
        if "analysis_history" not in st.session_state:
            st.session_state.analysis_history = []

    def display_chat_messages(self):
        """Display chat messages with improved styling and error handling"""
        try:
            for message in st.session_state.diet_messages:
                with st.chat_message(message.role):
                    if hasattr(message, 'image') and message.image:
                        try:
                            st.image(message.image, width=300, caption="Uploaded Food Image")
                        except Exception as e:
                            logger.error(f"Error displaying image: {str(e)}")
                            st.error("Unable to display image")
                    
                    if message.content:
                        st.markdown(message.content)
        except Exception as e:
            logger.error(f"Error in display_chat_messages: {str(e)}")
            st.error("Error displaying chat history")

    def typing_animation(self, response_text: str, speed: float = 0.04) -> str:
        """Create typing animation with configurable speed and error handling"""
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
            logger.error(f"Error in typing animation: {str(e)}")
            return response_text

    def process_image(self, image: Any, user_info: Dict[str, Any]) -> Optional[str]:
        """Process uploaded food image with comprehensive error handling"""
        try:
            with st.spinner("🔍 Analyzing your food..."):
                prediction = predict_food(image)
                
                if not prediction:
                    st.warning("Unable to identify the food in the image. Please try another photo.")
                    return None
                
                # Add confidence score display
                confidence = prediction.get('score', 0) * 100
                st.info(f"Identified as {prediction['label']} (Confidence: {confidence:.1f}%)")
                
                if confidence < 50:
                    st.warning("Low confidence in food identification. Results may not be accurate.")
                
                analysis = self.llm.analyze_diet(prediction, user_info)
                
                # Store analysis in history
                st.session_state.analysis_history.append({
                    'food': prediction['label'],
                    'confidence': confidence,
                    'analysis': analysis,
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
                })
                
                return analysis
                
        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            st.error("Unable to process the image. Please try again.")
            return None

    def display_diet_analyzer(self, user_info: Dict[str, Any]):
        """Enhanced main render method with improved UI/UX"""
        st.header("🥗 Diet Analyzer")
        
        # Create tabs for different views
        tab1, tab2 = st.tabs(["Analysis", "History"])
        
        with tab1:
            self.display_chat_messages()
            
            # File uploader with clear instructions
            st.markdown("### Upload Food Image")
            st.markdown("📸 Supported formats: JPG, JPEG, PNG, WebP")
            st.session_state.uploaded_image = st.file_uploader(
                "Choose an image of your food",
                type=['jpg', 'jpeg', 'png', 'webp'],
                key="diet_uploader",
                help="For best results, ensure the food is clearly visible and well-lit"
            )

            # Improved button layout
            col1, col2, col3 = st.columns([1, 1, 2])
            
            # Analyze button with loading state
            analyze_button = col1.button(
                "🔍 Analyze",
                key="analyze_diet",
                disabled=st.session_state.diet_processing
            )
            
            # Clear chat button
            clear_button = col2.button("🗑️ Clear Chat", key="clear_diet")
            
            if analyze_button:
                self._handle_analysis(user_info)
            
            if clear_button:
                self._handle_clear_chat()
        
        with tab2:
            self._display_analysis_history()

    def _handle_analysis(self, user_info: Dict[str, Any]):
        """Handle the analysis process with proper error handling"""
        if st.session_state.uploaded_image is None:
            st.warning("⚠️ Please upload an image first.")
            return
            
        if not st.session_state.diet_processing:
            st.session_state.diet_processing = True
            
            # Add user message with image
            st.session_state.diet_messages.append(Message(
                role='user',
                content='Uploaded an image for analysis',
                image=st.session_state.uploaded_image
            ))
            
            # Process image and get analysis
            analysis = self.process_image(st.session_state.uploaded_image, user_info)
            
            if analysis:
                # Add assistant response with typing effect
                with st.chat_message("assistant"):
                    final_response = self.typing_animation(analysis)
                    st.session_state.diet_messages.append(Message(
                        role='assistant',
                        content=final_response
                    ))
            
            st.session_state.diet_processing = False
            st.rerun()

    def _handle_clear_chat(self):
        """Handle clearing the chat history"""
        st.session_state.diet_messages = [Message(
            role="assistant",
            content="Hello! I'm your Diet Analyzer. Upload food images and I'll provide nutritional insights. 📸"
        )]
        st.session_state.diet_processing = False
        st.rerun()

    def _display_analysis_history(self):
        """Display historical analysis with filtering and sorting options"""
        if not st.session_state.analysis_history:
            st.info("No analysis history yet. Start by analyzing some food images!")
            return
            
        st.subheader("📊 Analysis History")
        
        # Add sorting options
        sort_by = st.selectbox(
            "Sort by:",
            ["Most Recent", "Highest Confidence", "Lowest Confidence"]
        )
        
        history = sorted(
            st.session_state.analysis_history,
            key=lambda x: {
                "Most Recent": lambda y: y['timestamp'],
                "Highest Confidence": lambda y: -y['confidence'],
                "Lowest Confidence": lambda y: y['confidence']
            }[sort_by](x)
        )
        
        for entry in history:
            with st.expander(f"{entry['food']} - {entry['timestamp']}"):
                st.markdown(f"**Confidence:** {entry['confidence']:.1f}%")
                st.markdown(f"**Analysis:** {entry['analysis']}")