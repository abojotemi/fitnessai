import time
from chat import predict_food
from llm import LLMHandler
import streamlit as st
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DietAnalyzer:
    """Diet analysis component for FitnessCoachApp"""
    def __init__(self):
        # Initialize session states for diet analyzer
        self.llm = LLMHandler()
        if "diet_messages" not in st.session_state:
            st.session_state.diet_messages = [{
                "role": "assistant",
                "content": "Hello, I am a helpful Diet analyzer. Please paste images of your food and I will analyze it for you."
            }]
        if "diet_processing" not in st.session_state:
            st.session_state.diet_processing = False

    def display_chat_messages(self):
        """Display chat messages with images and text"""
        for message in st.session_state.diet_messages:
            with st.chat_message(message['role']):
                if 'image' in message:
                    st.image(message['image'], width=200)
                if message.get('content'):
                    st.markdown(message['content'])

    def typing_animation(self, response_text):
        """Create typing animation effect for responses"""
        placeholder = st.empty()
        displayed_message = ""
        for word in response_text.split():
            displayed_message += word + " "
            time.sleep(0.04)
            placeholder.markdown(displayed_message + "▌")
        placeholder.markdown(response_text)
        return response_text

    def process_image(self, image, user_info):
        """Process uploaded food image and get analysis"""
        try:
            prediction = predict_food(image)
            if prediction:
                analysis = self.llm.analyze_diet(prediction, user_info)
                return analysis
            return None
        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            st.error(f"Error processing image: {str(e)}")
            return None

    def display_diet_analyzer(self, user_info):
        """Main render method for diet analyzer interface"""
        st.header("Diet Analyzer")
        
        # Display existing chat
        self.display_chat_messages()

        # Handle image upload
        uploaded_image = st.file_uploader(
            "Upload your image here",
            type=['jpg', 'jpeg', 'png', 'webp'],
            key="diet_uploader"
        )

        # Create columns for buttons
        col1, col2 = st.columns([1, 4])

        # Analyze button
        if col1.button("Analyze", key="analyze_diet"):
            if uploaded_image is not None and not st.session_state.diet_processing:
                st.session_state.diet_processing = True
                
                # Add user message with image
                st.session_state.diet_messages.append({
                    'role': 'user',
                    'content': 'Uploaded an image for analysis',
                    'image': uploaded_image
                })
                
                # Process image and get analysis
                analysis = self.process_image(uploaded_image, user_info)
                    
                if analysis:
                    # Add assistant response with typing effect
                    with st.chat_message("assistant"):
                        final_response = self.typing_animation(analysis)
                        st.session_state.diet_messages.append({
                            'role': 'assistant',
                            'content': final_response
                        })
                else:
                    st.error("Could not analyze the image. Please try another image.")
                    logger.error("Could not analyze the image. Please try another image.")
                    
                st.session_state.diet_processing = False
                st.rerun()
            elif uploaded_image is None:
                st.warning("Please upload an image first.")

        # Clear chat button
        if col2.button("Clear Chat", key="clear_diet"):
            st.session_state.diet_messages = [{
                "role": "assistant",
                "content": "Hello, I am a helpful Diet analyzer. Please paste images of your food and I will analyze it for you."
            }]
            st.session_state.diet_processing = False
            st.rerun()