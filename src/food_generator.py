# food_generator.py
import time
from dataclasses import dataclass
import streamlit as st
import logging
from image_processing import generate_food_image

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class GeneratedImage:
    prompt: str
    image: bytes
    timestamp: str

class FoodImageGenerator:
    """Food image generation component using text-to-image model"""
    
    def __init__(self):
        self._initialize_session_state()
    
    def _initialize_session_state(self):
        """Initialize session state variables for image generation"""
        if "generation_history" not in st.session_state:
            st.session_state.generation_history = []
        if "is_generating" not in st.session_state:
            st.session_state.is_generating = False
            
    def display(self):
        """Display the food image generation interface"""
        st.header("🎨 Food Image Generator")
        
        # Main generation interface
        col1, col2 = st.columns([2, 1])
        
        with col1:
            self._display_generation_interface()
            
        with col2:
            self._display_quick_prompts()
            
        # History section
        st.divider()
        self._display_generation_history()
    
    def _display_generation_interface(self):
        """Display the main generation interface"""
        st.markdown("""
        ### Generate Custom Food Images
        Describe the food you'd like to visualize in detail. Consider including:
        - Main ingredients
        - Cooking style
        - Plating/presentation
        - Lighting and photography style
        """)
        
        # Text input for generation prompt
        prompt = st.text_area(
            "Describe your ideal food image",
            height=100,
            placeholder="E.g., A rustic artisanal pizza with buffalo mozzarella, fresh basil, and cherry tomatoes, wood-fired with a perfectly charred crust, professional food photography with natural lighting",
            help="More detailed descriptions tend to give better results"
        )
        
        # Style selections
        col1, col2 = st.columns(2)
        with col1:
            style = st.selectbox(
                "Photography Style",
                ["Professional Food Photography", "Casual/Instagram Style", 
                 "Overhead Shot", "Close-up/Macro", "Minimalist"]
            )
        with col2:
            lighting = st.selectbox(
                "Lighting",
                ["Natural Daylight", "Studio Lighting", "Warm/Ambient", 
                 "Dramatic/Moody", "Bright and Airy"]
            )
        
        # Generation button
        if st.button("🎨 Generate Image", 
                    disabled=st.session_state.is_generating,
                    use_container_width=True):
            self._handle_generation(prompt, style, lighting)
    
    def _display_quick_prompts(self):
        """Display quick prompt suggestions"""
        st.markdown("### Quick Prompts")
        
        quick_prompts = [
            "Colorful Buddha Bowl",
            "Gourmet Burger",
            "Fresh Fruit Platter",
            "Elegant Sushi Plate",
            "Decadent Chocolate Cake"
        ]
        
        for prompt in quick_prompts:
            if st.button(f"✨ {prompt}", use_container_width=True):
                self._handle_generation(
                    prompt,
                    "Professional Food Photography",
                    "Natural Daylight"
                )
    
    def _handle_generation(self, prompt: str, style: str, lighting: str):
        """Handle the image generation process"""
        if not prompt:
            st.warning("Please enter a description first!")
            return
        
        st.session_state.is_generating = True
        
        try:
            # Combine prompt with style and lighting
            full_prompt = f"{prompt}, {style}, {lighting}"
            
            # Generate image
            image_data = generate_food_image(full_prompt)
            
            if image_data:
                # Store in history
                st.session_state.generation_history.append(
                    GeneratedImage(
                        prompt=prompt,
                        image=image_data,
                        timestamp=time.strftime('%Y-%m-%d %H:%M:%S')
                    )
                )
                
                # Display the generated image
                st.image(image_data, caption=prompt)
                
                # Offer download button
                st.download_button(
                    label="📥 Download Image",
                    data=image_data,
                    file_name=f"generated_food_{int(time.time())}.png",
                    mime="image/png"
                )
        except Exception as e:
            logger.error(f"Error in image generation: {str(e)}")
            st.error("Failed to generate image. Please try again.")
        finally:
            st.session_state.is_generating = False
    
    def _display_generation_history(self):
        """Display the history of generated images"""
        st.header("📸 Generation History")
        
        if not st.session_state.generation_history:
            st.info("No images generated yet. Try creating some!")
            return
        
        # Display images in a grid
        cols = st.columns(3)
        for idx, entry in enumerate(reversed(st.session_state.generation_history)):
            with cols[idx % 3]:
                st.image(entry.image, caption=entry.prompt)
                st.caption(f"Generated: {entry.timestamp}")
                
                # Download button for each image
                st.download_button(
                    label="📥 Download",
                    data=entry.image,
                    file_name=f"generated_food_{idx}.png",
                    mime="image/png",
                    key=f"download_{idx}"
                )
            
            # Add a divider after every 3 images
            if (idx + 1) % 3 == 0:
                st.divider()