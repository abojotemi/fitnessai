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

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
from collections import Counter
import time
import logging

class UsageAnalytics:
    """Analytics component to track food image generation usage"""
    
    def __init__(self):
        # Initialize session state for tracking if not already exists
        self._initialize_analytics_state()
    
    def _initialize_analytics_state(self):
        """Initialize session state variables for analytics tracking"""
        if "usage_logs" not in st.session_state:
            st.session_state.usage_logs = []
    
    def log_generation(self, prompt: str, style: str, lighting: str):
        """Log details of each image generation attempt"""
        log_entry = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'prompt': prompt,
            'style': style,
            'lighting': lighting,
            'success': True  # Assuming success, can be modified based on actual generation result
        }
        st.session_state.usage_logs.append(log_entry)
    
    def display(self):
        """Display the usage analytics dashboard"""
        st.header("📊 Usage Analytics")
        
        # Ensure we have usage data
        if not hasattr(st.session_state, 'usage_logs') or not st.session_state.usage_logs:
            st.info("No usage data available yet. Generate some images first!")
            return
        
        # Convert logs to DataFrame
        df = pd.DataFrame(st.session_state.usage_logs)
        
        # Tabs for different analytics views
        tab1, tab2, tab3 = st.tabs(["Overview", "Generation Trends", "Detailed Insights"])
        
        with tab1:
            self._overview_tab(df)
        
        with tab2:
            self._generation_trends_tab(df)
        
        with tab3:
            self._detailed_insights_tab(df)
    
    def _overview_tab(self, df):
        """Display high-level overview of image generation"""
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Generations", len(df))
        
        with col2:
            st.metric("Unique Styles Used", df['style'].nunique())
        
        with col3:
            st.metric("Unique Lighting Conditions", df['lighting'].nunique())
        
        # Pie chart of styles
        fig_styles = px.pie(df, names='style', title='Image Generation Styles')
        st.plotly_chart(fig_styles, use_container_width=True)
    
    def _generation_trends_tab(self, df):
        """Display trends in image generation"""
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Daily generation count
        daily_counts = df.groupby(df['timestamp'].dt.date).size()
        
        fig_daily = go.Figure(data=[
            go.Bar(x=daily_counts.index.astype(str), y=daily_counts.values)
        ])
        fig_daily.update_layout(
            title='Daily Image Generation Count',
            xaxis_title='Date',
            yaxis_title='Number of Generations'
        )
        st.plotly_chart(fig_daily, use_container_width=True)
        
        # Most common styles and lighting conditions
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Most Used Styles")
            style_counts = df['style'].value_counts()
            st.bar_chart(style_counts)
        
        with col2:
            st.subheader("Most Used Lighting")
            lighting_counts = df['lighting'].value_counts()
            st.bar_chart(lighting_counts)
    
    def _detailed_insights_tab(self, df):
        """Provide detailed analytics and data exploration"""
        # Prompt word cloud (conceptual representation)
        st.subheader("Prompt Word Analysis")
        all_prompts = ' '.join(df['prompt'])
        word_counts = Counter(all_prompts.split())
        top_words = dict(sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:10])
        
        st.bar_chart(top_words)
        
        # Raw data display
        st.subheader("Raw Generation Logs")
        st.dataframe(df)


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
        if "analyzer" not in st.session_state:
            st.session_state.analyzer = UsageAnalytics()
        
    def display(self):
        """Display the food image generation interface"""
        st.header("🎨 Food Image Generator")
        tabs = st.tabs(['Generator', 'Analytics'])
        
        with tabs[0]:
        # Main generation interface
            col1, col2 = st.columns([2, 1])
            
            with col1:
                self._display_generation_interface()
                
            with col2:
                self._display_quick_prompts()
                
            # History section
            st.divider()
            self._display_generation_history()
        with tabs[1]:
            st.session_state.analyzer.display()
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
        st.session_state.analyzer.log_generation(prompt, style, lighting)
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