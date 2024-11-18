import base64
from io import BytesIO
import requests
from PIL import Image
import streamlit as st
from dotenv import load_dotenv
import os
import tempfile
import logging
import time
from pathlib import Path
from typing import Optional, Any
import time
import requests
import streamlit as st
import json
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

starry_api_key = os.getenv("STARRYAI_API_KEY")
if not starry_api_key:
    raise ValueError("STARRYAI_API_KEY not found in environment variables")
class ImageGenerationError(Exception):
    """Custom exception for image generation errors"""
    pass

def create_image(prompt: str) -> int:
    """Create a new image generation request."""
    url = "https://api.starryai.com/creations/"
    
    payload = {
        "model": "lyra",
        "aspectRatio": "square",
        "highResolution": False,
        "images": 1,
        "steps": 20,
        "initialImageMode": "color",
        "prompt": prompt
    }
    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "X-API-Key": starry_api_key,
    }

    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        return json.loads(response.text)['id']
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to create image request: {str(e)}")
        raise ImageGenerationError(f"Failed to initiate image generation: {str(e)}")

def check_image_status(creation_id: int) -> dict[str, Any]:
    """Check the status of an image generation request."""
    url = f"https://api.starryai.com/creations/{creation_id}"
    headers = {
        "accept": "application/json",
        "X-API-Key": starry_api_key
    }
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to check image status: {str(e)}")
        raise ImageGenerationError(f"Failed to check image status: {str(e)}")

def get_image(creation_id: int, max_attempts: int = 40, initial_delay: float = 3.0) -> Optional[str]:
    """
    Poll for the generated image URL with adaptive polling strategy.
    
    Args:
        creation_id: The ID of the creation to check
        max_attempts: Maximum number of polling attempts
        initial_delay: Initial delay between polling attempts in seconds
        
    Returns:
        Optional[str]: The image URL if successful, None if not ready after max attempts
    """
    delay = initial_delay
    progress_counter = 0
    last_status = None
    consecutive_errors = 0
    
    for attempt in range(max_attempts):
        try:
            if st.spinner is not None:
                progress_counter = min(95, progress_counter + 5)  # Cap at 95% to avoid false completion
                st.spinner(f"🎨 Generating image... {progress_counter}% ({attempt + 1}/{max_attempts})")
            
            data = check_image_status(creation_id)
            consecutive_errors = 0  # Reset error counter on successful request
            
            current_status = data.get('status')
            if current_status != last_status:
                logger.info(f"Image generation status changed to: {current_status}")
                last_status = current_status
            
            # Check if the image is ready
            if current_status == 'completed' and data['images'] and data['images'][0].get('url'):
                logger.info("Image generation completed successfully")
                return data['images'][0]['url']
            
            # Handle various status cases
            if current_status == 'failed':
                logger.error("Image generation failed")
                return None
            elif current_status == 'expired':
                logger.error("Image generation request expired")
                return None
            elif current_status != 'in progress':
                logger.warning(f"Unexpected status: {current_status}")
            
            # Adaptive delay based on attempt number
            if attempt < max_attempts - 1:
                if attempt < 5:
                    delay = initial_delay  # Quick checks initially
                elif attempt < 15:
                    delay = min(delay * 1.2, 10.0)  # Gradual increase
                else:
                    delay = min(delay * 1.1, 15.0)  # Slower increase for later attempts
                
                logger.info(f"Waiting {delay:.1f} seconds before next attempt")
                time.sleep(delay)
                
        except ImageGenerationError as e:
            consecutive_errors += 1
            logger.warning(f"Error checking status (attempt {attempt + 1}/{max_attempts}): {str(e)}")
            
            if consecutive_errors >= 3:
                logger.error("Too many consecutive errors, aborting")
                return None
                
            if attempt < max_attempts - 1:
                time.sleep(min(delay * 1.5, 20.0))
    
    logger.warning(f"Image not ready after {max_attempts} attempts")
    return None

def generate_image_from_text(prompt: str) -> Optional[str]:
    """
    Generate an image from text with improved error handling and status checking.
    
    Args:
        prompt (str): Text prompt to generate image from
        
    Returns:
        Optional[str]: Generated image URL or None if generation fails
    """
    try:
        if st.spinner is not None:
            with st.spinner("🎨 Initializing image generation..."):
                creation_id = create_image(prompt)
        else:
            creation_id = create_image(prompt)
        
        return get_image(creation_id)
            
    except ImageGenerationError as e:
        logger.error(f"Image generation error: {str(e)}")
        if st.error is not None:
            st.error("Failed to generate image. Please try again.")
        return None
    except Exception as e:
        logger.error(f"Unexpected error during image generation: {str(e)}")
        if st.error is not None:
            st.error("An unexpected error occurred. Please try again.")
        return None

# Optional: Add a retry wrapper for the entire generation process
def generate_image_with_retry(prompt: str, max_retries: int = 2) -> Optional[str]:
    """Attempt to generate an image with retries."""
    for attempt in range(max_retries):
        try:
            result = generate_image_from_text(prompt)
            if result:
                return result
            if attempt < max_retries - 1:
                logger.info(f"Retrying image generation (attempt {attempt + 2}/{max_retries})")
                time.sleep(5)  # Wait before retrying
        except Exception as e:
            logger.error(f"Error during generation attempt {attempt + 1}: {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(5)
    return None

def process_image(img):
    # Open and convert image to RGB (Gemini prefers RGB format)
    img = Image.open(img)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Convert image to bytes
    buffered = BytesIO()
    img.save(buffered, format="JPEG")  # Convert to JPEG format
    img_bytes = buffered.getvalue()
    
    # Encode image bytes to base64
    img_base64 = base64.b64encode(img_bytes).decode("utf-8")
    return img_base64

def generate_food_image(prompt: str) -> Optional[bytes]:
    """
    Generate a food image from a text prompt with error handling and user feedback.
    
    Args:
        prompt (str): Text description of the food to generate
        
    Returns:
        Optional[bytes]: Generated image data or None if generation fails
    """
    try:
        with st.spinner("Generating Image..."):
            image_data = generate_image_with_retry(prompt)
            
            if image_data:
                st.success("Image generated successfully!")
                return image_data
            return None
            
    except requests.exceptions.RequestException as e:
        st.error("️Error generating image. The service might be temporarily unavailable. Please try again in a few moments.")
        logger.error(f"Full error details: {str(e)}")
        return None
    except Exception as e:
        st.error(f"Error generating image: {str(e)}")
        logger.error(f"Error generating image: {str(e)}")
        return None