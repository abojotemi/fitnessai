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
API_KEY = os.getenv("HF_API_KEY")
if not API_KEY:
    raise ValueError("HF_API_KEY not found in environment variables")

# API endpoints
FOOD_API_URL = "https://api-inference.huggingface.co/models/nateraw/food"
headers = {"Authorization": f"Bearer {API_KEY}"}


starry_api_key = os.getenv("STARRYAI_API_KEY")
if not starry_api_key:
    raise ValueError("STARRYAI_API_KEY not found in environment variables")
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

    response = requests.post(url, json=payload, headers=headers)
    response.raise_for_status()
    return json.loads(response.text)['id']

def get_image(creation_id: int, max_attempts: int = 20, initial_delay: float = 5.0) -> Optional[str]:
    """
    Poll for the generated image URL with exponential backoff.
    
    Args:
        creation_id: The ID of the creation to check
        max_attempts: Maximum number of polling attempts
        initial_delay: Initial delay between polling attempts in seconds
        
    Returns:
        Optional[str]: The image URL if successful, None if not ready after max attempts
    """
    url = f"https://api.starryai.com/creations/{creation_id}"
    headers = {
        "accept": "application/json",
        "X-API-Key": starry_api_key
    }
    
    delay = initial_delay
    
    for attempt in range(max_attempts):
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            data = response.json()
            
            # Check if the image is ready
            if data['status'] == 'completed' and data['images'] and data['images'][0]['url']:
                return data['images'][0]['url']
            
            # If still in progress, wait and try again
            if data['status'] == 'in progress':
                if attempt < max_attempts - 1:  # Don't sleep on the last attempt
                    time.sleep(delay)
                    delay = min(delay * 1.5, 30.0)  # Exponential backoff, capped at 30 seconds
                continue
                
            # Handle failed or unexpected status
            if data['status'] in ['failed', 'expired']:
                logger.error(f"Image generation failed or expired. Status: {data['status']}")
                return None
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Error checking image status (attempt {attempt + 1}/{max_attempts}): {str(e)}")
            if attempt < max_attempts - 1:
                time.sleep(delay)
                delay = min(delay * 1.5, 30.0)
            continue
            
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
        # Create the image request
        creation_id = create_image(prompt)
        
        # Poll for the result
        if st.spinner is not None:  # Check if we're in a Streamlit context
            with st.spinner(f"🎨 Generating image... This may take up to a minute"):
                return get_image(creation_id)
        else:
            return get_image(creation_id)
            
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to generate image: {str(e)}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error during image generation: {str(e)}")
        return None

def query_image_with_retry(image_path: str, max_retries: int = 5, initial_retry_delay: float = 20.0) -> dict:
    """
    Query the Hugging Face API with retry logic for model loading.
    
    Args:
        image_path (str): Path to the image file
        max_retries (int): Maximum number of retry attempts
        initial_retry_delay (float): Initial delay in seconds between retries
        
    Returns:
        dict: API response
        
    Raises:
        requests.exceptions.RequestException: If all retry attempts fail
    """
    retry_delay = initial_retry_delay

    for attempt in range(max_retries):
        try:
            with open(image_path, "rb") as file:
                data = file.read()
                response = requests.post(
                    FOOD_API_URL,
                    headers=headers,
                    data=data,
                    timeout=30
                )
                
            if response.status_code == 503 and "is currently loading" in response.text:
                estimated_time = response.json().get("estimated_time", retry_delay)
                logger.info(f"Model is loading. Waiting {estimated_time:.1f} seconds...")
                
                progress_text = "Model is loading... Please wait."
                progress_bar = st.progress(0, text=progress_text)
                
                chunks = 20
                chunk_time = estimated_time / chunks
                for i in range(chunks):
                    time.sleep(chunk_time)
                    progress_bar.progress((i + 1) / chunks, 
                                       text=f"Model is loading... {((i + 1) / chunks * 100):.0f}%")
                
                progress_bar.empty()
                continue
                
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed (attempt {attempt + 1}/{max_retries}): {str(e)}")
            logger.error(f"Response content: {response.content if 'response' in locals() else 'No response'}")
            
            if attempt == max_retries - 1:
                raise
            
            time.sleep(retry_delay)
            retry_delay *= 1.5

def save_uploadedfile(uploadedfile) -> Optional[Path]:
    """
    Safely save uploaded file to a temporary directory.
    
    Args:
        uploadedfile: Streamlit UploadedFile object
        
    Returns:
        Path: Path to saved temporary file
    """
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploadedfile.name).suffix) as tmp_file:
            tmp_file.write(uploadedfile.getvalue())
            return Path(tmp_file.name)
    except Exception as e:
        logger.error(f"Failed to save uploaded file: {str(e)}")
        raise

def predict_food(uploaded_file) -> Optional[dict]:
    """
    Process uploaded file and get prediction with improved error handling and user feedback.
    """
    if uploaded_file is not None:
        # Check file size (10MB limit)
        if uploaded_file.size > 10 * 1024 * 1024:
            st.error("File size too large. Please upload an image smaller than 10MB.")
            return None

        try:
            # Save image to temporary file
            with st.spinner("Processing image..."):
                temp_path = save_uploadedfile(uploaded_file)
                
                try:
                    # Query the API with retry logic
                    result = query_image_with_retry(str(temp_path))
                    
                    # Parse and return result
                    caption = parse_response(result)
                    if caption:
                        st.success("Image analyzed successfully!")
                        return caption
                    return None
                    
                except requests.exceptions.RequestException as e:
                    st.error("️Error analyzing image. The service might be temporarily unavailable. Please try again in a few moments.")
                    logger.error(f"Full error details: {str(e)}")
                    return None
                finally:
                    # Clean up temporary file
                    if temp_path.exists():
                        temp_path.unlink()
                    
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")
            logger.error(f"Error processing image: {str(e)}")
            return None

def generate_food_image(prompt: str) -> Optional[bytes]:
    """
    Generate a food image from a text prompt with error handling and user feedback.
    
    Args:
        prompt (str): Text description of the food to generate
        
    Returns:
        Optional[bytes]: Generated image data or None if generation fails
    """
    try:
        image_data = generate_image_from_text(prompt)
        
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

def parse_response(result: Any) -> Optional[dict]:
    """
    Parse the API response with improved error handling.
    
    Args:
        result: The API response from the model
        
    Returns:
        Optional[dict]: The parsed response or None if parsing fails
    """
    try:
        # Check if result is a list of dictionaries
        if isinstance(result, list) and len(result) > 0:
            return result[0]
        # Check if result is a dictionary
        elif isinstance(result, dict):
            return result
        else:
            logger.error(f"Unexpected response format: {result}")
            return None
    except Exception as e:
        logger.error(f"Failed to parse response: {str(e)}")
        return None