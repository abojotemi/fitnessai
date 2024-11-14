import requests
from PIL import Image
import streamlit as st
from dotenv import load_dotenv
import os
import tempfile
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
API_KEY = os.getenv("HF_API_KEY")
if not API_KEY:
    raise ValueError("HF_API_KEY not found in environment variables")

# Hugging Face API details
API_URL = "https://api-inference.huggingface.co/models/Salesforce/blip-image-captioning-base"
headers = {"Authorization": f"Bearer {API_KEY}"}

def query_image(image_path):
    """
    Query the Hugging Face API with an image file.
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        dict: API response
        
    Raises:
        requests.exceptions.RequestException: If API call fails
    """
    try:
        with open(image_path, "rb") as file:
            data = file.read()
            response = requests.post(
                API_URL, 
                headers=headers, 
                data=data,  # Send raw bytes instead of using files parameter
                timeout=10
            )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"API request failed: {str(e)}")
        logger.error(f"Response content: {response.content if 'response' in locals() else 'No response'}")
        raise

def save_uploadedfile(uploadedfile):
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

def predict_image(uploaded_file):
    if uploaded_file is not None:
        # Check file size (10MB limit)
        if uploaded_file.size > 10 * 1024 * 1024:
            st.error("File size too large. Please upload an image smaller than 10MB.")
            return

        try:
            # Save image to temporary file
            with st.spinner("Processing image..."):
                temp_path = save_uploadedfile(uploaded_file)
                
                try:
                    # Query the API
                    result = query_image(str(temp_path))
                    
                    # Display the caption
                    if isinstance(result, list):
                        caption = result[0].get("generated_text", "No caption generated.")
                    else:
                        caption = result.get("generated_text", "No caption generated.")
                    st.success("Equipment analyzed successfully!")
                    st.success(f"**Caption**: {caption}")
                    return caption
                    
                except requests.exceptions.RequestException as e:
                    st.error(f"Error calling the API: {str(e)}")
                    logger.error(f"Full error details: {str(e)}")
                finally:
                    # Clean up temporary file
                    temp_path.unlink()
                    
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")