# utils.py
import io
from PIL import Image
import streamlit as st
from config import config

def validate_image(image_file: str | bytes | io.BytesIO) -> bool:
    """Validate uploaded image file."""
    try:
        with Image.open(image_file) as img:
            # Check file size
            img.verify()
            if image_file.size > config.MAX_IMAGE_SIZE:
                raise ValueError("Image file too large")
            
            # Check file type
            fmt = img.format.lower()
            if fmt not in config.VALID_IMAGE_TYPES:
                raise ValueError(f"Unsupported image type: {fmt}")
            
            return True
    except Exception as e:
        st.error(f"Image validation failed: {str(e)}")
        return False

def cache_key_builder(*args, **kwargs) -> str:
    """Build cache key from args and kwargs."""
    return f"{'-'.join(str(arg) for arg in args)}-{'-'.join(f'{k}:{v}' for k, v in kwargs.items())}"

