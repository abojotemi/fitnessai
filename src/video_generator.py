import base64
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import streamlit as st

import requests
from dotenv import load_dotenv

# Configure logging with more detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('video_generator.log')
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class DIDConfig:
    base_url: str = 'https://api.d-id.com'
    poll_interval: int = 5
    max_retries: int = 3  # Added max retries for API calls

class VideoGenerator:
    def __init__(self):
        logger.info("Initializing VideoGenerator")
        load_dotenv()
        self.api_key = self._get_api_key()
        self.headers = self._create_headers()
        self.config = DIDConfig()
        logger.debug("VideoGenerator initialized successfully")

    def _get_api_key(self) -> str:
        logger.debug("Retrieving API key from environment")
        api_key = os.getenv("DID_API_KEY")
        if not api_key:
            logger.error("DID_API_KEY not found in environment variables")
            raise ValueError("DID_API_KEY is not set in the environment.")
        return api_key

    def _create_headers(self) -> dict:
        logger.debug("Creating API headers")
        auth_base64 = base64.b64encode(f'{self.api_key}:'.encode('ascii')).decode('ascii')
        return {
            'Authorization': f'Basic {auth_base64}',
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        }

    def _create_payload(self, input_text: str) -> dict:
        logger.debug("Creating API payload")
        return {
            'script': {
                'type': 'text',
                'input': input_text,
                'provider': {
                    'type': 'microsoft',
                    'voice_id': 'Sara',
                }
            },
        }

    def _create_clip(self, payload: dict) -> Optional[str]:
        url = f'{self.config.base_url}/clips'
        logger.info(f"Creating clip with payload length: {len(str(payload))}")
        
        for attempt in range(self.config.max_retries):
            try:
                response = requests.post(url, json=payload, headers=self.headers)
                if response.status_code == 201:
                    clip_id = response.json().get('id')
                    if clip_id:
                        logger.info(f"Clip created successfully with ID: {clip_id}")
                        return clip_id
                    logger.error("Failed to get clip ID from response")
                    return None
                logger.warning(f"Attempt {attempt + 1} failed with status {response.status_code}")
                time.sleep(2 ** attempt)  # Exponential backoff
            except requests.exceptions.RequestException as e:
                logger.error(f"Request error on attempt {attempt + 1}: {e}")
                if attempt == self.config.max_retries - 1:
                    return None
                time.sleep(2 ** attempt)
        
        return None

    def _wait_for_clip(self, clip_id: str) -> Optional[str]:
        url = f'{self.config.base_url}/clips/{clip_id}'
        logger.info(f"Waiting for clip {clip_id} to complete")
        
        start_time = time.time()
        while True:
            if time.time() - start_time > 300:  # 5-minute timeout
                logger.error("Clip generation timed out after 5 minutes")
                return None
                
            time.sleep(self.config.poll_interval)
            try:
                response = requests.get(url, headers=self.headers)
                if response.status_code == 200:
                    status_data = response.json()
                    status = status_data.get('status')
                    
                    if status == 'done':
                        result_url = status_data.get('result_url')
                        logger.info(f"Clip completed successfully: {result_url}")
                        return result_url
                    elif status == 'error':
                        logger.error(f"Clip generation failed: {status_data.get('error')}")
                        return None
                    
                    logger.debug(f"Clip status: {status}")
                else:
                    logger.error(f"Error checking clip status: {response.status_code}")
                    return None
            except requests.exceptions.RequestException as e:
                logger.error(f"Request error while checking clip status: {e}")
                return None

    def _save_video(self, result_url: str, clip_id: str) -> Optional[Path]:
        logger.info(f"Downloading video from {result_url}")
        try:
            response = requests.get(result_url)
            if response.status_code != 200:
                logger.error(f"Failed to download video: {response.status_code}")
                return None

            videos_dir = Path(__file__).parent / 'static' / 'videos'
            videos_dir.mkdir(parents=True, exist_ok=True)
            
            video_path = videos_dir / f'{clip_id}.mp4'
            video_path.write_bytes(response.content)
            
            logger.info(f"Video saved successfully at {video_path}")
            return video_path
            
        except Exception as e:
            logger.error(f"Error saving video: {e}", exc_info=True)
            return None

    @st.cache_data
    def generate_video(_self, input_text: str) -> Optional[Path]:
        """Generate a video using the D-ID API."""
        logger.info("Starting video generation process")
        start_time = time.time()
        try:
            payload = _self._create_payload(input_text)
            
            clip_id = _self._create_clip(payload)
            if not clip_id:
                logger.error("Failed to create clip")
                return None
                
            result_url = _self._wait_for_clip(clip_id)
            if not result_url:
                logger.error("Failed to get result URL")
                return None
                
            return _self._save_video(result_url, clip_id)
            
        except Exception as e:
            logger.error(f"Error in video generation process: {e}", exc_info=True)
            return None