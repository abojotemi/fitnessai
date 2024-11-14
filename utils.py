import re
import base64
import io
from gtts import gTTS
import assemblyai as aai
import os
from dotenv import load_dotenv
import tempfile
from pathlib import Path
import logging
import hashlib
import streamlit as st
import time

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def remove_markdown_formatting(text):
    """Remove common Markdown symbols using regular expressions"""
    # Remove headings (e.g., # Heading)
    text = re.sub(r'#', '', text)
    # Remove emphasis (e.g., **bold** or *italic*)
    text = re.sub(r'\*{1,2}', '', text)
    # Remove underscores (e.g., _italic_)
    text = re.sub(r'_', '', text)
    # Remove links (e.g., [text](url))
    text = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', text)
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def speech_to_text(audio_file):
    aai.settings.api_key = os.getenv("ASSEMBLYAI_API_KEY")
    transcriber = aai.Transcriber()
    transcript = transcriber.transcribe(audio_file)
    return transcript.text

def text_to_speech(text):
    """Convert text to speech and return the audio player HTML"""
    text = remove_markdown_formatting(text)
    tts = gTTS(text=text, lang='en')
    
    audio_fp = io.BytesIO()
    tts.write_to_fp(audio_fp)
    audio_fp.seek(0)
    
    audio_str = base64.b64encode(audio_fp.read()).decode()
    
    audio_html = f'''
        <audio controls autoplay>
            <source src="data:audio/mp3;base64,{audio_str}" type="audio/mp3">
        </audio>
    '''
    
    return audio_html


class TTSHandler:
    """Handles text-to-speech conversion with caching and chunking"""
    
    def __init__(self):
        self.cache_dir = Path(tempfile.gettempdir()) / "tts_cache"
        self.cache_dir.mkdir(exist_ok=True)
        self.chunk_size = 500  # Maximum characters per chunk
        
    def _get_cache_path(self, text: str) -> Path:
        """Generate a cache file path based on text content"""
        text_hash = hashlib.md5(text.encode()).hexdigest()
        return self.cache_dir / f"tts_{text_hash}.mp3"

    def _clean_text(self, text: str) -> str:
        """Clean and prepare text for TTS conversion"""
        # Remove markdown formatting
        text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)  # Remove links
        text = re.sub(r'[*_~`]', '', text)  # Remove formatting characters
        text = re.sub(r'#+\s', '', text)  # Remove headers
        text = re.sub(r'\n\s*\n', '\n', text)  # Remove extra newlines
        text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
        return text.strip()

    def _chunk_text(self, text: str) -> list[str]:
        """Split text into manageable chunks for TTS"""
        words = text.split()
        chunks = []
        current_chunk = []
        current_length = 0
        
        for word in words:
            word_length = len(word) + 1  # +1 for space
            if current_length + word_length > self.chunk_size:
                chunks.append(' '.join(current_chunk))
                current_chunk = [word]
                current_length = word_length
            else:
                current_chunk.append(word)
                current_length += word_length
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks

    def _create_audio_player(self, audio_data: bytes, chunk_index: int = 0) -> str:
        """Create HTML audio player with controls"""
        audio_str = base64.b64encode(audio_data).decode()
        return f'''
            <audio controls {'autoplay' if chunk_index == 0 else ''} class="tts-player">
                <source src="data:audio/mp3;base64,{audio_str}" type="audio/mp3">
                Your browser does not support the audio element.
            </audio>
        '''

    def text_to_speech(self, text: str, show_progress: bool = True) -> str | None:
        """
        Convert text to speech with progress tracking and error handling
        
        Args:
            text: Input text to convert to speech
            show_progress: Whether to show progress bar
            
        Returns:
            HTML string containing audio player(s) or None if conversion fails
        """
        try:
            # Clean and prepare text
            cleaned_text = self._clean_text(text)
            if not cleaned_text:
                logger.warning("Empty text after cleaning")
                return None

            # Check cache first
            cache_path = self._get_cache_path(cleaned_text)
            if cache_path.exists():
                logger.info("Using cached audio")
                with open(cache_path, 'rb') as f:
                    return self._create_audio_player(f.read())

            # Split into chunks if text is long
            chunks = self._chunk_text(cleaned_text)
            audio_players = []
            
            # Show progress if multiple chunks
            if show_progress and len(chunks) > 1:
                progress_text = "Converting text to speech..."
                progress_bar = st.progress(0, text=progress_text)
            
            for i, chunk in enumerate(chunks):
                try:
                    # Update progress
                    if show_progress and len(chunks) > 1:
                        progress = (i + 1) / len(chunks)
                        progress_bar.progress(progress, 
                                           text=f"Converting part {i + 1} of {len(chunks)}...")
                    
                    # Convert chunk to speech
                    tts = gTTS(text=chunk, lang='en', slow=False)
                    audio_fp = io.BytesIO()
                    tts.write_to_fp(audio_fp)
                    audio_fp.seek(0)
                    
                    # Create player for this chunk
                    audio_players.append(self._create_audio_player(audio_fp.read(), i))
                    
                    # Small delay between chunks to prevent rate limiting
                    if i < len(chunks) - 1:
                        time.sleep(0.5)
                        
                except Exception as e:
                    logger.error(f"Error converting chunk {i + 1}: {str(e)}")
                    st.warning(f"Error converting part {i + 1} of the text. Some audio may be missing.")
                    continue
            
            # Clear progress bar if it exists
            if show_progress and len(chunks) > 1:
                progress_bar.empty()
            
            # Save to cache if conversion was successful
            if len(audio_players) == len(chunks):
                try:
                    with open(cache_path, 'wb') as f:
                        audio_fp.seek(0)
                        f.write(audio_fp.read())
                except Exception as e:
                    logger.error(f"Error caching audio: {str(e)}")
            
            return "\n".join(audio_players)
            
        except Exception as e:
            logger.error(f"Error in text_to_speech: {str(e)}")
            st.error("Failed to convert text to speech. Please try again.")
            return None