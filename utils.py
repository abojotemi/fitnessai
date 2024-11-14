import re
import base64
import io
from gtts import gTTS
import assemblyai as aai
import os
from dotenv import load_dotenv

load_dotenv()

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