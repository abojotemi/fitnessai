# progress_journal.py

import streamlit as st
from datetime import datetime
import pandas as pd
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from pydantic import BaseModel
import base64
import io
from PIL import Image
import assemblyai as aai
import os
from pathlib import Path
import json
import sqlite3
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProgressEntry(BaseModel):
    """Data model for progress entries"""
    date: datetime
    weight: float
    measurements: Dict[str, float]
    mood: str
    energy_level: str
    photos: Optional[List[str]]  # Base64 encoded images
    voice_notes: Optional[str]  # Transcribed text
    text_notes: Optional[str]
    workout_intensity: str
    workout_duration: int  # in minutes
    goals_progress: Dict[str, Any]

class ProgressJournal:
    def __init__(self):
        self.setup_database()
        aai.settings.api_key = os.getenv("ASSEMBLYAI_API_KEY")
        self.aai_client = aai.Transcriber()
        
    def setup_database(self):
        """Initialize SQLite database for progress tracking"""
        conn = sqlite3.connect('fitness_progress.db')
        c = conn.cursor()
        
        # Create tables if they don't exist
        c.execute('''
            CREATE TABLE IF NOT EXISTS progress_entries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                date TEXT,
                weight REAL,
                measurements TEXT,
                mood TEXT,
                energy_level TEXT,
                workout_intensity TEXT,
                workout_duration INTEGER,
                text_notes TEXT,
                goals_progress TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        c.execute('''
            CREATE TABLE IF NOT EXISTS media_files (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                entry_id INTEGER,
                media_type TEXT,
                content TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (entry_id) REFERENCES progress_entries(id)
            )
        ''')
        
        conn.commit()
        conn.close()

    def process_voice_note(self, audio_file) -> str:
        """Transcribe voice note using AssemblyAI"""
        try:
            transcript = self.aai_client.transcribe(audio_file)
            return transcript.text
        except Exception as e:
            logger.error(f"Error transcribing voice note: {e}")
            return ""

    def process_progress_photo(self, image_file) -> str:
        """Process and encode progress photo"""
        try:
            image = Image.open(image_file)
            # Resize image to reduce storage size while maintaining quality
            max_size = (800, 800)
            image.thumbnail(max_size, Image.Resampling.LANCZOS)
            
            # Convert to JPEG format
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=85)
            return base64.b64encode(buffer.getvalue()).decode()
        except Exception as e:
            logger.error(f"Error processing progress photo: {e}")
            return ""

    def save_entry(self, entry: ProgressEntry, user_id: str):
        """Save progress entry to database"""
        conn = sqlite3.connect('fitness_progress.db')
        c = conn.cursor()
        
        try:
            # Insert main entry
            c.execute('''
                INSERT INTO progress_entries (
                    user_id, date, weight, measurements, mood, 
                    energy_level, workout_intensity, workout_duration,
                    text_notes, goals_progress
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                user_id,
                entry.date.isoformat(),
                entry.weight,
                json.dumps(entry.measurements),
                entry.mood,
                entry.energy_level,
                entry.workout_intensity,
                entry.workout_duration,
                entry.text_notes,
                json.dumps(entry.goals_progress)
            ))
            
            entry_id = c.lastrowid
            
            # Save photos
            if entry.photos:
                for photo in entry.photos:
                    c.execute('''
                        INSERT INTO media_files (entry_id, media_type, content)
                        VALUES (?, ?, ?)
                    ''', (entry_id, 'photo', photo))
            
            # Save voice note transcription
            if entry.voice_notes:
                c.execute('''
                    INSERT INTO media_files (entry_id, media_type, content)
                    VALUES (?, ?, ?)
                ''', (entry_id, 'voice_note', entry.voice_notes))
            
            conn.commit()
            return entry_id
            
        except Exception as e:
            conn.rollback()
            logger.error(f"Error saving progress entry: {e}")
            raise
        finally:
            conn.close()

    def get_entries(self, user_id: str, start_date: datetime = None, end_date: datetime = None) -> List[Dict]:
        """Retrieve progress entries with optional date range"""
        conn = sqlite3.connect('fitness_progress.db')
        c = conn.cursor()
        
        try:
            query = '''
                SELECT p.*, m.media_type, m.content
                FROM progress_entries p
                LEFT JOIN media_files m ON p.id = m.entry_id
                WHERE p.user_id = ?
            '''
            params = [user_id]
            
            if start_date:
                query += ' AND p.date >= ?'
                params.append(start_date.isoformat())
            if end_date:
                query += ' AND p.date <= ?'
                params.append(end_date.isoformat())
            
            query += ' ORDER BY p.date DESC'
            
            c.execute(query, params)
            rows = c.fetchall()
            
            # Process rows into structured data
            entries = {}
            for row in rows:
                entry_id = row[0]
                if entry_id not in entries:
                    entries[entry_id] = {
                        'date': datetime.fromisoformat(row[2]),
                        'weight': row[3],
                        'measurements': json.loads(row[4]),
                        'mood': row[5],
                        'energy_level': row[6],
                        'workout_intensity': row[7],
                        'workout_duration': row[8],
                        'text_notes': row[9],
                        'goals_progress': json.loads(row[10]),
                        'photos': [],
                        'voice_notes': None
                    }
                
                # Add media files
                if row[-2] == 'photo':
                    entries[entry_id]['photos'].append(row[-1])
                elif row[-2] == 'voice_note':
                    entries[entry_id]['voice_notes'] = row[-1]
            
            return list(entries.values())
            
        finally:
            conn.close()

class ProgressJournalUI:
    def __init__(self, journal: ProgressJournal):
        self.journal = journal

    def display_entry_form(self):
        """Render the progress entry form"""
        st.subheader("📝 Add Progress Entry")
        
        with st.form("progress_entry_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                weight = st.number_input("Weight (kg)", min_value=30.0, max_value=300.0)
                mood = st.select_slider("Mood", options=["😞", "😐", "🙂", "😊", "🤗"])
                energy_level = st.select_slider(
                    "Energy Level",
                    options=["Very Low", "Low", "Medium", "High", "Very High"]
                )
            
            with col2:
                # Measurements
                st.markdown("#### Body Measurements (cm)")
                chest = st.number_input("Chest", min_value=0.0)
                waist = st.number_input("Waist", min_value=0.0)
                hips = st.number_input("Hips", min_value=0.0)
            
            # Workout details
            col3, col4 = st.columns(2)
            with col3:
                workout_intensity = st.select_slider(
                    "Workout Intensity",
                    options=["Rest Day", "Light", "Moderate", "Intense", "Very Intense"]
                )
            with col4:
                workout_duration = st.number_input("Workout Duration (minutes)", min_value=0, max_value=300)
            
            # Media uploads
            st.markdown("#### Media")
            photos = st.file_uploader(
                "Progress Photos",
                type=['jpg', 'jpeg', 'png'],
                accept_multiple_files=True
            )
            
            voice_note = st.file_uploader("Voice Note", type=['mp3', 'wav', 'm4a'])
            
            # Text notes
            text_notes = st.text_area("Additional Notes")
            
            # Goals progress
            st.markdown("#### Goals Progress")
            goals_progress = {}
            if 'user_info' in st.session_state and st.session_state.user_info.goals:
                goals = st.session_state.user_info.goals.split('\n')
                for goal in goals:
                    if goal.strip():
                        progress = st.slider(
                            f"Progress on: {goal}",
                            min_value=0,
                            max_value=100,
                            value=50,
                            help="Slide to indicate progress percentage"
                        )
                        goals_progress[goal] = progress
            
            submitted = st.form_submit_button("Save Entry")
            
            if submitted:
                try:
                    # Process media files
                    processed_photos = []
                    if photos:
                        for photo in photos:
                            processed_photo = self.journal.process_progress_photo(photo)
                            if processed_photo:
                                processed_photos.append(processed_photo)
                    
                    voice_note_text = None
                    if voice_note:
                        voice_note_text = self.journal.process_voice_note(voice_note)
                    
                    # Create entry
                    entry = ProgressEntry(
                        date=datetime.now(),
                        weight=weight,
                        measurements={
                            'chest': chest,
                            'waist': waist,
                            'hips': hips
                        },
                        mood=mood,
                        energy_level=energy_level,
                        photos=processed_photos,
                        voice_notes=voice_note_text,
                        text_notes=text_notes,
                        workout_intensity=workout_intensity,
                        workout_duration=workout_duration,
                        goals_progress=goals_progress
                    )
                    
                    # Save entry
                    self.journal.save_entry(entry, st.session_state.user_info.name)
                    st.success("Progress entry saved successfully! 🎉")
                    st.balloons()
                    
                except Exception as e:
                    st.error(f"Error saving progress entry: {str(e)}")

    def display_progress_view(self):
        """Render progress history view"""
        st.subheader("📊 Progress History")
        
        # Date range selector
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("From Date")
        with col2:
            end_date = st.date_input("To Date")
        
        # Get entries
        entries = self.journal.get_entries(
            st.session_state.user_info.name,
            start_date=datetime.combine(start_date, datetime.min.time()),
            end_date=datetime.combine(end_date, datetime.max.time())
        )
        
        # Display entries
        for entry in entries:
            with st.expander(f"Entry: {entry['date'].strftime('%Y-%m-%d %H:%M')}"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"**Weight:** {entry['weight']} kg")
                    st.markdown(f"**Mood:** {entry['mood']}")
                    st.markdown(f"**Energy Level:** {entry['energy_level']}")
                    st.markdown(f"**Workout Intensity:** {entry['workout_intensity']}")
                    st.markdown(f"**Workout Duration:** {entry['workout_duration']} minutes")
                
                with col2:
                    st.markdown("**Measurements:**")
                    for part, measurement in entry['measurements'].items():
                        st.markdown(f"- {part.title()}: {measurement} cm")
                
                # Display photos in a grid
                if entry['photos']:
                    st.markdown("**Progress Photos:**")
                    photo_cols = st.columns(len(entry['photos']))
                    for idx, photo in enumerate(entry['photos']):
                        with photo_cols[idx]:
                            st.image(
                                f"data:image/jpeg;base64,{photo}",
                                use_container_width=True
                            )
                
                # Display voice note transcription
                if entry['voice_notes']:
                    st.markdown("**Voice Note Transcription:**")
                    st.markdown(f"*{entry['voice_notes']}*")
                
                # Display text notes
                if entry['text_notes']:
                    st.markdown("**Notes:**")
                    st.markdown(entry['text_notes'])
                
                # Display goals progress
                if entry['goals_progress']:
                    st.markdown("**Goals Progress:**")
                    for goal, progress in entry['goals_progress'].items():
                        st.progress(progress / 100)
                        st.markdown(f"*{goal}:* {progress}%")

def initialize_progress_journal():
    """Initialize and return ProgressJournal instance"""
    journal = ProgressJournal()
    ui = ProgressJournalUI(journal)
    return ui
