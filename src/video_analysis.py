from datetime import datetime
import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi
from urllib.parse import urlparse, parse_qs
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate
import yt_dlp
import os
import shutil
from dotenv import load_dotenv
import time
import logging


# Initialize logging and environment
# logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()

# Constants
PERSIST_DIRECTORY = "./chroma_db"
K_RESULTS = 3  # Hardcoded number of results


# Clear ChromaDB system cache to prevent tenant errors
# os.makedirs("video_cache", exist_ok=True)
# chromadb.api.client.SharedSystemClient.clear_system_cache()

# Set your Google API key in Streamlit secrets or environment variables
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
# Initialize persist directory

# System prompt for the LLM
SYSTEM_PROMPT = """You are an intelligent video content analyzer. Your task is to provide accurate, relevant answers to questions about the video content using the provided context from the video transcript. Please follow these guidelines:

1. Base your answers solely on the provided context
2. If the context doesn't contain enough information to answer the question, clearly state that
3. Include relevant quotes from the transcript when appropriate
4. Maintain a natural, conversational tone while being informative

Title: {title}
Context: {context}

Question: {question}

Please provide a clear and concise answer based on the above context."""

def get_video_id(url):
    """Extract video ID from YouTube URL"""
    parsed_url = urlparse(url)
    if parsed_url.hostname == 'youtu.be':
        return parsed_url.path[1:]
    if parsed_url.hostname in ('www.youtube.com', 'youtube.com'):
        if parsed_url.path == '/watch':
            return parse_qs(parsed_url.query)['v'][0]
    return None

def get_video_title(url):
    """Get video title using yt-dlp"""
    ydl_opts = {
        'quiet': True,
        'no_warnings': True,
        'extract_flat': True
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            result = ydl.extract_info(url, download=False)
            return result.get('title', None)
    except Exception as e:
        st.error(f"Error fetching video title: {str(e)}")
        return None

def get_transcript(video_id):
    """Get video transcript"""
    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        return ' '.join([entry['text'] for entry in transcript_list])
    except Exception as e:
        st.error(f"Error fetching transcript: {str(e)}")
        return None

def create_embeddings_and_store(text, title):
    """Create embeddings and store in Chroma"""
    # Create a clean directory for the new embeddings
    collection_path = os.path.join(PERSIST_DIRECTORY, title.replace(" ", "_")[:100])
    if os.path.exists(collection_path):
        shutil.rmtree(collection_path)
    os.makedirs(collection_path)
    
    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=750,
        chunk_overlap=50
    )
    chunks = text_splitter.split_text(text)
    
    # Initialize Google's embedding model
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    # Create and persist vector store
    vectorstore = Chroma.from_texts(
        texts=chunks,
        embedding=embeddings,
        persist_directory=collection_path
    )
    
    return vectorstore

def get_llm_response(title: str, query: str, context: str) -> str:
    """Get LLM response based on query and context"""
    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0.7,
        )
        
        messages = [
            ('system', SYSTEM_PROMPT),
            ('human', f"Context: {context}\n\nQuestion: {query}")
        ]
        
        prompt = ChatPromptTemplate.from_messages(messages)
        chain = prompt | llm
        
        response = chain.invoke({
            'title': title,
            'context': context,
            'question': query
        })
        
        return response.content
    except Exception as e:
        logger.error(f"Error getting LLM response: {str(e)}")
        return None


class VideoRAGManager:
    def __init__(self):
        # Initialize persistent session states
        if 'video_history' not in st.session_state:
            st.session_state.video_history = []
        if 'current_video' not in st.session_state:
            st.session_state.current_video = None
        if 'video_data' not in st.session_state:
            st.session_state.video_data = {}
        if 'total_processed' not in st.session_state:
            st.session_state.total_processed = 0
        if 'processing' not in st.session_state:
            st.session_state.processing = False
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = {}

    def add_to_history(self, video_id, title):
        """Add video to history with timestamp"""
        if video_id not in [v['id'] for v in st.session_state.video_history]:
            st.session_state.video_history.insert(0, {
                'id': video_id,
                'title': title,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M")
            })
            # Keep only last 5 videos
            st.session_state.video_history = st.session_state.video_history[:5]

    def add_to_chat_history(self, video_id, query, response):
        """Store chat history for each video"""
        if video_id not in st.session_state.chat_history:
            st.session_state.chat_history[video_id] = []
        st.session_state.chat_history[video_id].append({
            'query': query,
            'response': response,
            'timestamp': datetime.now().strftime("%H:%M")
        })

def display_video_tab():
    manager = VideoRAGManager()

    # Create two columns: sidebar and main content
    col1, col2 = st.columns([1, 3])

    with col1:
        st.markdown("### 📊 Dashboard")
        st.metric("Videos Analyzed", st.session_state.total_processed)
        
        # Recent Videos Section
        st.markdown("### 📜 Recent Videos")
        for video in st.session_state.video_history:
            with st.expander(f"📺 {video['title'][:30]}...", expanded=False):
                st.write(f"Analyzed: {video['timestamp']}")
                if st.button("Load Video", key=f"load_{video['id']}"):
                    st.session_state.current_video = video['id']
                    st.experimental_rerun()

        # Help Section
        with st.expander("ℹ️ Help", expanded=False):
            st.markdown("""
            **How to use:**
            1. Paste a YouTube URL
            2. Wait for processing
            3. Ask questions about the video
            4. View your chat history
            
            **Tips:**
            - Be specific in your questions
            - Use keywords from the video
            - Questions are analyzed using AI
            """)

    with col2:
        st.markdown("## 🎥 Video Content Analysis")
        st.write("Post the link to your fitness related video below, and ask questions based on the video.")
        
        # URL Input with clear button
        url_col1, url_col2 = st.columns([4, 1])
        with url_col1:
            url = st.text_input("🔗 Enter YouTube URL:", key="url_input", 
                              placeholder="https://youtube.com/watch?v=...")
        with url_col2:
            if st.button("🗑️ Clear", type="secondary"):
                if 'url_input' in st.session_state:
                    st.session_state.url_input = ""
                    st.rerun()

        if url and not st.session_state.processing:
            video_id = get_video_id(url)
            if video_id:
                # Process Video
                if video_id not in st.session_state.video_data:
                    st.session_state.processing = True
                    
                    with st.status("🎬 Processing Video...", expanded=True) as status:
                        st.write("Fetching video details...")
                        title = get_video_title(url)
                        transcript = get_transcript(video_id)
                        
                        if title and transcript:
                            st.write(f"📝 Analyzing: {title}")
                            try:
                                # Create embeddings with progress
                                progress_text = "Creating AI embeddings..."
                                my_bar = st.progress(0, text=progress_text)
                                for i in range(100):
                                    time.sleep(0.01)
                                    my_bar.progress(i + 1, text=progress_text)
                                
                                vectorstore = create_embeddings_and_store(transcript, title)
                                
                                # Store data and update history
                                st.session_state.video_data[video_id] = {
                                    'title': title,
                                    'transcript': transcript,
                                    'vectorstore': vectorstore,
                                    'processed': True
                                }
                                manager.add_to_history(video_id, title)
                                st.session_state.total_processed += 1
                                status.update(label="✅ Processing Complete!", state="complete")
                                
                            except Exception as e:
                                st.error(f"❌ Error: {str(e)}")
                                status.update(label="❌ Processing Failed", state="error")
                    
                    st.session_state.processing = False
                
                # Display Video and Chat Interface
                video_data = st.session_state.video_data[video_id]
                
                # Video Player
                with st.expander("📺 Video Player", expanded=True):
                    st.video(url)
                    st.caption(f"Title: {video_data['title']}")

                # Chat Interface
                st.markdown("### 💬 Ask About the Video")
                
                # Display chat history
                if video_id in st.session_state.chat_history:
                    for chat in st.session_state.chat_history[video_id]:
                        with st.chat_message("user"):
                            st.write(f"🕒 {chat['timestamp']}")
                            st.write(chat['query'])
                        with st.chat_message("assistant"):
                            st.write(chat['response'])

                # Query input
                query = st.chat_input("Type your question here...")
                
                if query:
                    with st.chat_message("user"):
                        st.write(query)
                    
                    with st.chat_message("assistant"):
                        with st.spinner("🤔 Thinking..."):
                            docs = video_data['vectorstore'].similarity_search(query, k=K_RESULTS)
                            context = "\n".join([doc.page_content for doc in docs])
                            response = get_llm_response(video_data['title'], query, context)
                            
                            if response:
                                st.write(response)
                                manager.add_to_chat_history(video_id, query, response)
            else:
                st.error("❌ Invalid YouTube URL")