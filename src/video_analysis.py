from datetime import datetime
import httpx
import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi
from urllib.parse import urlparse, parse_qs
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
import yt_dlp
import logging
from dotenv import load_dotenv
import time

from llm import LLMHandler
from proxy import ProxyRotator

# Initialize logging and environment
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
load_dotenv()

# Constants
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = "video-rag"  # Choose your index name
K_RESULTS = 3

# Set your Google API key
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
if PINECONE_INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=768,  # Dimension for Google's embedding model
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )
# System prompt remains the same

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

proxy_rotator = ProxyRotator()
def get_transcript_without_proxy(video_id):
    """Fallback method to get transcript without proxy"""
    try:
        # Get list of available transcripts
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        
        # Try to get English transcript first
        try:
            transcript = transcript_list.find_transcript(['en'])
        except:
            # If English isn't available, get the first available transcript
            available_transcripts = transcript_list.manual_transcripts
            if available_transcripts:
                transcript = list(available_transcripts.values())[0]
            else:
                # Try auto-generated transcripts
                available_transcripts = transcript_list.generated_transcripts
                if available_transcripts:
                    transcript = list(available_transcripts.values())[0]
                else:
                    raise Exception("No transcripts available")

        # Fetch the actual transcript data
        transcript_data = transcript.fetch()
        
        # Safely extract text
        transcript_texts = []
        for entry in transcript_data:
            if isinstance(entry, dict) and 'text' in entry:
                transcript_texts.append(entry['text'])
                
        if not transcript_texts:
            raise Exception("No valid text entries found")
            
        return ' '.join(transcript_texts)
        
    except Exception as e:
        logger.error(f"Error fetching transcript without proxy: {str(e)}")
        st.error(f"Error fetching transcript without proxy: {str(e)}")
        return None

def get_transcript(video_id):
    """Get transcript with proxy rotation and fallback"""
    max_retries = 3
    retries = 0
    
    while retries < max_retries:
        proxies = proxy_rotator.get_proxy()
        if not proxies:
            logger.warning("No proxies available. Trying without proxy...")
            return get_transcript_without_proxy(video_id)
        
        try:
            # Configure proxy
            proxies_transport = httpx.HTTPTransport(proxy=proxies['https'])
            client = httpx.Client(transport=proxies_transport, timeout=10.0)  # 10 second timeout
            YouTubeTranscriptApi.http_client = client
            
            # Get list of available transcripts
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
            
            # Try to get English transcript first
            try:
                transcript = transcript_list.find_transcript(['en'])
            except:
                # If English isn't available, get the first available transcript
                available_transcripts = transcript_list.manual_transcripts
                if available_transcripts:
                    transcript = list(available_transcripts.values())[0]
                else:
                    # Try auto-generated transcripts
                    available_transcripts = transcript_list.generated_transcripts
                    if available_transcripts:
                        transcript = list(available_transcripts.values())[0]
                    else:
                        raise Exception("No transcripts available")

            # Fetch the actual transcript data
            transcript_data = transcript.fetch()
            
            # Safely extract text
            transcript_texts = []
            for entry in transcript_data:
                if isinstance(entry, dict) and 'text' in entry:
                    transcript_texts.append(entry['text'])
            
            if not transcript_texts:
                raise Exception("No valid text entries found")
            
            return ' '.join(transcript_texts)
            
        except Exception as e:
            logger.error(f"Error with proxy {proxies['https']}: {str(e)}")
            retries += 1
            if retries == max_retries:
                logger.warning("Max retries reached. Trying without proxy...")
                return get_transcript_without_proxy(video_id)
        
        finally:
            if 'client' in locals():
                client.close()
    
    return None
        
def create_embeddings_and_store(text, video_id, title):
    """Create embeddings and store in Pinecone"""
    try:
        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=750,
            chunk_overlap=50
        )
        chunks = text_splitter.split_text(text)
        
        # Initialize Google's embedding model
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        
        # Create metadata for each chunk
        texts_with_metadata = [
            {
                "text": chunk,
                "video_id": video_id,
                "title": title,
                "chunk_id": i
            } for i, chunk in enumerate(chunks)
        ]
        
        # Initialize or get existing index

        index = pc.Index(PINECONE_INDEX_NAME)
        
        # Create and return vector store
        vectorstore = PineconeVectorStore(
            index=index,
            embedding=embeddings,
            text_key="text"
        )
        
        # Upsert documents with metadata
        vectorstore.add_texts(
            texts=[d["text"] for d in texts_with_metadata],
            metadatas=texts_with_metadata
        )
        
        return vectorstore
    
    except Exception as e:
        logger.error(f"Error creating embeddings and vectorstore: {e}")
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
        if 'video_id' not in st.session_state:
            st.session_state.video_id = None

    def add_to_history(self, video_id, title):
        if video_id not in [v['id'] for v in st.session_state.video_history]:
            st.session_state.video_history.insert(0, {
                'id': video_id,
                'title': title,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M")
            })
            st.session_state.video_history = st.session_state.video_history[:5]

    def add_to_chat_history(self, video_id, query, response):
        if video_id not in st.session_state.chat_history:
            st.session_state.chat_history[video_id] = []
        st.session_state.chat_history[video_id].append({
            'query': query,
            'response': response,
            'timestamp': datetime.now().strftime("%H:%M")
        })

    def get_vectorstore(self, video_id):
        """Get Pinecone vectorstore for video"""
        try:
            index = pc.Index(PINECONE_INDEX_NAME)
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            
            # Create vector store with filter for specific video
            vectorstore = PineconeVectorStore(
                index=index,
                embedding=embeddings,
                text_key="text"
            )
            
            return vectorstore
        except Exception as e:
            logger.error(f"Error getting vectorstore: {e}")
            return None
        
        

class VideoAnalyzer:
    def __init__(self):
        self.manager = VideoRAGManager()
        
    def display(self):

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
                    st.divider()
                    st.write(video['title'])
                    

            # Help Section
            with st.expander("ℹ️ Help", expanded=True):
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
            st.write("Paste the url of the fitness related YouTube video you want analyzed. Then ask questions based on the video.")
            
            # URL Input with clear button
            url_col1, url_col2 = st.columns([4, 1])
            with url_col1:
                st.session_state.current_video = st.text_input("🔗 Enter YouTube URL:", key="url_input", 
                                placeholder="https://youtube.com/watch?v=...", value=st.session_state.current_video or "")
            with url_col2:
                if st.button("🗑️ Clear", type="secondary"):
                    if 'url_input' in st.session_state:
                        st.session_state.url_input = ""
                        st.rerun()

            if st.session_state.current_video and not st.session_state.processing:
                st.session_state.video_id = get_video_id(st.session_state.current_video)
                if st.session_state.video_id:
                    # Get vectorstore for video
                    vectorstore = self.manager.get_vectorstore(st.session_state.video_id)
                    
                    # Process Video if not already processed
                    if st.session_state.video_id not in st.session_state.video_data:
                        st.session_state.processing = True
                        
                        with st.status("🎬 Processing Video...", expanded=True) as status:
                            st.write("Fetching video details...")
                            title = get_video_title(st.session_state.current_video)
                            transcript = get_transcript(st.session_state.video_id)
                            
                            if title and transcript:
                                st.write(f"📝 Analyzing: {title}")  
                                try:
                                    progress_text = "Creating AI embeddings..."
                                    my_bar = st.progress(0, text=progress_text)
                                    for i in range(100):
                                        time.sleep(0.01)
                                        my_bar.progress(i + 1, text=progress_text)
                                    
                                    vectorstore = create_embeddings_and_store(transcript, st.session_state.video_id, title)
                                    
                                    st.session_state.video_data[st.session_state.video_id] = {
                                        'title': title,
                                        'transcript': transcript,
                                        'vectorstore': vectorstore,
                                        'processed': True
                                    }
                                    self.manager.add_to_history(st.session_state.video_id, title)
                                    st.session_state.total_processed += 1
                                    status.update(label="✅ Processing Complete!", state="complete")
                                    
                                except Exception as e:
                                    st.error(f"❌ Error: {str(e)}")
                                    status.update(label="❌ Processing Failed", state="error")
                        
                        st.session_state.processing = False
                    
                    # Display Video and Chat Interface
                    video_data = st.session_state.video_data[st.session_state.video_id]
                    
                    # Video Player
                    with st.expander("📺 Video Player", expanded=True):
                        st.video(st.session_state.current_video)
                        st.caption(f"Title: {video_data['title']}")

                    # Chat Interface
                    st.markdown("### 💬 Ask About the Video")
                    
                    # Display chat history
                    if st.session_state.video_id in st.session_state.chat_history:
                        for chat in st.session_state.chat_history[st.session_state.video_id]:
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
                                # Use filter to get only relevant documents for this video
                                docs = video_data['vectorstore'].similarity_search(
                                    query,
                                    k=K_RESULTS,
                                    filter={"video_id": st.session_state.video_id}
                                )
                                context = "\n".join([doc.page_content for doc in docs])
                                response = LLMHandler.video_analyzer_llm(video_data['title'], query, context)
                                
                                if response:
                                    st.write(response)
                                    self.manager.add_to_chat_history(st.session_state.video_id, query, response)
                else:
                    st.error("❌ Invalid YouTube URL")