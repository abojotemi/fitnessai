import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
from pathlib import Path
import logging
from typing import Dict, Any, List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize analytics data storage
ANALYTICS_FILE = Path("data/analytics_data.json")
ANALYTICS_FILE.parent.mkdir(parents=True, exist_ok=True)

class AnalyticsManager:
    def __init__(self):
        self.analytics_data = self._load_analytics_data()
    
    def _load_analytics_data(self) -> Dict:
        """Load analytics data from file"""
        try:
            if ANALYTICS_FILE.exists():
                with open(ANALYTICS_FILE, 'r') as f:
                    return json.load(f)
            return {
                'user_interactions': [],
                'tts_requests': [],
                'stt_requests': [],
                'response_times': [],
                'workout_generations': [],
                'diet_analyses': []
            }
        except Exception as e:
            logger.error(f"Error loading analytics data: {e}")
            return {
                'user_interactions': [],
                'tts_requests': [],
                'stt_requests': [],
                'response_times': [],
                'workout_generations': [],
                'diet_analyses': []
            }
    
    def _save_analytics_data(self):
        """Save analytics data to file"""
        try:
            with open(ANALYTICS_FILE, 'w') as f:
                json.dump(self.analytics_data, f)
        except Exception as e:
            logger.error(f"Error saving analytics data: {e}")

    def log_event(self, category: str, data: Dict[str, Any]):
        """Log an analytics event"""
        try:
            event = {
                'timestamp': datetime.now().isoformat(),
                'data': data
            }
            if category in self.analytics_data:
                self.analytics_data[category].append(event)
                self._save_analytics_data()
        except Exception as e:
            logger.error(f"Error logging event: {e}")

def log_user_interaction(action: str, details: Dict[str, Any]):
    """Log user interaction events"""
    try:
        analytics_manager = AnalyticsManager()
        analytics_manager.log_event('user_interactions', {
            'action': action,
            'details': details
        })
    except Exception as e:
        logger.error(f"Error logging user interaction: {e}")

def log_tts_request(text_length: int, processing_time: float):
    """Log text-to-speech request metrics"""
    try:
        analytics_manager = AnalyticsManager()
        analytics_manager.log_event('tts_requests', {
            'text_length': text_length,
            'processing_time': processing_time
        })
    except Exception as e:
        logger.error(f"Error logging TTS request: {e}")


def log_response_time(feature: str, response_time: float):
    """Log response time metrics"""
    try:
        analytics_manager = AnalyticsManager()
        analytics_manager.log_event('response_times', {
            'feature': feature,
            'response_time': response_time
        })
    except Exception as e:
        logger.error(f"Error logging response time: {e}")

def create_time_series_chart(data: List[Dict], value_key: str, title: str) -> go.Figure:
    """Create a time series chart using plotly"""
    try:
        df = pd.DataFrame(data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        fig = px.line(
            df,
            x='timestamp',
            y=value_key,
            title=title
        )
        fig.update_layout(
            xaxis_title="Time",
            yaxis_title=value_key.replace('_', ' ').title(),
            showlegend=True
        )
        return fig
    except Exception as e:
        logger.error(f"Error creating time series chart: {e}")
        return None

def display_analytics():
    """Display analytics dashboard"""
    try:
        st.title("📊 Analytics Dashboard")
        
        # Load analytics data
        analytics_manager = AnalyticsManager()
        data = analytics_manager.analytics_data
        
        # Create tabs for different analytics sections
        tabs = st.tabs(["Usage Metrics", "Performance Metrics", "User Engagement", "Feature Usage"])
        
        # Usage Metrics Tab
        with tabs[0]:
            st.header("Usage Metrics")
            
            # Key metrics in columns
            col1, col2, col3 = st.columns(3)
            
            with col1:
                total_interactions = len(data['user_interactions'])
                st.metric("Total Interactions", total_interactions)
            
            with col2:
                if data['user_interactions']:
                    last_24h_interactions = sum(
                        1 for x in data['user_interactions']
                        if datetime.fromisoformat(x['timestamp']) > datetime.now() - timedelta(days=1)
                    )
                    st.metric("Last 24h Interactions", last_24h_interactions)
            
            with col3:
                if data['user_interactions']:
                    active_features = len({
                        x['data']['action'] for x in data['user_interactions']
                    })
                    st.metric("Active Features", active_features)
            
            # Usage trends chart
            if data['user_interactions']:
                df_interactions = pd.DataFrame([
                    {
                        'timestamp': datetime.fromisoformat(x['timestamp']),
                        'action': x['data']['action']
                    }
                    for x in data['user_interactions']
                ])
                
                daily_usage = df_interactions.groupby(
                    df_interactions['timestamp'].dt.date
                ).size().reset_index()
                daily_usage.columns = ['date', 'count']
                
                fig = px.line(
                    daily_usage,
                    x='date',
                    y='count',
                    title='Daily Usage Trends'
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Performance Metrics Tab
        with tabs[1]:
            st.header("Performance Metrics")
            
            if data['response_times']:
                # Response time metrics
                df_response = pd.DataFrame([
                    {
                        'timestamp': datetime.fromisoformat(x['timestamp']),
                        'feature': x['data']['feature'],
                        'response_time': x['data']['response_time']
                    }
                    for x in data['response_times']
                ])
                
                # Average response times by feature
                avg_response = df_response.groupby('feature')['response_time'].mean()
                
                fig = px.bar(
                    avg_response.reset_index(),
                    x='feature',
                    y='response_time',
                    title='Average Response Time by Feature'
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Response time trends
                fig_trend = px.line(
                    df_response,
                    x='timestamp',
                    y='response_time',
                    color='feature',
                    title='Response Time Trends'
                )
                st.plotly_chart(fig_trend, use_container_width=True)
        
        # User Engagement Tab
        with tabs[2]:
            st.header("User Engagement")
            
            if data['user_interactions']:
                # Feature usage distribution
                feature_counts = df_interactions['action'].value_counts()
                
                fig = px.pie(
                    values=feature_counts.values,
                    names=feature_counts.index,
                    title='Feature Usage Distribution'
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # User engagement timeline
                st.subheader("User Engagement Timeline")
                timeline_df = df_interactions.copy()
                timeline_df['date'] = timeline_df['timestamp'].dt.date
                timeline_df['hour'] = timeline_df['timestamp'].dt.hour
                
                engagement_heatmap = pd.crosstab(
                    timeline_df['date'],
                    timeline_df['hour']
                ).fillna(0)
                
                fig = px.imshow(
                    engagement_heatmap,
                    title='Engagement Heatmap (Hour of Day)',
                    labels=dict(x="Hour of Day", y="Date", color="Interactions")
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Feature Usage Tab
        with tabs[3]:
            st.header("Feature Usage Analysis")
            
            if data['workout_generations'] or data['diet_analyses']:
                col1, col2 = st.columns(2)
                
                with col1:
                    if data['workout_generations']:
                        workout_df = pd.DataFrame([
                            {
                                'timestamp': datetime.fromisoformat(x['timestamp']),
                                'type': x['data'].get('type', 'Unknown')
                            }
                            for x in data['workout_generations']
                        ])
                        
                        fig = px.pie(
                            workout_df,
                            names='type',
                            title='Workout Types Generated'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    if data['diet_analyses']:
                        diet_df = pd.DataFrame([
                            {
                                'timestamp': datetime.fromisoformat(x['timestamp']),
                                'category': x['data'].get('category', 'Unknown')
                            }
                            for x in data['diet_analyses']
                        ])
                        
                        fig = px.pie(
                            diet_df,
                            names='category',
                            title='Diet Analysis Categories'
                        )
                        st.plotly_chart(fig, use_container_width=True)
            
            # Export functionality
            st.subheader("Export Analytics Data")
            if st.button("Export to CSV"):
                try:
                    # Convert analytics data to DataFrame
                    export_data = pd.DataFrame(data['user_interactions'])
                    csv = export_data.to_csv(index=False)
                    
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name="analytics_export.csv",
                        mime="text/csv"
                    )
                except Exception as e:
                    logger.error(f"Error exporting analytics data: {e}")
                    st.error("Error exporting data. Please try again.")
    
    except Exception as e:
        logger.error(f"Error displaying analytics: {e}")
        st.error("An error occurred while loading analytics. Please try again.")