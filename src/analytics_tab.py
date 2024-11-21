import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
from pathlib import Path
import logging
from typing import Dict, Any, List
from collections import defaultdict

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
    
    def _get_default_analytics(self) -> Dict:
        """Return default analytics structure"""
        return {
            'user_interactions': [],
            'tts_requests': [],
            'stt_requests': [],
            'response_times': [],
            'workout_generations': [],
            'diet_analyses': [],
            'video_analysis': [],
            'video_generation': [],
            'food_generation': [],
            'feature_performance': {
                'workout': {'success_rate': 0, 'avg_time': 0, 'total_requests': 0},
                'diet': {'success_rate': 0, 'avg_time': 0, 'total_requests': 0},
                'video_analysis': {'success_rate': 0, 'avg_time': 0, 'total_requests': 0},
                'video_generation': {'success_rate': 0, 'avg_time': 0, 'total_requests': 0},
                'food_generation': {'success_rate': 0, 'avg_time': 0, 'total_requests': 0},
                'qa': {'success_rate': 0, 'avg_time': 0, 'total_requests': 0}
            },
            'error_tracking': {},
            'user_metrics': {
                'total_users': 0,
                'active_users': [],  # Changed from set() to list for JSON serialization
                'country_distribution': {},
                'platform_usage': {}
            }
        }

    def _load_analytics_data(self) -> Dict:
        """Load analytics data from file with proper initialization"""
        try:
            if ANALYTICS_FILE.exists():
                with open(ANALYTICS_FILE, 'r') as f:
                    data = json.load(f)
                    # Ensure all required keys exist
                    default_data = self._get_default_analytics()
                    for key in default_data:
                        if key not in data:
                            data[key] = default_data[key]
                    return data
            return self._get_default_analytics()
        except Exception as e:
            logger.error(f"Error loading analytics data: {e}")
            return self._get_default_analytics()
    
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

    def update_feature_performance(self, feature: str, success: bool, processing_time: float):
        """Update performance metrics for a specific feature"""
        try:
            perf = self.analytics_data['feature_performance'][feature]
            perf['total_requests'] += 1
            if success:
                new_success_rate = ((perf['success_rate'] * (perf['total_requests'] - 1)) + 100) / perf['total_requests']
                perf['success_rate'] = round(new_success_rate, 2)
            else:
                new_success_rate = (perf['success_rate'] * (perf['total_requests'] - 1)) / perf['total_requests']
                perf['success_rate'] = round(new_success_rate, 2)
            
            # Update average processing time
            perf['avg_time'] = round(((perf['avg_time'] * (perf['total_requests'] - 1)) + processing_time) / perf['total_requests'], 2)
            
            self._save_analytics_data()
        except Exception as e:
            logger.error(f"Error updating feature performance: {e}")

    def update_user_metrics(self, user_id: str, country: str = None, platform: str = None):
        """Update user-related metrics"""
        try:
            metrics = self.analytics_data['user_metrics']
            
            # Update total users and active users if new
            if user_id not in metrics['active_users']:
                metrics['total_users'] += 1
                metrics['active_users'].append(user_id)
            
            # Update country distribution
            if country:
                metrics['country_distribution'][country] = metrics['country_distribution'].get(country, 0) + 1
            
            # Update platform usage
            if platform:
                metrics['platform_usage'][platform] = metrics['platform_usage'].get(platform, 0) + 1
            
            # Save the updated metrics
            self._save_analytics_data()
            logger.info(f"Updated user metrics for user {user_id}")
        except Exception as e:
            logger.error(f"Error updating user metrics: {e}")

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

def log_video_analysis(success: bool, processing_time: float, video_type: str, error: str = None):
    """Log video analysis metrics"""
    try:
        analytics_manager = AnalyticsManager()
        analytics_manager.update_feature_performance('video_analysis', success, processing_time)
        analytics_manager.log_event('video_analysis', {
            'success': success,
            'processing_time': processing_time,
            'video_type': video_type,
            'error': error
        })
    except Exception as e:
        logger.error(f"Error logging video analysis: {e}")

def log_video_generation(success: bool, processing_time: float, error: str = None):
    """Log video generation metrics"""
    try:
        analytics_manager = AnalyticsManager()
        analytics_manager.update_feature_performance('video_generation', success, processing_time)
        analytics_manager.log_event('video_generation', {
            'success': success,
            'processing_time': processing_time,
            'error': error
        })
    except Exception as e:
        logger.error(f"Error logging video generation: {e}")

def log_food_generation(success: bool, processing_time: float, error: str = None):
    """Log food generation metrics"""
    try:
        analytics_manager = AnalyticsManager()
        analytics_manager.update_feature_performance('food_generation', success, processing_time)
        analytics_manager.log_event('food_generation', {
            'success': success,
            'processing_time': processing_time,
            'error': error
        })
    except Exception as e:
        logger.error(f"Error logging food generation: {e}")

def log_diet_analysis(success: bool, processing_time: float, user_id: str, error: str = None):
    """Log diet analysis metrics"""
    try:
        analytics_manager = AnalyticsManager()
        analytics_manager.update_feature_performance('diet', success, processing_time)
        analytics_manager.log_event('diet_analyses', {
            'success': success,
            'processing_time': processing_time,
            'user_id': user_id,
            'error': error,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Error logging diet analysis: {e}")

def display_analytics():
    """Display analytics dashboard with all metrics"""
    st.title("📊 Analytics Dashboard")
    
    analytics_manager = AnalyticsManager()
    data = analytics_manager.analytics_data

    # Create tabs for different analytics views
    tabs = st.tabs(["Overview", "Feature Performance", "User Metrics", "Response Times", "Error Analysis"])

    with tabs[0]:
        st.header("Overview")
        
        # Create metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_interactions = len(data['user_interactions'])
            st.metric("Total Interactions", total_interactions)
        
        with col2:
            total_users = data['user_metrics']['total_users']
            st.metric("Total Users", total_users)
        
        with col3:
            active_users = len(data['user_metrics']['active_users'])
            st.metric("Active Users", active_users)
        
        with col4:
            avg_response_time = sum(entry['data']['response_time'] 
                                  for entry in data['response_times']) / len(data['response_times']) if data['response_times'] else 0
            st.metric("Avg Response Time", f"{avg_response_time:.2f}s")

        

        # Create feature usage chart
        feature_usage = {
            'Workout Generation': len(data['workout_generations']),
            'Diet Analysis': len(data['diet_analyses']),
            'Video Analysis': len(data['video_analysis']),
            'Video Generation': len(data['video_generation']),
            'Food Generation': len(data['food_generation'])
        }
        
        fig = px.bar(
            x=list(feature_usage.keys()),
            y=list(feature_usage.values()),
            title="Feature Usage",
            labels={'x': 'Feature', 'y': 'Usage Count'}
        )
        st.plotly_chart(fig, use_container_width=True)

    with tabs[1]:
        st.header("Feature Performance")
        
        # Create performance metrics table
        performance_data = []
        for feature, metrics in data['feature_performance'].items():
            performance_data.append({
                'Feature': feature.replace('_', ' ').title(),
                'Success Rate': f"{metrics['success_rate']}%",
                'Avg Time': f"{metrics['avg_time']}s",
                'Total Requests': metrics['total_requests']
            })
        
        if performance_data:
            df = pd.DataFrame(performance_data)
            st.dataframe(df, use_container_width=True)
            
            # Create success rate chart
            fig = px.bar(
                df,
                x='Feature',
                y='Success Rate',
                title="Feature Success Rates",
                text='Success Rate'
            )
            st.plotly_chart(fig, use_container_width=True)


    with tabs[2]:
        st.header("User Metrics")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Users", data['user_metrics']['total_users'])
        with col2:
            st.metric("Active Users", len(data['user_metrics']['active_users']))
        with col3:
            # Calculate returning users (users who have used the app more than once)
            platform_usage = sum(data['user_metrics']['platform_usage'].values())
            returning_users = platform_usage - data['user_metrics']['total_users']
            st.metric("Returning Users", max(0, returning_users))
        
        # Geographic distribution
        if data['user_metrics']['country_distribution']:
            st.subheader("Geographic Distribution")
            country_data = pd.DataFrame(
                list(data['user_metrics']['country_distribution'].items()),
                columns=['Country', 'Users']
            )
            
            # Create two visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                # Bar chart for top countries
                fig_bar = px.bar(
                    country_data.sort_values('Users', ascending=False).head(10),
                    x='Country',
                    y='Users',
                    title="Top 10 Countries by Users"
                )
                st.plotly_chart(fig_bar, use_container_width=True)
            
            with col2:
                # World map visualization
                fig_map = px.choropleth(
                    country_data,
                    locations='Country',
                    locationmode='country names',
                    color='Users',
                    title="Global User Distribution",
                    color_continuous_scale='Viridis'
                )
                fig_map.update_layout(height=400)
                st.plotly_chart(fig_map, use_container_width=True)
        
        # Platform usage
        if data['user_metrics']['platform_usage']:
            st.subheader("Platform Usage")
            platform_data = pd.DataFrame(
                list(data['user_metrics']['platform_usage'].items()),
                columns=['Platform', 'Usage']
            )
            
            # Create two visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                # Pie chart
                fig_pie = px.pie(
                    platform_data,
                    values='Usage',
                    names='Platform',
                    title="Platform Distribution"
                )
                st.plotly_chart(fig_pie, use_container_width=True)
            
            with col2:
                # Bar chart
                fig_bar = px.bar(
                    platform_data,
                    x='Platform',
                    y='Usage',
                    title="Platform Usage Comparison"
                )
                st.plotly_chart(fig_bar, use_container_width=True)
        
        # User engagement over time
        if data['user_interactions']:
            st.subheader("User Engagement Over Time")
            
            # Create daily active users chart
            interactions_df = pd.DataFrame([
                {
                    'date': datetime.fromisoformat(entry['timestamp']).date(),
                    'user_id': entry['data'].get('user_id', 'unknown')
                }
                for entry in data['user_interactions']
            ])
            
            daily_users = interactions_df.groupby('date')['user_id'].nunique().reset_index()
            fig_engagement = px.line(
                daily_users,
                x='date',
                y='user_id',
                title="Daily Active Users",
                labels={'user_id': 'Number of Users', 'date': 'Date'}
            )
            st.plotly_chart(fig_engagement, use_container_width=True)

    with tabs[3]:
        st.header("Response Times")
        
        if data['response_times']:
            # Convert response times data to DataFrame
            response_times_df = pd.DataFrame([
                {
                    'timestamp': datetime.fromisoformat(entry['timestamp']),
                    'response_time': entry['data']['response_time'],
                    'feature': entry['data'].get('feature', 'Unknown')
                }
                for entry in data['response_times']
            ])
            
            # Create time series chart
            fig = px.line(
                response_times_df,
                x='timestamp',
                y='response_time',
                color='feature',
                title="Response Times Over Time"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Average response time by feature
            avg_by_feature = response_times_df.groupby('feature')['response_time'].mean()
            st.subheader("Average Response Time by Feature")
            st.dataframe(avg_by_feature)

    with tabs[4]:
        st.header("Error Analysis")
        
        if data['error_tracking']:
            error_data = pd.DataFrame(
                list(data['error_tracking'].items()),
                columns=['Error Type', 'Count']
            )
            
            fig = px.pie(
                error_data,
                values='Count',
                names='Error Type',
                title="Error Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)