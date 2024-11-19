import random
from typing import Dict
import requests

class ProxyRotator:
    def __init__(self):
        self.proxies = []
        self.update_proxy_list()
    
    def update_proxy_list(self):
        """Update list of free proxies"""
        try:
            # Get free proxies from public API
            response = requests.get('https://api.proxyscrape.com/v2/?request=displayproxies&protocol=http&timeout=10000&country=all&ssl=all&anonymity=all')
            if response.status_code == 200:
                self.proxies = [f"http://{proxy}" for proxy in response.text.split('\n') if proxy.strip()]
        except Exception as e:
            logger.error(f"Error updating proxy list: {e}")
    
    def get_proxy(self) -> Dict[str, str]:
        """Get a random proxy from the list"""
        if not self.proxies:
            self.update_proxy_list()
        if self.proxies:
            proxy = random.choice(self.proxies)
            return {
                'http': proxy,
                'https': proxy
            }
        return None

# Usage in get_transcript function
proxy_rotator = ProxyRotator()

def get_transcript(video_id):
    proxies = proxy_rotator.get_proxy()
    if not proxies:
        st.warning("No proxies available. Trying without proxy...")
        return get_transcript_without_proxy(video_id)
        
    try:
        import httpx
        proxies_transport = httpx.HTTPTransport(proxy=proxies['https'])
        client = httpx.Client(transport=proxies_transport)
        YouTubeTranscriptApi.http_client = client
        
        # Rest of your transcript fetching code...
    
    except Exception as e:
        logger.error(f"Proxy error: {e}")
        # Try next proxy
        return get_transcript(video_id)