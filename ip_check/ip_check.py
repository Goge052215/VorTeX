import logging
import socket
import requests
from requests.exceptions import RequestException
from urllib.request import urlopen

logger = logging.getLogger(__name__)

class IPCheck:
    @staticmethod
    def ip_check():
        """Simple check if internet connection is available."""
        try:
            socket.create_connection(("8.8.8.8", 53), timeout=3)
            
            try:
                response = urlopen('https://api64.ipify.org?format=json', timeout=3)
                return True
            except:
                pass

            # Third try: Alternative IP service
            try:
                response = requests.get('http://ip-api.com/json/', timeout=3)
                if response.status_code == 200:
                    return True
            except:
                pass

            return True

        except Exception as e:
            logger.warning(f"Connection check failed: {e}")
            return False
