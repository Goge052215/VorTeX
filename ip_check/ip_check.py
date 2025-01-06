import logging

logger = logging.getLogger(__name__)

class IPCheck:
    def ip_check():
        """Check if the user is in mainland China based on IP address."""
        try:
            try:
                import requests
            except ImportError:
                logger.warning("requests module not found. Installing...")
                import subprocess
                try:
                    subprocess.check_call(['pip', 'install', 'requests'])
                    import requests
                except Exception as e:
                    logger.error(f"Failed to install requests: {e}")
                    return False

                # Use a reliable IP geolocation API with rate limit handling
            response = requests.get('https://ipapi.co/json/', timeout=5)
            if response.status_code == 429:
                logger.warning("IP API rate limit reached, using alternative method")
                # Fall back to ping test
                return False
            elif response.status_code == 200:
                data = response.json()
                country_code = data.get('country_code')
                country_name = data.get('country_name')
                region = data.get('region')
                city = data.get('city')
                ip = data.get('ip')

                # Log detailed location info
                logger.info(f"IP Address: {ip}")
                logger.info(f"Location: {city}, {region}, {country_name} ({country_code})")

                # Check if in mainland China
                if country_code == 'CN' and region not in ['Hong Kong', 'Macau', 'Taiwan']:
                    logger.info("User location: Mainland China")
                    return True
                else:
                    logger.info("User location: Not in Mainland China")
                    return False
        except Exception as e:
            logger.error(f"Error checking location: {str(e)}")
            # If we can't determine location, fall back to ping test
            return False
