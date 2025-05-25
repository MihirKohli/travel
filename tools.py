import requests
import os
from dotenv import load_dotenv
load_dotenv()

OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
GOOGLE_PLACES_API_KEY = os.getenv("GOOGLE_PLACES_API_KEY")


DEFAULT_WEATHER = {
    "Paris": {"description": "partly cloudy", "temp": 18},
    "New York": {"description": "sunny", "temp": 22},
    "Tokyo": {"description": "light rain", "temp": 20},
    "Rome": {"description": "clear sky", "temp": 24},
    "London": {"description": "overcast clouds", "temp": 16}
}

def weather_tool(destination: str) -> str:
    # added default weather if no api key been provided
    """
    Fetches the current weather for a given destination using the OpenWeather API.
    If the API key is missing or an error occurs, returns default weather info if available.
    
    Args:
        destination (str): The name of the location to get weather information for.
    
    Returns:
        str: A human-readable weather summary for the destination.
    """
    if not OPENWEATHER_API_KEY:
        default = DEFAULT_WEATHER.get(destination)
        if default:
            return f"Expected weather in {destination} is {default['description']} with average {default['temp']}°C."
        else:
            return f"Weather data for {destination} is currently unavailable."

    url = (
        f"http://api.openweathermap.org/data/2.5/weather"
        f"?q={destination}&appid={OPENWEATHER_API_KEY}&units=metric"
    )
    try:
        resp = requests.get(url)
        if resp.status_code != 200:
            raise ValueError("Bad response")
        data = resp.json()
        temp = data['main']['temp']
        description = data['weather'][0]['description']
        return f"Expected weather in {destination} is {description} with average {temp}°C."
    except Exception:
        default = DEFAULT_WEATHER.get(destination)
        if default:
            return f"Expected weather in {destination} is {default['description']} with average {default['temp']}°C."
        return f"Could not fetch weather for {destination}."



DEFAULT_ATTRACTIONS = {
    "Paris": ["Eiffel Tower", "Louvre Museum", "Notre-Dame Cathedral"],
    "New York": ["Statue of Liberty", "Central Park", "Times Square"],
    "Tokyo": ["Senso-ji Temple", "Shibuya Crossing", "Tokyo Tower"],
    "Rome": ["Colosseum", "Trevi Fountain", "Pantheon"],
    "London": ["Big Ben", "Tower Bridge", "British Museum"]
}

def attractions_tool(destination: str, interests: list[str]) -> str:
    """
    Fetches top attractions in a destination based on user interests using the Google Places API.
    If the API key is missing or no results found, returns default attractions if available.
    
    Args:
        destination (str): The location to find attractions in.
        interests (list[str]): List of user interests (e.g., ["museums", "parks"]).
    
    Returns:
        str: A human-readable summary of top attractions matching the interests.
    """
    # added default attactions if no api key been provided
    attractions = []

    if not GOOGLE_PLACES_API_KEY:
        default = DEFAULT_ATTRACTIONS.get(destination)
        if default:
            return f"Top attractions in {destination}: {', '.join(default)}."
        else:
            return f"Attractions data for {destination} is currently unavailable."

    for interest in interests:
        query = f"{interest} in {destination}"
        url = (
            f"https://maps.googleapis.com/maps/api/place/textsearch/json?"
            f"query={query}&key={GOOGLE_PLACES_API_KEY}"
        )
        try:
            resp = requests.get(url)
            if resp.status_code != 200:
                continue

            results = resp.json().get("results", [])
            if results:
                name = results[0]["name"]
                attractions.append(name)
        except Exception:
            continue

    if not attractions:
        default = DEFAULT_ATTRACTIONS.get(destination)
        if default:
            return f"Top attractions in {destination}: {', '.join(default)}."
        return f"No attractions found for interests in {destination}."

    return f"Top attractions in {destination} for {', '.join(interests)}: {', '.join(attractions)}."
