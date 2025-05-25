from pydantic import BaseModel
from typing import Optional, List, Dict, Any, TypedDict

class TravelRequest(BaseModel):
    user_input: str
    question: Optional[str] = None

class TravelResponse(BaseModel):
    destination: Optional[str] = None
    itinerary: Optional[str] = None
    weather: Optional[str] = None
    attractions: Optional[str] = None
    answer: Optional[str] = None
    error: Optional[str] = None
    follow_up: Optional[str] = None  # Added missing field

class TravelState(TypedDict, total=False):
    """
    State dictionary for travel planning workflow.
    Contains user input, preferences, destination, itinerary, and other travel data.
    """
    user_input: str
    question: Optional[str]
    preferences: Optional[Dict[str, Any]]
    missing_fields: Optional[List[str]]
    follow_up: Optional[str]
    destination: Optional[str]
    itinerary: Optional[str]
    weather: Optional[Any]
    attractions: Optional[Any]
    answer: Optional[str]
    error: Optional[str]
    output: Optional[Dict[str, Any]]