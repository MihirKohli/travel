from pydantic import BaseModel
from typing import TypedDict


class TravelRequest(BaseModel):
    user_input: str
    question: str | None = None

class TravelResponse(BaseModel):
    destination: str | None
    itinerary: str | None
    weather: str | None
    attractions: str | None
    answer: str | None
    error: str | None


class TravelState(TypedDict):
	user_input: str
	preferences: dict
	destination: str
	itinerary: str
	weather: str
	attractions: str
	question: str
	answer: str
	error: str
	output: dict