# Travel Itinerary Assistant

A flexible travel itinerary assistant that combines large language models (LLMs), vector similarity search, and external APIs to craft personalized travel plans based on user preferences. It also uses retrieval-augmented generation (RAG) to answer questions and provide rich travel insights.

---

## Features

- **Preference Extraction:** Parses user input into structured preferences like budget, trip duration, and interests using an LLM.  
- **Destination Recommendation:** Suggests destinations matching user preferences via vector similarity search.  
- **Itinerary Generation:** Creates customized daily plans based on chosen destination and user interests.  
- **API Integrations:** Retrieves up-to-date weather info and popular attractions for the destination.  
- **Retrieval-Augmented QA:** Provides accurate answers to user questions by leveraging a knowledge base with RAG.  
- **Error Resilience:** Handles errors gracefully at every step, defaulting to fallback data when necessary.

---

## Installation & Setup

### Requirements

- Python 3.11 or higher  
- Docker and Docker Compose installed  

### Steps

1. Rename `local.env` to `.env` in the project root.  
2. Edit the `.env` file to add your API keys and environment variables (e.g., `OPENAI_API_KEY`, `OPENWEATHER_API_KEY`, `GOOGLE_PLACES_API_KEY`).  
3. Build and start the Docker containers:

```bash
docker-compose build --no-cache
docker-compose up
