import os
import json
import time
from typing import Dict, Any
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage
from langchain_openai import ChatOpenAI
from langchain.tools import Tool
from langgraph.graph import StateGraph
from dotenv import load_dotenv
from tools import weather_tool, attractions_tool
from vector_stores import find_destinations, answer_question
from schema import TravelState

# Load environment variables
load_dotenv()

# Validate OpenAI API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

def extract_preferences(user_input: str, max_retries: int = 3) -> Dict[str, Any]:
    """
    Extracts user travel preferences with retry logic and better error handling
    """
    for attempt in range(max_retries):
        try:
            print(f"Attempting to extract preferences (attempt {attempt + 1}/{max_retries})")
            
            response = llm.invoke([HumanMessage(content=PREF_PROMPT.format(input=user_input))])
            content = response.content.strip()
            
            # Clean up the response
            if content.startswith('```json'):
                content = content.replace('```json', '').replace('```', '').strip()
            elif content.startswith('```'):
                content = content.replace('```', '').strip()
            
            # Parse JSON
            preferences = json.loads(content)
            print(f"Successfully extracted preferences: {preferences}")
            return preferences
            
        except json.JSONDecodeError as e:
            print(f"JSON parsing error on attempt {attempt + 1}: {e}")
            if attempt == max_retries - 1:
                print("Using fallback preferences due to JSON parsing failure")
                return {"budget": "", "duration": "", "interests": []}
            
        except Exception as e:
            print(f"Connection/API error on attempt {attempt + 1}: {e}")
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 2  # Exponential backoff
                print(f"Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
            else:
                print("All attempts failed, using fallback preferences")
                return {"budget": "", "duration": "", "interests": []}
    
    return {"budget": "", "duration": "", "interests": []}

# Initialize LLM with better configuration
llm = ChatOpenAI(
    model="gpt-4o-mini",
    api_key=OPENAI_API_KEY,
    timeout=30,  # 30 second timeout
    max_retries=3,  # Retry failed requests
    request_timeout=30,  # Request timeout
    temperature=0.1  # Lower temperature for more consistent results
)

PREF_PROMPT = PromptTemplate.from_template(
    """
    Extract travel preferences from the user message. Return ONLY a valid JSON object with keys: budget, duration, interests.
    
    Rules:
    - budget: Extract numeric value as string (e.g., "1000") or empty string if not specified
    - duration: Extract number of days as string (e.g., "7") or empty string if not specified  
    - interests: Extract as array of strings (e.g., ["hiking", "culture"]) or empty array if not specified
    
    Example format:
    {{"budget": "1000", "duration": "7", "interests": ["hiking", "culture"]}}
    
    If information is missing, use empty string for budget/duration and empty array for interests.
    
    Message: {input}
    
    JSON Response:
    """
)

ITINERARY_PROMPT = PromptTemplate.from_template(
    """
    Create a {duration}-day itinerary for a trip to {destination}.
    Budget: ${budget}
    Interests: {interests}
    
    Provide a detailed day-by-day breakdown with specific recommendations for:
    - Morning activities
    - Afternoon activities  
    - Evening activities
    - Recommended restaurants/dining
    - Transportation tips
    - Estimated daily costs
    
    Format as a clear, structured itinerary.
    """
)

def generate_itinerary(destination: str, preferences: Dict[str, Any], max_retries: int = 3) -> str:
    """
    Generates a travel itinerary with retry logic
    """
    for attempt in range(max_retries):
        try:
            prompt = ITINERARY_PROMPT.format(
                duration=preferences.get('duration', '7'),
                destination=destination,
                interests=", ".join(preferences.get('interests', [])),
                budget=preferences.get('budget', '1000')
            )
            
            response = llm.invoke([HumanMessage(content=prompt)])
            return response.content
            
        except Exception as e:
            print(f"Error generating itinerary (attempt {attempt + 1}): {e}")
            if attempt < max_retries - 1:
                time.sleep((attempt + 1) * 2)
            else:
                return f"Unable to generate detailed itinerary for {destination} due to connection issues. Please try again later."

def preference_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Extract preferences and identify missing fields with enhanced error handling"""
    try:
        user_input = state.get('user_input', '')
        if not user_input.strip():
            state['preferences'] = {"budget": "", "duration": "", "interests": []}
            state['missing_fields'] = ["budget", "duration", "interests"]
            return state
            
        preferences = extract_preferences(user_input)
        missing = []

        if not preferences.get("budget"):
            missing.append("budget")
        if not preferences.get("duration"):
            missing.append("duration")
        if not preferences.get("interests") or len(preferences.get("interests", [])) == 0:
            missing.append("interests")

        state['preferences'] = preferences
        state['missing_fields'] = missing
        
        print(f"Preferences extracted: {preferences}")
        print(f"Missing fields: {missing}")
        
        return state
        
    except Exception as e:
        print(f"Critical error in preference_node: {e}")
        state['error'] = f"Failed to process your request. Please check your internet connection and try again."
        return state

def missing_info_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Handle missing information by generating follow-up questions"""
    missing = state.get("missing_fields", [])
    questions = []

    if "budget" in missing:
        questions.append("What is your approximate budget for the trip?")
    if "duration" in missing:
        questions.append("How many days do you want your trip to last?")
    if "interests" in missing:
        questions.append("What kind of activities or experiences are you interested in (e.g., culture, adventure, relaxation, food)?")

    if questions:
        state['follow_up'] = " ".join(questions)
    else:
        state['follow_up'] = "Please provide more details about your travel preferences."
        
    return state

def destination_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Find suitable travel destination based on user preferences"""
    try:
        if 'error' in state or state.get('missing_fields'):
            return state
            
        results = find_destinations(state['preferences'])
        if not results:
            state['error'] = "No destinations found matching your preferences. Please try with different criteria."
            return state
            
        state['destination'] = results[0].metadata['name']
        print(f"Selected destination: {state['destination']}")
        return state
        
    except Exception as e:
        print(f"Error in destination_node: {e}")
        state['error'] = f"Unable to find destinations at the moment. Please try again."
        return state

def itinerary_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Generate travel itinerary based on destination and preferences"""
    try:
        if 'error' in state or state.get('missing_fields'):
            return state
            
        itinerary = generate_itinerary(state['destination'], state['preferences'])
        state['itinerary'] = itinerary
        return state
        
    except Exception as e:
        print(f"Error in itinerary_node: {e}")
        state['error'] = f"Unable to generate itinerary at the moment. Please try again."
        return state

def tool_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Fetch additional information like weather and attractions"""
    try:
        if 'error' in state or state.get('missing_fields'):
            return state
            
        # These functions have their own error handling and return empty strings on failure
        state['weather'] = weather_tool(state['destination'])
        state['attractions'] = attractions_tool(
            state['destination'], 
            state['preferences'].get('interests', [])
        )
        return state
        
    except Exception as e:
        print(f"Error in tool_node: {e}")
        # Don't set error here as weather/attractions are optional
        state['weather'] = ""
        state['attractions'] = ""
        return state

def rag_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Answer user question using RAG if question is provided"""
    try:
        if 'error' in state:
            return state
            
        question = state.get('question')
        if question and question.strip():
            answer = answer_question(question)
            state['answer'] = answer
            
        return state
        
    except Exception as e:
        print(f"Error in rag_node: {e}")
        # Don't set error for RAG failures as it's optional
        state['answer'] = f"Unable to answer the question at the moment: {state.get('question', '')}"
        return state

def final_output_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Prepare final output with all collected information"""
    output = {
        "destination": state.get("destination"),
        "itinerary": state.get("itinerary"),
        "weather": state.get("weather"),
        "attractions": state.get("attractions"),
        "answer": state.get("answer"),
        "error": state.get("error"),
        "follow_up": state.get("follow_up")
    }
    
    # Clean up output values
    for key, value in output.items():
        if value is not None and not isinstance(value, str):
            if isinstance(value, dict):
                output[key] = str(value.get('result', value))
            else:
                output[key] = str(value)
    
    state['output'] = output
    return state

# Create the workflow graph
graph = StateGraph(dict)

# Add nodes
graph.add_node("preference_node", preference_node)
graph.add_node("missing_info_node", missing_info_node)
graph.add_node("destination_node", destination_node)
graph.add_node("itinerary_node", itinerary_node)
graph.add_node("tool_node", tool_node)
graph.add_node("rag_node", rag_node)
graph.add_node("final_output_node", final_output_node)

# Define conditional routing function
def route_after_preferences(state: Dict[str, Any]) -> str:
    """Route based on whether we have missing fields"""
    if state.get("missing_fields"):
        return "missing_info_node"
    else:
        return "destination_node"

# Add conditional edges
graph.add_conditional_edges("preference_node", route_after_preferences)

# Add regular edges
graph.add_edge("missing_info_node", "final_output_node")
graph.add_edge("destination_node", "itinerary_node")
graph.add_edge("itinerary_node", "tool_node")
graph.add_edge("tool_node", "rag_node")
graph.add_edge("rag_node", "final_output_node")

# Set entry and finish points
graph.set_entry_point("preference_node")
graph.set_finish_point("final_output_node")

# Compile the graph
app_graph = graph.compile()
