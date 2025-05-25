from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage
from langchain_openai import ChatOpenAI
from langchain.tools import Tool
from langgraph.graph import StateGraph
import json
import os
from dotenv import load_dotenv
from tools import weather_tool, attractions_tool
from vector_stores import find_destinations, answer_question
from schema import TravelState

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

def extract_preferences(user_input):
	"""
    Extracts user travel preferences from a given input string using a language model (LLM).
    
    Args:
        user_input (str): A string containing the user's natural language input (e.g., "I want a 1-week beach vacation under $1500").
    
    Returns:
        dict: A dictionary (parsed from JSON) containing preferences such as budget, duration, and interests.
              If the response can't be parsed, returns a default preference dictionary.
    """
	try:
		response = llm.invoke([HumanMessage(content=PREF_PROMPT.format(input=user_input))])
		content = response.content.strip()
		
		if content.startswith('```json'):
			content = content.replace('```json', '').replace('```', '').strip()
		elif content.startswith('```'):
			content = content.replace('```', '').strip()
		
		try:
			return json.loads(content)
		except json.JSONDecodeError:
			return {
				"budget": "1000",
				"duration": "7", 
				"interests": ["sightseeing", "culture"]
			}
	except Exception as e:
		print(f"Error extracting preferences: {e}")
		return {
			"budget": "1000",
			"duration": "7",
			"interests": ["sightseeing", "culture"]
		}
	

llm = ChatOpenAI(model="gpt-4o-mini")

PREF_PROMPT = PromptTemplate.from_template(
	"""
	Extract travel preferences from the user message. Return ONLY a valid JSON object with keys: budget, duration, interests.
	
	Example format:
	{{"budget": "1000", "duration": "7", "interests": ["hiking", "culture"]}}
	
	Message: {input}
	
	JSON Response:
	"""
)

tools = [
	Tool(name="WeatherTool", func=weather_tool, description="Provides weather info."),
	Tool(name="AttractionTool", func=attractions_tool, description="Suggests attractions.")
]

ITINERARY_PROMPT = PromptTemplate.from_template(
	"""
	Create a {duration}-day itinerary for a trip to {destination}.
	Interests: {interests}. Budget: ${budget}.
	Include local attractions and balanced activities.
	"""
)

def generate_itinerary(destination, preferences):
	"""
    Generates a travel itinerary using a language model (LLM) based on user preferences and destination.
    
    Args:
        destination (str): The travel destination (e.g., "Paris").
        preferences (dict): A dictionary containing user preferences such as:
            - duration (str): Number of days for the trip
            - interests (list): List of interest topics (e.g., ["culture", "food"])
            - budget (str): Approximate budget amount in dollars or preferred currency

    Returns:
        str: A generated travel itinerary in plain text, created by the LLM.
    """
	prompt = ITINERARY_PROMPT.format(
		duration=preferences['duration'],
		destination=destination,
		interests=", ".join(preferences['interests']),
		budget=preferences['budget']
	)
	return llm.invoke([HumanMessage(content=prompt)]).content




def preference_node(state):
	"""
    Extracts user preferences from their input and updates the shared state dictionary.
    
    Args:
        state (dict): A shared state object that must include:
            - 'user_input' (str): The raw user input to process.
    
    Updates:
        - Adds 'preferences' (dict) to the state if extraction is successful.
        - Adds 'error' (str) to the state if extraction fails.
    
    Returns:
        dict: The updated state dictionary, containing either preferences or an error message.
    """
	try:
		state['preferences'] = extract_preferences(state['user_input'])
		return state
	except Exception as e:
		print(f"Error in preference_node: {e}")
		state['error'] = f"Failed to extract preferences: {str(e)}"
		return state

def destination_node(state):
	"""
    Finds a suitable travel destination based on user preferences and updates the state.
    
    Args:
        state (dict): A shared state object that must include:
            - 'preferences' (dict): User preferences like budget, interests, and duration.
            - May already include an 'error' key if a previous step failed.

    Updates:
        - Adds 'destination' (str) to the state if a match is found.
        - Adds or preserves 'error' (str) in the state if something fails.

    Returns:
        dict: The updated state dictionary, containing either a destination or an error message.
    """
	try:
		if 'error' in state:
			return state
		results = find_destinations(state['preferences'])
		if not results:
			state['error'] = "No destinations found matching your preferences"
			return state
		state['destination'] = results[0].metadata['name']
		return state
	except Exception as e:
		print(f"Error in destination_node: {e}")
		state['error'] = f"Failed to find destinations: {str(e)}"
		return state

def itinerary_node(state):
	"""
    Generates a travel itinerary based on the chosen destination and user preferences, updating the state.
    
    Args:
        state (dict): A shared state dictionary that should include:
            - 'destination' (str): The selected travel destination.
            - 'preferences' (dict): User preferences like budget, interests, and duration.
            - May also contain an 'error' key if a previous step failed.

    Updates:
        - Adds 'itinerary' (str) to the state with the generated travel plan.
        - Adds or preserves 'error' (str) if itinerary generation fails.

    Returns:
        dict: The updated state dictionary containing either the itinerary or an error message.
    """
	try:
		if 'error' in state:
			return state
		state['itinerary'] = generate_itinerary(state['destination'], state['preferences'])
		return state
	except Exception as e:
		print(f"Error in itinerary_node: {e}")
		state['error'] = f"Failed to generate itinerary: {str(e)}"
		return state

def tool_node(state):
	"""
    Fetches additional information like weather and attractions based on the destination and user interests,
    then updates the state dictionary.
    
    Args:
        state (dict): A shared state object that should include:
            - 'destination' (str): The selected travel destination.
            - 'preferences' (dict): User preferences, including 'interests' (list).
            - May already contain an 'error' key if previous steps failed.
    
    Updates:
        - Adds 'weather' (any): Weather information for the destination.
        - Adds 'attractions' (any): Attractions related to the user's interests at the destination.
        - Adds or preserves 'error' (str) if fetching info fails.
    
    Returns:
        dict: The updated state dictionary containing additional info or an error message.
    """
	try:
		if 'error' in state:
			return state
		state['weather'] = weather_tool(state['destination'])
		state['attractions'] = attractions_tool(state['destination'], state['preferences']['interests'])
		return state
	except Exception as e:
		print(f"Error in tool_node: {e}")
		state['error'] = f"Failed to get additional info: {str(e)}"
		return state

def rag_node(state):
	"""
    Answers a user question using a retrieval-augmented generation (RAG) method, updating the state with the answer.
    
    Args:
        state (dict): A shared state dictionary that may include:
            - 'question' (str): The user's question to be answered.
            - May also include an 'error' key if previous steps failed.
    
    Updates:
        - Adds 'answer' (str) to the state if the question is present and answered.
        - Adds or preserves 'error' (str) if answering the question fails.
    
    Returns:
        dict: The updated state dictionary containing the answer or an error message.
    """
	try:
		if 'error' in state:
			return state
		if 'question' in state and state['question']:
			state['answer'] = answer_question(state['question'])
		return state
	except Exception as e:
		print(f"Error in rag_node: {e}")
		state['error'] = f"Failed to answer question: {str(e)}"
		return state

def error_handler_node(state):
	state['error'] = "Sorry, something went wrong. Please try again."
	return state

def final_output_node(state):
	# extract in json manner
	output = {
		"destination": state.get("destination"),
		"itinerary": state.get("itinerary"),
		"weather": state.get("weather"),
		"attractions": state.get("attractions"),
		"answer": state.get("answer"),
		"error": state.get("error")
	}
	
	for key, value in output.items():
		if value is not None and not isinstance(value, str):
			if isinstance(value, dict):
				output[key] = str(value.get('result', value))
			else:
				output[key] = str(value)
	
	state['output'] = output
	return state

graph = StateGraph(TravelState)

# Adding nodes (steps) to the graph, each representing a part of the travel planning workflow
graph.add_node("preference_node", preference_node)
graph.add_node("destination_node", destination_node)
graph.add_node("itinerary_node", itinerary_node)
graph.add_node("tool_node", tool_node)
graph.add_node("rag_node", rag_node)
graph.add_node("error_handler_node", error_handler_node)
graph.add_node("final_output_node", final_output_node)

# Defining the flow between nodes in the graph
graph.add_edge("preference_node", "destination_node")
graph.add_edge("destination_node", "itinerary_node")
graph.add_edge("itinerary_node", "tool_node")
graph.add_edge("tool_node", "rag_node")
graph.add_edge("rag_node", "final_output_node")

# Setting the starting point of the graph processing
graph.set_entry_point("preference_node")

# Setting the endpoint where processing finishes
graph.set_finish_point("final_output_node")

# Compile the graph to prepare it for execution
app_graph = graph.compile()
