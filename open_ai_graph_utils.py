import os
import json
import time
from typing import Dict, Any, List, Optional
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage, AIMessage, BaseMessage
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

class ConversationManager:
    """Manages conversation history and context for conversational RAG"""
    
    def __init__(self):
        self.conversation_history: List[BaseMessage] = []
        self.max_history_length = 10  # Keep last 10 exchanges
        
    def add_user_message(self, message: str):
        """Add user message to conversation history"""
        self.conversation_history.append(HumanMessage(content=message))
        self._trim_history()
        
    def add_ai_message(self, message: str):
        """Add AI response to conversation history"""
        self.conversation_history.append(AIMessage(content=message))
        self._trim_history()
        
    def _trim_history(self):
        """Keep only the most recent messages"""
        if len(self.conversation_history) > self.max_history_length * 2:
            self.conversation_history = self.conversation_history[-self.max_history_length * 2:]
            
    def get_context_summary(self) -> str:
        """Generate a summary of conversation context"""
        if not self.conversation_history:
            return ""
            
        recent_messages = self.conversation_history[-6:]  # Last 3 exchanges
        context = "Recent conversation context:\n"
        for msg in recent_messages:
            role = "User" if isinstance(msg, HumanMessage) else "Assistant"
            context += f"{role}: {msg.content[:200]}...\n"
        return context
        
    def get_history_for_llm(self) -> List[BaseMessage]:
        """Get formatted conversation history for LLM"""
        return self.conversation_history.copy()

# Global conversation manager
conversation_manager = ConversationManager()

def extract_preferences(user_input: str, conversation_context: str = "", max_retries: int = 3) -> Dict[str, Any]:
    """
    Extracts user travel preferences with conversation context
    """
    for attempt in range(max_retries):
        try:
            print(f"Attempting to extract preferences (attempt {attempt + 1}/{max_retries})")
            
            # Enhanced prompt with conversation context
            context_aware_prompt = f"""
            {conversation_context}
            
            Current user message: {user_input}
            
            Extract travel preferences from the current message and conversation context. 
            Return ONLY a valid JSON object with keys: budget, duration, interests.
            
            Rules:
            - budget: Extract numeric value as string (e.g., "1000") or empty string if not specified
            - duration: Extract number of days as string (e.g., "7") or empty string if not specified  
            - interests: Extract as array of strings (e.g., ["hiking", "culture"]) or empty array if not specified
            - Consider previous conversation context when extracting preferences
            - If user refers to "what I said earlier" or similar, use context to understand
            
            Example format:
            {{"budget": "1000", "duration": "7", "interests": ["hiking", "culture"]}}
            
            JSON Response:
            """
            
            response = llm.invoke([HumanMessage(content=context_aware_prompt)])
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
                wait_time = (attempt + 1) * 2
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
    timeout=30,
    max_retries=3,
    request_timeout=30,
    temperature=0.1
)

CONVERSATIONAL_RAG_PROMPT = PromptTemplate.from_template(
    """
    You are a helpful travel assistant with access to a knowledge base. 
    Use the conversation history and retrieved information to provide contextual, helpful responses.
    
    Conversation History:
    {conversation_history}
    
    Current Question: {question}
    
    Retrieved Information:
    {retrieved_info}
    
    Instructions:
    - Consider the conversation context when answering
    - If the user refers to previous parts of the conversation, acknowledge that context
    - Use the retrieved information to provide accurate, helpful responses
    - If the retrieved information doesn't contain relevant info, say so and offer to help differently
    - Keep responses conversational and natural
    - If this is a follow-up question, build upon previous responses
    
    Response:
    """
)

ITINERARY_PROMPT = PromptTemplate.from_template(
    """
    Create a {duration}-day itinerary for a trip to {destination}.
    Budget: ${budget}
    Interests: {interests}
    
    Previous conversation context:
    {conversation_context}
    
    Provide a detailed day-by-day breakdown with specific recommendations for:
    - Morning activities
    - Afternoon activities  
    - Evening activities
    - Recommended restaurants/dining
    - Transportation tips
    - Estimated daily costs
    
    Consider any preferences or constraints mentioned in the conversation.
    Format as a clear, structured itinerary.
    """
)

def generate_itinerary(destination: str, preferences: Dict[str, Any], 
                      conversation_context: str = "", max_retries: int = 3) -> str:
    """
    Generates a travel itinerary with conversation context
    """
    for attempt in range(max_retries):
        try:
            prompt = ITINERARY_PROMPT.format(
                duration=preferences.get('duration', '7'),
                destination=destination,
                interests=", ".join(preferences.get('interests', [])),
                budget=preferences.get('budget', '1000'),
                conversation_context=conversation_context
            )
            
            response = llm.invoke([HumanMessage(content=prompt)])
            return response.content
            
        except Exception as e:
            print(f"Error generating itinerary (attempt {attempt + 1}): {e}")
            if attempt < max_retries - 1:
                time.sleep((attempt + 1) * 2)
            else:
                return f"Unable to generate detailed itinerary for {destination} due to connection issues. Please try again later."

def conversational_rag_answer(question: str, conversation_history: List[BaseMessage]) -> str:
    """
    Answer questions using conversational RAG with history context
    """
    try:
        # Get retrieved information from vector store
        retrieved_info = answer_question(question)
        
        # Format conversation history for the prompt
        history_text = ""
        for msg in conversation_history[-6:]:  # Last 3 exchanges
            role = "User" if isinstance(msg, HumanMessage) else "Assistant"
            history_text += f"{role}: {msg.content}\n"
        
        # Use conversational RAG prompt
        prompt = CONVERSATIONAL_RAG_PROMPT.format(
            conversation_history=history_text,
            question=question,
            retrieved_info=retrieved_info
        )
        
        response = llm.invoke([HumanMessage(content=prompt)])
        return response.content
        
    except Exception as e:
        print(f"Error in conversational RAG: {e}")
        return "I'm having trouble accessing my knowledge base right now. Could you please rephrase your question or try again?"

def detect_intent(user_input: str, conversation_context: str = "") -> str:
    """
    Detect user intent to route appropriately
    """
    try:
        intent_prompt = f"""
        Conversation context:
        {conversation_context}
        
        Current user message: {user_input}
        
        Classify the user's intent into one of these categories:
        - "travel_planning": User wants to plan a new trip or modify existing plans
        - "question": User has a specific question about travel, destinations, or previous recommendations
        - "clarification": User is providing additional information or clarifying previous input
        - "greeting": User is greeting or engaging in small talk
        
        Respond with just the category name.
        """
        
        response = llm.invoke([HumanMessage(content=intent_prompt)])
        intent = response.content.strip().lower()
        
        if intent in ["travel_planning", "question", "clarification", "greeting"]:
            return intent
        else:
            return "question"  # Default fallback
            
    except Exception as e:
        print(f"Error detecting intent: {e}")
        return "question"  # Default fallback

def preference_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Extract preferences with conversation context"""
    try:
        user_input = state.get('user_input', '')
        if not user_input.strip():
            state['preferences'] = {"budget": "", "duration": "", "interests": []}
            state['missing_fields'] = ["budget", "duration", "interests"]
            return state
        
        # Add user message to conversation history
        conversation_manager.add_user_message(user_input)
        
        # Get conversation context
        context = conversation_manager.get_context_summary()
        
        # Detect intent
        intent = detect_intent(user_input, context)
        state['intent'] = intent
        
        # Extract preferences with context
        preferences = extract_preferences(user_input, context)
        missing = []

        if not preferences.get("budget"):
            missing.append("budget")
        if not preferences.get("duration"):
            missing.append("duration")
        if not preferences.get("interests") or len(preferences.get("interests", [])) == 0:
            missing.append("interests")

        state['preferences'] = preferences
        state['missing_fields'] = missing
        state['conversation_context'] = context
        
        print(f"Intent: {intent}")
        print(f"Preferences extracted: {preferences}")
        print(f"Missing fields: {missing}")
        
        return state
        
    except Exception as e:
        print(f"Critical error in preference_node: {e}")
        state['error'] = f"Failed to process your request. Please check your internet connection and try again."
        return state

def missing_info_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Handle missing information with conversational context"""
    missing = state.get("missing_fields", [])
    intent = state.get("intent", "travel_planning")
    
    if missing and intent in ["travel_planning", "clarification"]:
        questions = []

        if "budget" in missing:
            questions.append("What is your approximate budget for the trip?")
        if "duration" in missing:
            questions.append("How many days do you want your trip to last?")
        if "interests" in missing:
            questions.append("What kind of activities or experiences are you interested in (e.g., culture, adventure, relaxation, food)?")

        if questions:
            response = "I need a bit more information to help you plan your trip. " + " ".join(questions)
            state['follow_up'] = response
            conversation_manager.add_ai_message(response)
        else:
            response = "Please provide more details about your travel preferences."
            state['follow_up'] = response
            conversation_manager.add_ai_message(response)
    
    return state

def destination_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Find suitable travel destination based on preferences"""
    try:
        if 'error' in state or state.get('missing_fields'):
            return state
            
        results = find_destinations(state['preferences'])
        if not results:
            error_msg = "No destinations found matching your preferences. Please try with different criteria."
            state['error'] = error_msg
            conversation_manager.add_ai_message(error_msg)
            return state
            
        state['destination'] = results[0].metadata['name']
        print(f"Selected destination: {state['destination']}")
        return state
        
    except Exception as e:
        print(f"Error in destination_node: {e}")
        error_msg = f"Unable to find destinations at the moment. Please try again."
        state['error'] = error_msg
        conversation_manager.add_ai_message(error_msg)
        return state

def itinerary_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Generate travel itinerary with conversation context"""
    try:
        if 'error' in state or state.get('missing_fields'):
            return state
            
        context = state.get('conversation_context', '')
        itinerary = generate_itinerary(state['destination'], state['preferences'], context)
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
            
        state['weather'] = weather_tool(state['destination'])
        state['attractions'] = attractions_tool(
            state['destination'], 
            state['preferences'].get('interests', [])
        )
        return state
        
    except Exception as e:
        print(f"Error in tool_node: {e}")
        state['weather'] = ""
        state['attractions'] = ""
        return state

def rag_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Answer user questions using conversational RAG"""
    try:
        if 'error' in state:
            return state
            
        # Handle different intents
        intent = state.get('intent', 'question')
        user_input = state.get('user_input', '')
        
        if intent == "question" and user_input.strip():
            # Use conversational RAG for questions
            conversation_history = conversation_manager.get_history_for_llm()
            answer = conversational_rag_answer(user_input, conversation_history)
            state['answer'] = answer
            conversation_manager.add_ai_message(answer)
        elif intent == "greeting":
            greeting_response = "Hello! I'm here to help you plan your travels. What kind of trip are you thinking about?"
            state['answer'] = greeting_response
            conversation_manager.add_ai_message(greeting_response)
            
        return state
        
    except Exception as e:
        print(f"Error in rag_node: {e}")
        error_msg = f"Unable to answer the question at the moment: {state.get('user_input', '')}"
        state['answer'] = error_msg
        conversation_manager.add_ai_message(error_msg)
        return state

def final_output_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Prepare final output with all collected information"""
    
    # Determine response based on intent and available information
    intent = state.get('intent', 'travel_planning')
    
    if intent == "question" or intent == "greeting":
        # For questions, focus on the answer
        output = {
            "answer": state.get("answer"),
            "conversation_active": True,
            "intent": intent
        }
    elif state.get('missing_fields'):
        # Missing information case
        output = {
            "follow_up": state.get("follow_up"),
            "conversation_active": True,
            "missing_fields": state.get('missing_fields')
        }
    elif state.get('error'):
        # Error case
        output = {
            "error": state.get("error"),
            "conversation_active": True
        }
    else:
        output = {
            "destination": state.get("destination"),
            "itinerary": state.get("itinerary"),
            "weather": state.get("weather"),
            "attractions": state.get("attractions"),
            "preferences": state.get("preferences"),
            "conversation_active": True
        }
        
        if state.get("itinerary"):
            conversation_manager.add_ai_message(f"I've created a travel plan for {state.get('destination')}. Here's your itinerary: {state.get('itinerary')[:200]}...")
    
    for key, value in output.items():
        if value is not None and not isinstance(value, (str, bool, list, dict)):
            if isinstance(value, dict):
                output[key] = str(value.get('result', value))
            else:
                output[key] = str(value)
    
    state['output'] = output
    return state

graph = StateGraph(dict)

graph.add_node("preference_node", preference_node)
graph.add_node("missing_info_node", missing_info_node)
graph.add_node("destination_node", destination_node)
graph.add_node("itinerary_node", itinerary_node)
graph.add_node("tool_node", tool_node)
graph.add_node("rag_node", rag_node)
graph.add_node("final_output_node", final_output_node)

def route_after_preferences(state: Dict[str, Any]) -> str:
    """Route based on intent and missing fields"""
    intent = state.get("intent", "travel_planning")
    missing_fields = state.get("missing_fields", [])
    
    if intent in ["question", "greeting"]:
        return "rag_node"
    elif missing_fields and intent in ["travel_planning", "clarification"]:
        return "missing_info_node"
    else:
        return "destination_node"

graph.add_conditional_edges("preference_node", route_after_preferences)

graph.add_edge("missing_info_node", "final_output_node")
graph.add_edge("destination_node", "itinerary_node")
graph.add_edge("itinerary_node", "tool_node")
graph.add_edge("tool_node", "rag_node")
graph.add_edge("rag_node", "final_output_node")

graph.set_entry_point("preference_node")
graph.set_finish_point("final_output_node")

app_graph = graph.compile()

def run_conversation(user_input: str) -> Dict[str, Any]:
    """
    Run a conversational turn with the travel planning system
    """
    initial_state = {
        "user_input": user_input,
        "conversation_id": "default", 
    }
    
    result = app_graph.invoke(initial_state)
    return result.get('output', {})

def start_new_conversation():
    """Reset conversation history for a new conversation"""
    global conversation_manager
    conversation_manager = ConversationManager()
    return "New conversation started. How can I help you plan your travels?"

def get_conversation_summary():
    """Get a summary of the current conversation"""
    return conversation_manager.get_context_summary()