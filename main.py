from fastapi import FastAPI
from schema import TravelRequest, TravelResponse
from contextlib import asynccontextmanager
from vector_stores import initialize_vector_space
from open_ai_graph_utils import app_graph

@asynccontextmanager
async def lifespan(app: FastAPI):
    # this will initialize local vector storage on startup
    initialize_vector_space()  
    yield

app = FastAPI(lifespan=lifespan)

@app.post("/plan", response_model=TravelResponse)
async def plan_trip(req: TravelRequest):
    state = {"user_input": req.user_input}
    final_state = app_graph.invoke(state)
    return final_state['output']