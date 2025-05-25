
from langchain.docstore.document import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from pathlib import Path

vector_store = None
rag_vector_store = None

llm = ChatOpenAI(model="gpt-4o-mini")

def initialize_vector_space():
    global vector_store, rag_vector_store
    rag_docs = [
        {
            "text": "Hiking in Banff is best from June to September when trails are clear of snow and weather is mild.",
            "source": "banff.ca/hiking"
        },
        {
            "text": "Kyoto's cherry blossom season typically peaks in the first week of April. Popular spots include Maruyama Park and the Philosopher's Path.",
            "source": "kyototravelguide.jp/sakura"
        },
        {
            "text": "Barcelona has a Mediterranean climate. The best time to visit is from April to June and September to November to avoid summer crowds.",
            "source": "barcelona.info/weather"
        },
        {
            "text": "Cape Town experiences summer from November to March, ideal for beach trips. For whale watching, visit between July and October.",
            "source": "capetown.com/seasons"
        },
        {
            "text": "The Northern Lights are most visible in Reykjavik from September to April. Avoid city light pollution for best viewing.",
            "source": "visiticeland.com/northernlights"
        }
    ]
    destination_data = [
        {
            "name": "Kyoto",
            "description": (
                "Kyoto is a historical city in Japan known for its classical Buddhist temples, gardens, imperial palaces, "
                "Shinto shrines, and traditional wooden houses. It offers cultural experiences like tea ceremonies, geisha shows, "
                "and seasonal festivals like the Gion Matsuri."
            )
        },
        {
            "name": "Banff",
            "description": (
                "Banff is a picturesque resort town within Banff National Park in Alberta, Canada. Known for its stunning mountain scenery, "
                "hot springs, hiking trails, and winter sports. Popular for outdoor activities such as skiing, canoeing, and wildlife viewing."
            )
        },
        {
            "name": "Barcelona",
            "description": (
                "Barcelona is a vibrant Spanish city famed for its art and architecture. Highlights include Gaudí's Sagrada Família, Park Güell, "
                "Mediterranean beaches, bustling food markets, and a rich nightlife. A great mix of culture, history, and leisure."
            )
        },
        {
            "name": "Cape Town",
            "description": (
                "Cape Town is a coastal city in South Africa with stunning beaches, Table Mountain, vineyards, and vibrant neighborhoods. "
                "It's ideal for nature, adventure sports, and history. Offers safaris nearby and a diverse culinary scene."
            )
        },
        {
            "name": "Reykjavik",
            "description": (
                "Reykjavik, the capital of Iceland, offers access to incredible landscapes, including geysers, glaciers, and volcanoes. "
                "It's known for Northern Lights viewing, hot springs like the Blue Lagoon, and quirky Nordic culture."
            )
        }
    ]
    destinations_faiss = Path('destinations_faiss')
    destination_docs_faiss = Path('destination_docs_faiss')  

    if not destinations_faiss.exists() or not destination_docs_faiss.exists():
        # check if local database does not exist then create one
        create_rag_docs_index(rag_docs)
        create_destination_vector_index(destination_data)

    embedding_model = OpenAIEmbeddings()
    try:
        # load from local storage into global variable for accessibility
        vector_store = FAISS.load_local("destinations_faiss", embedding_model, allow_dangerous_deserialization=True)
        rag_vector_store = FAISS.load_local("destination_docs_faiss", embedding_model, allow_dangerous_deserialization=True)
        print("Vector stores loaded successfully")
    except Exception as e:
        print(f"Error loading vector stores: {e}")
        vector_store = create_destination_vector_index(destination_data, return_store=True)
        rag_vector_store = create_rag_docs_index(rag_docs, return_store=True)
    


def create_destination_vector_index(data, save_path="destinations_faiss", return_store=False):
    # be default openai-ada is used SOTA model for embedding
    """
    Creates and saves a FAISS vector index for travel destinations using their descriptions.
    
    Args:
        data (list of dict): List of destination data, each containing:
            - 'name' (str): Name of the destination
            - 'description' (str): Description text of the destination
        save_path (str, optional): Path to save the FAISS index locally. Defaults to "destinations_faiss".
        return_store (bool, optional): If True, returns the created vector store object. Defaults to False.
    
    Returns:
        FAISS vector store (optional): The created vector store if return_store is True, otherwise None.
    """
    embedding_model = OpenAIEmbeddings()
    documents = [Document(page_content=d["description"], metadata={"name": d["name"]}) for d in data]
    vector_store = FAISS.from_documents(documents, embedding_model)
    vector_store.save_local(save_path)
    print(f"Saved destination index to {save_path}")
    if return_store:
        return vector_store

def create_rag_docs_index(docs, save_path="destination_docs_faiss", return_store=False):
    # be default openai-ada is used SOTA model for embedding
    """
    Creates and saves a FAISS vector index from a list of documents for retrieval-augmented generation (RAG).
    
    Args:
        docs (list of dict): List of documents, each with at least a "text" field and optionally "source" metadata.
        save_path (str, optional): Path to save the FAISS index locally. Defaults to "destination_docs_faiss".
        return_store (bool, optional): If True, returns the created vector store object. Defaults to False.
    
    Returns:
        FAISS vector store (optional): The created vector store if return_store is True, otherwise None.
    """
    embedding_model = OpenAIEmbeddings()
    documents = [Document(page_content=doc["text"], metadata={"source": doc.get("source", "")}) for doc in docs]
    vector_store = FAISS.from_documents(documents, embedding_model)
    vector_store.save_local(save_path)
    print(f"Saved RAG docs index to {save_path}")
    if return_store:
        return vector_store
    
def find_destinations(preferences):
    # similarity search for user preference
    """
    Finds travel destinations that match user preferences using similarity search.
    
    Args:
        preferences (dict): User preferences including:
            - 'budget' (str): Budget for the trip
            - 'duration' (str): Trip duration in days
            - 'interests' (list): List of user interests (e.g., ["sightseeing", "culture"])
    
    Returns:
        list: A list of top matching destination objects from the vector store (up to 3).
    """
    global vector_store
    query = f"{preferences['budget']} {preferences['duration']} {', '.join(preferences['interests'])}"
    return vector_store.similarity_search(query, k=3)

def answer_question(question):
    # answer user query using langgraph rag
    """
    Answers a user's question using a retrieval-augmented generation (RAG) approach with LangGraph.
    
    Args:
        question (str): The question asked by the user.
    
    Returns:
        str: The answer generated by the RAG system, converted to a string.
    """
    global rag_vector_store
    retriever = rag_vector_store.as_retriever()
    rag_qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    result = rag_qa.invoke(question)
    if isinstance(result, dict):
        return result.get('result', str(result))
    return str(result)