from langchain.docstore.document import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from pathlib import Path

vector_store = None
rag_vector_store = None

llm = ChatOpenAI(model="gpt-4o-mini")

def initialize_vector_space():
    """Initialize vector stores for destinations and RAG documents"""
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
                "and seasonal festivals like the Gion Matsuri. Perfect for culture, history, and spiritual experiences."
            )
        },
        {
            "name": "Banff",
            "description": (
                "Banff is a picturesque resort town within Banff National Park in Alberta, Canada. Known for its stunning mountain scenery, "
                "hot springs, hiking trails, and winter sports. Popular for outdoor activities such as skiing, canoeing, and wildlife viewing. "
                "Ideal for nature lovers and adventure seekers."
            )
        },
        {
            "name": "Barcelona",
            "description": (
                "Barcelona is a vibrant Spanish city famed for its art and architecture. Highlights include Gaudí's Sagrada Família, Park Güell, "
                "Mediterranean beaches, bustling food markets, and a rich nightlife. A great mix of culture, history, and leisure. "
                "Perfect for art enthusiasts and beach lovers."
            )
        },
        {
            "name": "Cape Town",
            "description": (
                "Cape Town is a coastal city in South Africa with stunning beaches, Table Mountain, vineyards, and vibrant neighborhoods. "
                "It's ideal for nature, adventure sports, and history. Offers safaris nearby and a diverse culinary scene. "
                "Great for wildlife, wine tasting, and outdoor adventures."
            )
        },
        {
            "name": "Reykjavik",
            "description": (
                "Reykjavik, the capital of Iceland, offers access to incredible landscapes, including geysers, glaciers, and volcanoes. "
                "It's known for Northern Lights viewing, hot springs like the Blue Lagoon, and quirky Nordic culture. "
                "Perfect for nature photography and unique experiences."
            )
        }
    ]
    
    destinations_faiss = Path('destinations_faiss')
    destination_docs_faiss = Path('destination_docs_faiss')  

    try:
        embedding_model = OpenAIEmbeddings()
        
        if destinations_faiss.exists() and destination_docs_faiss.exists():
            # Load existing vector stores
            vector_store = FAISS.load_local("destinations_faiss", embedding_model, allow_dangerous_deserialization=True)
            rag_vector_store = FAISS.load_local("destination_docs_faiss", embedding_model, allow_dangerous_deserialization=True)
            print("Vector stores loaded successfully from disk")
        else:
            # Create new vector stores
            vector_store = create_destination_vector_index(destination_data, return_store=True)
            rag_vector_store = create_rag_docs_index(rag_docs, return_store=True)
            print("New vector stores created successfully")
            
    except Exception as e:
        print(f"Error with vector stores: {e}")
        # Fallback: create new stores
        vector_store = create_destination_vector_index(destination_data, return_store=True)
        rag_vector_store = create_rag_docs_index(rag_docs, return_store=True)

def create_destination_vector_index(data, save_path="destinations_faiss", return_store=False):
    """Creates and saves a FAISS vector index for travel destinations"""
    embedding_model = OpenAIEmbeddings()
    documents = [Document(page_content=d["description"], metadata={"name": d["name"]}) for d in data]
    vector_store = FAISS.from_documents(documents, embedding_model)
    vector_store.save_local(save_path)
    print(f"Saved destination index to {save_path}")
    if return_store:
        return vector_store

def create_rag_docs_index(docs, save_path="destination_docs_faiss", return_store=False):
    """Creates and saves a FAISS vector index from documents for RAG"""
    embedding_model = OpenAIEmbeddings()
    documents = [Document(page_content=doc["text"], metadata={"source": doc.get("source", "")}) for doc in docs]
    vector_store = FAISS.from_documents(documents, embedding_model)
    vector_store.save_local(save_path)
    print(f"Saved RAG docs index to {save_path}")
    if return_store:
        return vector_store
    
def find_destinations(preferences):
    """Find travel destinations that match user preferences using similarity search"""
    global vector_store
    interests_str = ', '.join(preferences.get('interests', []))
    budget_str = preferences.get('budget', '')
    duration_str = preferences.get('duration', '')
    
    query = f"Budget: {budget_str}, Duration: {duration_str} days, Interests: {interests_str}"
    return vector_store.similarity_search(query, k=3)

def answer_question(question):
    """Answer user's question using RAG approach"""
    global rag_vector_store
    try:
        retriever = rag_vector_store.as_retriever(search_kwargs={"k": 3})
        rag_qa = RetrievalQA.from_chain_type(
            llm=llm, 
            retriever=retriever,
            return_source_documents=False
        )
        result = rag_qa.invoke({"query": question})
        if isinstance(result, dict):
            return result.get('result', str(result))
        return str(result)
    except Exception as e:
        print(f"Error in RAG query: {e}")
        return f"I couldn't find specific information about: {question}"