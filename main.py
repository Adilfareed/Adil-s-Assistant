import os
import asyncio
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv, find_dotenv
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer
from langchain.embeddings.base import Embeddings

# ----------------------------------
# Load environment variables
# ----------------------------------
load_dotenv(find_dotenv())
load_dotenv()

# Fix asyncio event loop issue (Windows)
try:
    asyncio.get_event_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

# ----------------------------------
# FastAPI App Initialization
# ----------------------------------
app = FastAPI(
    title="MEDICO Chat API",
    description="A FastAPI backend for MEDICO medical assistant chatbot",
    version="1.0.0"
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace "*" with frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# FAISS DB Path
DB_FAISS_PATH = "vectorstore/db_faiss"

# ----------------------------------
# Custom Embeddings Wrapper (Local)
# ----------------------------------
class LocalHuggingFaceEmbeddings(Embeddings):
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        try:
            self.model = SentenceTransformer(model_name)
            print("✅ Local embedding model loaded successfully")
        except Exception as e:
            print("❌ Error loading local embedding model:", str(e))
            raise

    def embed_documents(self, texts):
        return self.model.encode(texts, convert_to_tensor=False).tolist()

    def embed_query(self, text):
        return self.model.encode([text], convert_to_tensor=False)[0].tolist()

# Instantiate once globally
embedding_model = LocalHuggingFaceEmbeddings()

# ----------------------------------
# Helper: Load FAISS Vector Store
# ----------------------------------
def get_vectorstore():
    try:
        return FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    except Exception as e:
        print("❌ Error loading FAISS:", str(e))
        return None

# ----------------------------------
# Helper: Custom Prompt Template
# ----------------------------------
def set_custom_prompt():
    template = """
    Use the given context to answer the user's question.
    You may also answer questions related to Adil Fareed.

    Guidelines:
    - Answers must be at least 2 lines.
    - If the user greets (e.g., hi, ok, nice), reply gently in a short conversational way.
    - Keep replies concise and conversational.
    - Never say "Further details are not provided in the context."
    - If unsure, reply: 
      "I am Adil Fareed's personal AI Assistant. Please ask relevant questions. If your question is relevant, check spelling."
    - Do not fabricate answers.
    - Start the answer directly, no small talk.

    Context: {context}
    Question: {question}
    """
    return PromptTemplate(template=template, input_variables=["context", "question"])

    

# ----------------------------------
# Input Schema for API
# ----------------------------------
class QueryRequest(BaseModel):
    query: str

# ----------------------------------
# API Route: Ask Question
# ----------------------------------
@app.post("/ask")
async def ask_medico(request: QueryRequest):
    try:
        # Load vector store
        vectorstore = get_vectorstore()
        if vectorstore is None:
            raise HTTPException(status_code=500, detail="Failed to load FAISS vector store.")

        # Setup QA Chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=ChatGoogleGenerativeAI(
                model="gemini-1.5-flash",
                google_api_key=os.getenv("GOOGLE_API_KEY"),
                temperature=0.5
            ),
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
            return_source_documents=True,
            chain_type_kwargs={'prompt': set_custom_prompt()}
        )

        # Get response
        response = qa_chain.invoke({"query": request.query})
        result = response.get("result", "I'm sorry, I couldn't generate a response.")
        sources = [doc.metadata.get("source", "Unknown") for doc in response.get("source_documents", [])]

        return {
            "query": request.query,
            "answer": result,
            "sources": sources
        }

    except Exception as e:
        import traceback
        print("❌ Error in /ask:", str(e))
        traceback.print_exc()  # <-- full stack trace in terminal
        raise HTTPException(status_code=500, detail=str(e))

# ----------------------------------
# Root Route
# ----------------------------------
@app.get("/")
def home():
    return {"message": "Welcome to Adil's AI Assistant Chat API! Use POST /ask to ask questions."}
