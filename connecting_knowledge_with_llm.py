import os
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings

load_dotenv()

# Step 1: Setup Google Gemini LLM
def load_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",  
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0.5
        
    )


# Step 2: Custom Prompt Template
CUSTOM_PROMPT_TEMPLATE = """
Use the pieces of information provided in the context to answer user's question.
If you don't know the answer, just say that you i am Adil fareed's Ai Assitant and please ask relevent questions. Don't try to make up an answer.
Don't provide anything out of the given context.if some one says hi or something reply gently.

Context: {context}
Question: {question}

Start the answer directly. No small talk please.
"""

def set_custom_prompt(template):
    return PromptTemplate(template=template, input_variables=["context", "question"])

# Step 3: Load FAISS vector store
DB_FAISS_PATH = "vectorstore/db_faiss"
embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

# Step 4: Create RetrievalQA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=load_llm(),
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={'k': 3}),
    return_source_documents=True,
    chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
)

# Step 5: Get query and run
user_query = input("Write Query Here: ")
response = qa_chain.invoke({'query': user_query})

# Output results
print("RESULT:", response["result"])
print("SOURCE DOCUMENTS:", response["source_documents"])
