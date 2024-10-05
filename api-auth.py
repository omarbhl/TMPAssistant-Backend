import os
import shutil
from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from passlib.context import CryptContext
from jose import JWTError, jwt
from datetime import datetime, timedelta
from typing import Optional

from sqlalchemy import create_engine, Column, Integer, String, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.future import select

from langchain_community.document_loaders.pdf import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.vectorstores.chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama

from prompt_enhance import enhance_prompt

# Configuration
SECRET_KEY = "123456789"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Environment variables
CHROMA_PATH = "chroma/"
DATA_PATH = "data/"

# Create directories if they don't exist
os.makedirs(CHROMA_PATH, exist_ok=True)
os.makedirs(DATA_PATH, exist_ok=True)

# SQLite URL
SQLALCHEMY_DATABASE_URL = "sqlite:///./users.db"

# Create SQLAlchemy engine
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})

# SessionLocal class used for creating sessions
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for our models
Base = declarative_base()

# User model
class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    email = Column(String, unique=True, index=True)
    full_name = Column(String)
    hashed_password = Column(String)
    disabled = Column(Boolean, default=False)

# Create database tables
Base.metadata.create_all(bind=engine)

# FastAPI instance
app = FastAPI()

# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# OAuth2 scheme
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# CORS configuration
origins = [
    "http://localhost",
    "http://localhost:3000",
    "http://localhost:11434",
    # Add more origins as needed
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Pydantic models
class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: Optional[str] = None

class UserBase(BaseModel):
    username: str
    email: str
    full_name: Optional[str] = None
    disabled: Optional[bool] = None

class UserCreate(UserBase):
    password: str

class UserInDB(UserBase):
    hashed_password: str

# Utility functions
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def get_user_by_username(db: Session, username: str):
    return db.execute(select(User).where(User.username == username)).scalars().first()

def get_user_by_email(db: Session, email: str):
    return db.execute(select(User).where(User.email == email)).scalars().first()

def create_user(db: Session, user: UserCreate):
    hashed_password = get_password_hash(user.password)
    db_user = User(
        username=user.username,
        email=user.email,
        full_name=user.full_name,
        hashed_password=hashed_password,
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

def authenticate_user(db: Session, username: str, password: str):
    user = get_user_by_username(db, username)
    if not user:
        return False
    if not verify_password(password, user.hashed_password):
        return False
    return user

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

async def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except JWTError:
        raise credentials_exception
    user = get_user_by_username(db, username=token_data.username)
    if user is None:
        raise credentials_exception
    return user

async def get_current_active_user(current_user: User = Depends(get_current_user)):
    if current_user and current_user.disabled:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user

# Routes for authentication
@app.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = authenticate_user(db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/register", response_model=UserBase)
async def register_user(user: UserCreate, db: Session = Depends(get_db)):
    db_user = get_user_by_email(db, user.email)
    if db_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    return create_user(db, user)

@app.get("/users/me/", response_model=UserBase)
async def read_users_me(current_user: User = Depends(get_current_active_user)):
    return current_user

@app.get("/protected-route/")
async def read_protected_route(current_user: User = Depends(get_current_active_user)):
    return {"message": "This is a protected route", "user": current_user.username}

@app.post("/logout")
async def logout():
    return {"message": "Logged out successfully"}

# RAG STUFF HERE

# Constants
PROMPT_TEMPLATE = """
This is a chat between a user and an artificial intelligence assistant. The assistant provides helpful, detailed, and polite answers to the user's questions based on the given context below using logic and reasoning. If the answer cannot be found in the context, the assistant should indicate this and, if possible, guide the user on how to proceed or where to find more information :

{context}

---

Answer the question based on the above context: {question}
"""

# Define request model
# using default values that work well for most cases
# keeping llama3 as default model user can change it inside the API call
class PromptRequest(BaseModel):
    user_prompt: str
    model: str = "gemma2:2b"
    temperature: float = 0.02
    top_k: int = 40
    top_p: float = 0.9

# Load documents from PDFs in DATA_PATH
def load_documents():
    document_loader = PyPDFDirectoryLoader(DATA_PATH)
    return document_loader.load()

# Split documents into chunks
def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)

# Get embedding function
def get_embedding_function():
    # embeddings = OllamaEmbeddings(model="llama3")
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    return embeddings

# Calculate chunk IDs
def calculate_chunk_ids(chunks):
    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        chunk.metadata["id"] = chunk_id

    return chunks

# Add documents to Chroma database
def add_to_chroma(chunks, batch_size=5):
    db = Chroma(
        persist_directory=CHROMA_PATH, embedding_function=get_embedding_function()
    )

    chunks_with_ids = calculate_chunk_ids(chunks)

    existing_items = db.get(include=[])
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)

    if len(new_chunks):
        print(f"👉 Adding new documents: {len(new_chunks)}")
        for i in range(0, len(new_chunks), batch_size):
            batch = new_chunks[i:i+batch_size]
            batch_ids = [chunk.metadata["id"] for chunk in batch]
            db.add_documents(batch, ids=batch_ids)
            print(f"Added batch {i // batch_size + 1}/{(len(new_chunks) + batch_size - 1) // batch_size}")

        db.persist()
    else:
        print("✅ No new documents to add")

# Query function
def query_rag(query_text, model_name, temperature, top_k, top_p):
    # preparing the DB
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # search the DB (Knn search)
    # getting all context documents from chromadb, after that we use the similarity_search_with_score method to get the most similar documents to the query
    # after that we check if any of the filenames contain substrings from the query
    # if we find a matching document, we use that as the context, otherwise we use the top K most similar documents as the context
    results = db.similarity_search_with_score(query_text, k=5)

    # extracting filenames from context
    filenames_in_context = [doc.metadata.get("source") for doc, _ in results]

    # check if use query has a part that matches the files we have in chromaDB
    matching_document = None
    for doc, _ in results:
        if doc.metadata.get("source"):
            for query_substr in query_text.split():
                if query_substr.lower() in doc.metadata.get("source").lower():
                    matching_document = doc
                    break
        if matching_document:
            break

    if matching_document:
        context_text = matching_document.page_content
    else:
        context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])

    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    # Choosing the model
    model = Ollama(
        model=model_name,
        base_url='http://localhost:11434',
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
    )


    # Invoking the model
    response_text = model.invoke(prompt)

    sources = [doc.metadata.get("id", None) for doc, _score in results]
    formatted_response = {
        "response": response_text,
        "sources": sources,
        # "matching_filename": matching_document.metadata.get("source") if matching_document else None
        "matching_files": matching_document.metadata.get("id") if matching_document else None,

    }
    return formatted_response

@app.get("/")
async def get_messages():
    messages = [
        {"type": "received", "text": "Hello! How are you?"},
        {"type": "sent", "text": "I'm good, thank you! How about you?"},
        {"type": "received", "text": "I'm doing well, thanks for asking. What are you up to today?"},
        {"type": "sent", "text": "Just working on a new project. You?"},
        {"type": "received", "text": "Sounds interesting! I'm just relaxing at home."},
        {"type": "sent", "text": "Nice! Any plans for the weekend?"},
        {"type": "received", "text": "Not yet, but I'm thinking of going hiking. What about you?"},
        {"type": "sent", "text": "That sounds great! I might join you if that's okay."},
        {"type": "received", "text": "Of course, the more the merrier!"},
        {"type": "sent", "text": "Looking forward to it!"},
    ]
    return messages

@app.post("/prompt")
async def prompt(request: PromptRequest):
    user_prompt = request.user_prompt
    model_name = request.model
    temperature = request.temperature
    top_k = request.top_k
    top_p = request.top_p

    # Enhancing the prompt using the ai model
    # To change the model used to enhance the prompt, change the model name in the enhance_prompt function
    # We are using mistral model to enhance the prompt
    enhanced_prompt = enhance_prompt(user_prompt)

    try:
        response = query_rag(enhanced_prompt, model_name, temperature, top_k, top_p)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {"message": "Success", "user_prompt": user_prompt,"enhanced prompt": enhanced_prompt ,"data": response}

@app.post("/upload")
async def upload_file(uploaded_file: UploadFile = File(...)):
    try:
        file_location = os.path.join(DATA_PATH, uploaded_file.filename)
        with open(file_location, "wb") as file:
            shutil.copyfileobj(uploaded_file.file, file)

        documents = load_documents()
        chunks = split_documents(documents)
        add_to_chroma(chunks)

        return {"message": "File uploaded and processed successfully", "file_name": uploaded_file.filename}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

@app.get("/getallfiles", response_class=JSONResponse)
async def get_all_files():
    try:
        # Load documents from the Chroma database
        db = Chroma(persist_directory=CHROMA_PATH, embedding_function=get_embedding_function())

        # Retrieve all documents in the Chroma database
        all_documents = db.get(include=["metadatas"])

        # Extract the filenames and metadata
        files_metadata = []
        for doc_metadata in all_documents["metadatas"]:
            file_metadata = {
                "filename": doc_metadata.get("source"),
                "page": doc_metadata.get("page"),
                "metadata": doc_metadata
            }
            files_metadata.append(file_metadata)

        return files_metadata

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)