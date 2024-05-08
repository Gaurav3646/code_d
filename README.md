fastapi
uvicorn
sqlalchemy
psycopg2
langchain_cohere
langchain_core
langchain_postgres


from fastapi import FastAPI
from api.routes import router as api_router

app = FastAPI()

# Include the API routes
app.include_router(api_router)


from fastapi import APIRouter, UploadFile, File
from langchain_cohere import CohereEmbeddings
from langchain_core.documents import Document
from langchain_postgres import PGVector
from langchain_postgres.vectorstores import PGVector
from database import SessionLocal

router = APIRouter()

# Initialize Cohere embeddings and PGVector
embeddings = CohereEmbeddings()
collection_name = "my_docs"
vectorstore = PGVector(embeddings=embeddings, collection_name=collection_name, connection=DATABASE_URL, use_jsonb=True)

# File upload endpoint
@router.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    try:
        # Save the uploaded file to a temporary directory
        with NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(await file.read())
            tmp_file_path = tmp_file.name

        # Extract text content from the file
        with open(tmp_file_path, "r", encoding="utf-8") as f:
            file_content = f.read()

        # Embed the text content
        document = Document(content=file_content)
        embedding = embeddings.embed(document)

        # Store the embedding in PGVector
        vectorstore.store(embedding)

        return {"message": "File uploaded successfully"}
    finally:
        # Delete the temporary file
        os.remove(tmp_file_path)



from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# PostgreSQL connection string
DATABASE_URL = "postgresql+psycopg2://langchain:langchain@localhost:6024/langchain"

# Create a database engine
engine = create_engine(DATABASE_URL)

# Create a sessionmaker object for database interactions
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)



from fastapi import FastAPI, File, UploadFile, HTTPException
from langchain_community.document_loaders import PyPDFLoader 
from langchain_community.document_loaders import Docx2txtLoader, TextLoader
from langchain_community.document_loaders.merge import MergedDataLoader
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_core.documents import Document
from langchain_postgres import PGVector
from langchain_postgres.vectorstores import PGVector
import os
import shutil
import tempfile
import uuid
import logging
from pathlib import Path

app = FastAPI()

UPLOAD_DIR = "uploads"
ALLOWED_MIME_TYPE = "application/pdf"

logging.basicConfig(level=logging.INFO)

@app.post("/file/upload")
async def upload_file(file: UploadFile = File(...)):
    if file.content_type!= ALLOWED_MIME_TYPE:
        raise HTTPException(400, detail="Invalid document type")

    file_name = f"{uuid.uuid4()}.pdf"
    storage_dir = Path(UPLOAD_DIR)
    storage_dir.mkdir(exist_ok=True)

    file_path = storage_dir / file_name
    try:
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
    except Exception as e:
        logging.error(f"Error uploading file: {e}")
        raise HTTPException(500, detail="Error uploading file")

   


    # Create a loader for each file
    loaders = PyPDFLoader(file_path=file_path)

    

    # loaders.append(TextLoader(files[0]))
    pages = loaders.load_and_split()

    
    # Load the document from the file path
    document = loaders.load()

# # Create an instance of FastEmbedEmbeddings
    embeddings = FastEmbedEmbeddings()


    document_embeddings = embeddings.embed_documents(file_name)



    # See docker command above to launch a postgres instance with pgvector enabled.
    connection = "postgresql+psycopg://langchain:langchain@localhost:6024/langchain"  # Uses psycopg3!
    collection_name = "my_docs"


    vectorstore = PGVector(
        embeddings=embeddings,
        collection_name=collection_name,
        connection=connection,
        use_jsonb=True,
    )

    vectorstore.add_documents(pages)

    os.remove(file_path)

    # return {"file_name": file_name, "documents": documents}
    return {"file_path": document_embeddings}





