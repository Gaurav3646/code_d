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
