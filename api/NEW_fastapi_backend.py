import os
import json
import uuid
import time
from pathlib import Path
from config import Config
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends, Request, Response, BackgroundTasks
from pydantic import BaseModel
from typing import Dict, Union, List, Optional
import uvicorn
from agents.agent_decision import process_query
from langchain_core.messages import HumanMessage, AIMessage

# Import Medical RAG components
from agents.rag_agent import MedicalRAG

config = Config()

# Add a simple in-memory storage (use a database in production)
conversation_store = {}

UPLOAD_FOLDER = "uploads"  # Define your desired upload folder
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # Create the folder if it doesn't exist

app = FastAPI(title="Multi-Agent Medical Chatbot", version="1.0")


embedding_model = config.rag.embedding_model

# Initialize Medical RAG system
llm = config.rag.llm
embedding_model = config.rag.embedding_model
rag_system = MedicalRAG(config, llm, embedding_model = embedding_model)

class QueryRequest(BaseModel):
    query: str

class IngestFileRequest(BaseModel):
    file_path: str

class IngestDirectoryRequest(BaseModel):
    directory_path: str

class IncrementalProcessRequest(BaseModel):
    directory_path: str
    hours_since_last_update: Optional[int] = 24

class PriorityProcessRequest(BaseModel):
    priority_files: List[str]

@app.post("/chat")
def chat(request: QueryRequest, response: Response, request_obj: Request):
    """Process user text query through the multi-agent system."""

    # Get or create session ID
    session_id = request_obj.cookies.get("session_id", str(uuid.uuid4()))
    
    # Set or retrieve conversation history for this session
    if session_id not in conversation_store:
        conversation_store[session_id] = []
    
    # Get existing history
    history = conversation_store[session_id]

    # Update conversation history with user query
    if isinstance(request.query, str):
        history.append(HumanMessage(content=request.query))
    else:
        history.append(HumanMessage(content=request.query["text"]))

    try:
        response_data = process_query(request.query, history)

        response_text = response_data['output'].content
        
        # Update conversation history with agent response
        history.append(AIMessage(content=response_text))

        # Keep history to reasonable size (optional)
        if len(history) > config.max_conversation_history:  # Keep last config.max_conversation_history messages
            history = history[-config.max_conversation_history:]
        
        # Update the stored history
        conversation_store[session_id] = history

        # Set session cookie
        response.set_cookie(key="session_id", value=session_id)

        print(history)

        return {"response": response_text, "agent": response_data["agent_name"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload")
async def upload_image(response: Response, request_obj: Request, image: UploadFile = File(...), text: str = Form("")):
    """Process medical image uploads with optional text input."""

    file_path = os.path.join(UPLOAD_FOLDER, image.filename)
    with open(file_path, "wb") as file:
        file.write(await image.read())
    
    # Get or create session ID
    session_id = request_obj.cookies.get("session_id", str(uuid.uuid4()))
    
    # Set or retrieve conversation history for this session
    if session_id not in conversation_store:
        conversation_store[session_id] = []
    
    # Get existing history
    history = conversation_store[session_id]
    
    # Update conversation history with user query
    if len(text):
        history.append(HumanMessage(content=text))
    else:
        history.append(HumanMessage(content="User uploaded an image for diagnosis."))

    try:
        query = {"text": text, "image": file_path}
        response_data = process_query(query)
        response_text = response_data['output'].content

        # Update conversation history with agent response
        history.append(AIMessage(content=response_text))

        # Keep history to reasonable size (optional)
        if len(history) > config.max_conversation_history:  # Keep last config.max_conversation_history messages
            history = history[-config.max_conversation_history:]
        
        # Update the stored history
        conversation_store[session_id] = history

        # Set session cookie
        response.set_cookie(key="session_id", value=session_id)

        print(history)

        return {"response": response_text, "agent": response_data["agent_name"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# # --- RAG Data Ingestion Endpoints ---

# @app.post("/ingest/file")
# async def ingest_file(request: IngestFileRequest, background_tasks: BackgroundTasks):
#     """Ingest a single file into the Medical RAG system."""
#     try:
#         # Check if file exists
#         if not os.path.exists(request.file_path):
#             raise HTTPException(status_code=404, detail=f"File not found: {request.file_path}")
        
#         # Process in background to avoid blocking
#         background_tasks.add_task(rag_system.ingest_file, request.file_path)
        
#         return {"status": "success", "message": f"File ingestion started for {request.file_path}"}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

# @app.post("/ingest/directory")
# async def ingest_directory(request: IngestDirectoryRequest, background_tasks: BackgroundTasks):
#     """Ingest an entire directory of files into the Medical RAG system."""
#     try:
#         # Check if directory exists
#         if not os.path.isdir(request.directory_path):
#             raise HTTPException(status_code=404, detail=f"Directory not found: {request.directory_path}")
        
#         # Process in background to avoid blocking
#         background_tasks.add_task(rag_system.ingest_directory, request.directory_path)
        
#         return {"status": "success", "message": f"Directory ingestion started for {request.directory_path}"}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

# @app.post("/ingest/incremental")
# async def process_incremental(request: IncrementalProcessRequest, background_tasks: BackgroundTasks):
#     """Process only new or modified files since last update."""
#     try:
#         # Check if directory exists
#         if not os.path.isdir(request.directory_path):
#             raise HTTPException(status_code=404, detail=f"Directory not found: {request.directory_path}")
        
#         # Calculate last update time
#         last_update_time = time.time() - (request.hours_since_last_update * 3600)
        
#         # Process in background to avoid blocking
#         background_tasks.add_task(
#             rag_system.process_ingested_data,
#             process_type="incremental",
#             directory=request.directory_path,
#             last_update=last_update_time
#         )
        
#         return {
#             "status": "success", 
#             "message": f"Incremental processing started for {request.directory_path}",
#             "time_threshold": time.ctime(last_update_time)
#         }
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

# @app.post("/ingest/priority")
# async def process_priority(request: PriorityProcessRequest, background_tasks: BackgroundTasks):
#     """Priority processing of specific files."""
#     try:
#         # Check if all files exist
#         missing_files = [f for f in request.priority_files if not os.path.exists(f)]
#         if missing_files:
#             raise HTTPException(status_code=404, detail=f"Files not found: {', '.join(missing_files)}")
        
#         # Process in background to avoid blocking
#         background_tasks.add_task(
#             rag_system.process_ingested_data,
#             process_type="priority",
#             priority_files=request.priority_files
#         )
        
#         return {
#             "status": "success", 
#             "message": f"Priority processing started for {len(request.priority_files)} files"
#         }
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

# @app.get("/rag/quality")
# async def analyze_source_quality():
#     """Analyze the quality of sources in the RAG system."""
#     try:
#         quality_analysis = rag_system.analyze_source_quality()
        
#         if not quality_analysis['success']:
#             raise HTTPException(status_code=500, detail="Failed to analyze source quality")
            
#         return {
#             "status": "success",
#             "quality_metrics": quality_analysis['quality_metrics']
#         }
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

# # @app.post("/rag/query")
# # async def direct_rag_query(request: QueryRequest):
# #     """Query the RAG system directly without going through the agent system."""
# #     try:
# #         query_result = rag_system.process_query(request.query)
        
# #         return {
# #             "response": query_result['response'],
# #             "sources": query_result.get('sources', [])
# #         }
# #     except Exception as e:
# #         raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)