import os
import json
import uuid
from config import Config
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends, Request, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Dict, Union
import uvicorn
from agents.agent_decision import process_query
from langchain_core.messages import HumanMessage, AIMessage

config = Config()

# Add a simple in-memory storage (use a database in production)
conversation_store = {}

UPLOAD_FOLDER = "uploads/backend"  # Define your desired upload folder
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # Create the folder if it doesn't exist

# Create output folders
SKIN_LESION_OUTPUT = "uploads/skin_lesion_output"
os.makedirs(SKIN_LESION_OUTPUT, exist_ok=True)

app = FastAPI(title="Multi-Agent Medical Chatbot", version="1.0")

# Mount the uploads directory to make images accessible via URL
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

class QueryRequest(BaseModel):
    query: str

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

        response_text = response_data['messages'][-1].content
        
        # Update conversation history with agent response
        history.append(AIMessage(content=response_text))

        # Keep history to reasonable size (optional)
        if len(history) > config.max_conversation_history:  # Keep last config.max_conversation_history messages
            history = history[-config.max_conversation_history:]
        
        # Update the stored history
        conversation_store[session_id] = history

        # Set session cookie
        response.set_cookie(key="session_id", value=session_id)

        # print(history)

        # Check if the agent is skin lesion segmentation and find the image path
        result = {
            "response": response_text, 
            "agent": response_data["agent_name"]
        }
        
        # If it's the skin lesion segmentation agent, check for output image
        # print("########## DEBUGGING ########## Agent Name:", response_data["agent_name"])
        if response_data["agent_name"] == "SKIN_LESION_AGENT, HUMAN_VALIDATION":
            segmentation_path = os.path.join(SKIN_LESION_OUTPUT, "segmentation_plot.png")
            if os.path.exists(segmentation_path):
                result["result_image"] = f"/uploads/skin_lesion_output/segmentation_plot.png"
                print(result)
            else:
                print("Skin Lesion Output path does not exist.")
        
        return result
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
        response_data = process_query(query, history)
        response_text = response_data['messages'][-1].content

        # Update conversation history with agent response
        history.append(AIMessage(content=response_text))

        # Keep history to reasonable size (optional)
        if len(history) > config.max_conversation_history:  # Keep last config.max_conversation_history messages
            history = history[-config.max_conversation_history:]
        
        # Update the stored history
        conversation_store[session_id] = history

        # Set session cookie
        response.set_cookie(key="session_id", value=session_id)

        # print(history)

        # Check if the agent is skin lesion segmentation and find the image path
        result = {
            "response": response_text, 
            "agent": response_data["agent_name"]
        }
        
        # If it's the skin lesion segmentation agent, check for output image
        # print("########## DEBUGGING ########## Agent Name:", response_data["agent_name"])
        if response_data["agent_name"] == "SKIN_LESION_AGENT, HUMAN_VALIDATION":
            segmentation_path = os.path.join(SKIN_LESION_OUTPUT, "segmentation_plot.png")
            if os.path.exists(segmentation_path):
                result["result_image"] = f"/uploads/skin_lesion_output/segmentation_plot.png"
                # print(result)
            else:
                print("Skin Lesion Output path does not exist.")
        
        # Remove temporary file after sending
        try:
            os.remove(file_path)
        except Exception as e:
            print(f"Failed to remove temporary file: {str(e)}")
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)