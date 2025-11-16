import os
import json
import uuid
from config import Config
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends, Request, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Dict, Union, Optional
import uvicorn
from agents.agent_decision import process_query

config = Config()

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

# Add a health check endpoint
@app.get("/health")
def health_check():
    """Health check endpoint for Docker health checks"""
    return {"status": "healthy"}

@app.post("/chat")
def chat(request: QueryRequest, response: Response, request_obj: Request):
    """Process user text query through the multi-agent system."""

    # Generate session ID for cookie if it doesn't exist
    session_id = request_obj.cookies.get("session_id", str(uuid.uuid4()))
    
    try:
        response_data = process_query(request.query)

        response_text = response_data['messages'][-1].content
        
        # Set session cookie
        response.set_cookie(key="session_id", value=session_id)

        # Check if the agent is skin lesion segmentation and find the image path
        result = {
            "response": response_text, 
            "agent": response_data["agent_name"]
        }
        
        # If it's the skin lesion segmentation agent, check for output image
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
    
    # Generate session ID for cookie if it doesn't exist
    session_id = request_obj.cookies.get("session_id", str(uuid.uuid4()))
    
    try:
        query = {"text": text, "image": file_path}
        response_data = process_query(query)
        response_text = response_data['messages'][-1].content

        # Set session cookie
        response.set_cookie(key="session_id", value=session_id)

        # Check if the agent is skin lesion segmentation and find the image path
        result = {
            "response": response_text, 
            "agent": response_data["agent_name"]
        }
        
        # If it's the skin lesion segmentation agent, check for output image
        if response_data["agent_name"] == "SKIN_LESION_AGENT, HUMAN_VALIDATION":
            segmentation_path = os.path.join(SKIN_LESION_OUTPUT, "segmentation_plot.png")
            if os.path.exists(segmentation_path):
                result["result_image"] = f"/uploads/skin_lesion_output/segmentation_plot.png"
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

@app.post("/validate")
def validate_medical_output(response: Response, request_obj: Request, validation_result: str = Form(...), comments: Optional[str] = Form(None)):
    """Handle human validation for medical AI outputs."""
    
    # Generate session ID for cookie if it doesn't exist
    session_id = request_obj.cookies.get("session_id", str(uuid.uuid4()))

    try:
        # Set session cookie
        response.set_cookie(key="session_id", value=session_id)
        
        # Re-run the agent decision system with the validation input
        validation_query = f"Validation result: {validation_result}"
        if comments:
            validation_query += f" Comments: {comments}"
        
        response_data = process_query(validation_query)

        if validation_result.lower() == 'yes':
            return {
                "status": "validated",
                "message": "**Output confirmed by human validator:**",
                "response": response_data['messages'][-1].content
            }
        else:
            return {
                "status": "rejected",
                "comments": comments,
                "message": "**Output requires further review:**",
                "response": response_data['messages'][-1].content
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)