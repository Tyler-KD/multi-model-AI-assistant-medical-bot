# from fastapi import FastAPI, UploadFile, File, Form
# from pydantic import BaseModel
# from orchestrator.agent_orchestrator import AgentOrchestrator

# app = FastAPI()
# orchestrator = AgentOrchestrator()

# class UserInput(BaseModel):
#     query: str
#     image_uploaded: bool = False

# @app.post("/query")
# async def process_query(user_input: UserInput):
#     """Handles user queries by passing them through the agent orchestrator."""
#     response = orchestrator.run(user_input.query, user_input.image_uploaded)
#     return {"response": response}

# @app.post("/upload-image")
# async def upload_image(file: UploadFile = File(...), query: str = Form(...)):
#     """Handles image uploads and passes them to the agent orchestrator."""
#     image_bytes = await file.read()
#     response = orchestrator.run(query, image_uploaded=True)
#     return {"response": response}


from ui.interface import create_ui

def main():
    # Launch Gradio UI (or you can add additional functionality as needed)
    create_ui().launch()

if __name__ == "__main__":
    main()