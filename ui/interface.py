import gradio as gr
from orchestrator.agent_orchestrator import AgentOrchestrator

# Initialize the agent orchestrator to interact with the agents
orchestrator = AgentOrchestrator()

# Define the function to handle user input and image
def process_input(image=None, text=None):
    if image:
        # Handle image input (you can modify this to work with your specific models)
        response = orchestrator.analyze_image(image)  # Adjust based on your orchestrator
    elif text:
        # Handle text input (query to the chatbot)
        response = orchestrator.get_response(text)  # Adjust based on your orchestrator
    return response

# Create the Gradio interface with both text and image inputs
def create_ui():
    interface = gr.Interface(
        fn=process_input,
        inputs=[
            gr.Image(label="Upload a Medical Image (e.g., X-ray, MRI, etc.)", type="pil"),  # Image input
            gr.Textbox(label="Enter your medical query", placeholder="Ask about medical conditions or symptoms", lines=2)  # Text input
        ],
        outputs=gr.Textbox(label="Chatbot Response"),
        title="Medical Assistance Chatbot",
        description="Upload medical images or ask medical questions, and get AI-powered insights for diagnosis and support."
    )
    return interface