"""
Agent Decision System for Multi-Agent Medical Chatbot

This module handles the orchestration of different agents using LangGraph.
It dynamically routes user queries to the appropriate agent based on content and context.
"""

import json
from typing import Dict, List, Optional, Any, Literal, TypedDict, Union
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import AzureChatOpenAI
from langgraph.graph import StateGraph, END
import os, getpass
from dotenv import load_dotenv
from agents.rag_agent import MedicalRAG
from agents.web_search_processor_agent import WebSearchProcessorAgent
from agents.image_analysis_agent import ImageAnalysisAgent

load_dotenv()

from config import Config

# Load configuration
config = Config()

class AgentConfig:
    """Configuration settings for the agent decision system."""
    
    # Decision model
    DECISION_MODEL = "gpt-4o"  # or whichever model you prefer
    DECISION_TEMPERATURE = 0.1
    
    # Vision model for image analysis
    VISION_MODEL = "gpt-4o"
    
    # Confidence threshold for responses
    CONFIDENCE_THRESHOLD = 0.85
    
    # System instructions for the decision agent
    DECISION_SYSTEM_PROMPT = """You are an intelligent medical triage system that routes user queries to 
    the appropriate specialized agent. Your job is to analyze the user's request and determine which agent 
    is best suited to handle it based on the query content, presence of images, and conversation context.

    Available agents:
    1. CONVERSATION_AGENT - For general chat, greetings, and non-medical questions.
    2. RAG_AGENT - For specific medical knowledge questions that can be answered from established medical literature.
    3. WEB_SEARCH_PROCESSOR_AGENT - For questions about recent medical developments, current outbreaks, or time-sensitive medical information.
    4. BRAIN_TUMOR_AGENT - For analysis of brain MRI images to detect and segment tumors.
    5. CHEST_XRAY_AGENT - For analysis of chest X-ray images to detect abnormalities.
    6. SKIN_LESION_AGENT - For analysis of skin lesion images to classify them as benign or malignant.

    Make your decision based on these guidelines:
    - If the user has not uploaded any image, always route to the conversation agent.
    - If the user uploads a medical image, decide which medical vision agent is appropriate based on the image type and the user's query. If the image is uploaded without a query, always route to the correct medical vision agent based on the image type.
    - If the user asks about recent medical developments or current health situations, use the web search pocessor agent.
    - If the user asks specific medical knowledge questions, use the RAG agent.
    - For general conversation, greetings, or non-medical questions, use the conversation agent. But if image is uploaded, always go to the medical vision agents first.

    You must provide your answer in JSON format with the following structure:
    {{
    "agent": "AGENT_NAME",
    "reasoning": "Your step-by-step reasoning for selecting this agent",
    "confidence": 0.95  // Value between 0.0 and 1.0 indicating your confidence in this decision
    }}
    """

    llm = config.rag.llm
    embedding_model = config.rag.embedding_model

    image_analyzer = ImageAnalysisAgent()


class AgentState(TypedDict):
    """State maintained across the workflow."""
    messages: List[BaseMessage]  # Conversation history
    agent_name: Optional[str]  # Current active agent
    current_input: Optional[Union[str, Dict]]  # Input to be processed
    has_image: bool  # Whether the current input contains an image
    image_type: Optional[str]  # Type of medical image if present
    output: Optional[str]  # Final output to user
    needs_human_validation: bool  # Whether human validation is required
    retrieval_confidence: float  # Confidence in retrieval (for RAG agent)


class AgentDecision(TypedDict):
    """Output structure for the decision agent."""
    agent: str
    reasoning: str
    confidence: float


def create_agent_graph():
    """Create and configure the LangGraph for agent orchestration."""

    # LLM
    decision_model = AzureChatOpenAI(
        deployment_name = os.getenv("deployment_name"),
        model_name = AgentConfig.DECISION_MODEL,
        azure_endpoint = os.getenv("azure_endpoint"),
        openai_api_key = os.getenv("openai_api_key"),
        openai_api_version = os.getenv("openai_api_version"),
        temperature = AgentConfig.DECISION_TEMPERATURE)
    
    # Initialize the output parser
    json_parser = JsonOutputParser(pydantic_object=AgentDecision)
    
    # Create the decision prompt
    decision_prompt = ChatPromptTemplate.from_messages([
        ("system", AgentConfig.DECISION_SYSTEM_PROMPT),
        ("human", "{input}")
    ])
    
    # Create the decision chain
    decision_chain = decision_prompt | decision_model | json_parser
    
    # Define graph state transformations
    def analyze_input(state: AgentState) -> AgentState:
        """Analyze the input to detect images and determine input type."""
        current_input = state["current_input"]
        has_image = False
        image_type = None
        
        # Check if input contains an image
        if isinstance(current_input, dict) and "image" in current_input:
            has_image = True
            image_type_response = AgentConfig.image_analyzer.process_image(current_input.get("image", None))
            image_type = image_type_response['image_type']
            print("ANALYZED IMAGE TYPE: ", image_type)
        
        return {
            **state,
            "has_image": has_image,
            "image_type": image_type
        }
    
    def route_to_agent(state: AgentState) -> Dict:
        """Make decision about which agent should handle the query."""
        messages = state["messages"]
        current_input = state["current_input"]
        has_image = state["has_image"]
        image_type = state["image_type"]
        
        # Prepare input for decision model
        input_text = ""
        if isinstance(current_input, str):
            input_text = current_input
        elif isinstance(current_input, dict):
            input_text = current_input.get("text", "")
        
        # Create context from recent conversation history (last 3 messages)
        recent_context = ""
        for msg in messages[-6:]:  # Get last 3 exchanges (6 messages)
            if isinstance(msg, HumanMessage):
                recent_context += f"User: {msg.content}\n"
            elif isinstance(msg, AIMessage):
                recent_context += f"Assistant: {msg.content}\n"
        
        # Combine everything for the decision input
        decision_input = f"""
        User query: {input_text}

        Recent conversation context:
        {recent_context}

        Has image: {has_image}
        Image type: {image_type if has_image else 'None'}

        Based on this information, which agent should handle this query?
        """
        
        # Make the decision
        decision = decision_chain.invoke({"input": decision_input})

        # Decided agent
        print(f"Decision: {decision['agent']}")
        
        # Update state with decision
        updated_state = {
            **state,
            "agent_name": decision["agent"],
        }
        
        # Route based on agent name and confidence
        if decision["confidence"] < AgentConfig.CONFIDENCE_THRESHOLD:
            return {"agent_state": updated_state, "next": "needs_validation"}
        
        return {"agent_state": updated_state, "next": decision["agent"]}

    # Define agent execution functions (these will be implemented in their respective modules)
    def run_conversation_agent(state: AgentState) -> AgentState:
        """Handle general conversation."""

        print(f"Selected agent: CONVERSATION_AGENT")

        messages = state["messages"]
        current_input = state["current_input"]
        
        # Prepare input for decision model
        input_text = ""
        if isinstance(current_input, str):
            input_text = current_input
        elif isinstance(current_input, dict):
            input_text = current_input.get("text", "")
        
        # Create context from recent conversation history (last 3 messages)
        recent_context = ""
        for msg in messages[-6:]:  # Get last 3 exchanges (6 messages)
            if isinstance(msg, HumanMessage):
                recent_context += f"User: {msg.content}\n"
            elif isinstance(msg, AIMessage):
                recent_context += f"Assistant: {msg.content}\n"
        
        # Combine everything for the decision input
        conversation_prompt = f"""User query: {input_text}

        Recent conversation context: {recent_context}

        You are an AI-powered Medical Conversation Assistant. Your goal is to facilitate smooth and informative conversations with users, handling both casual and medical-related queries. You must respond naturally while ensuring medical accuracy and clarity.

        ### Role & Capabilities
        - Engage in **general conversation** while maintaining professionalism.
        - Answer **medical questions** using verified knowledge.
        - Route **complex queries** to RAG (retrieval-augmented generation) or web search if needed.
        - Handle **follow-up questions** while keeping track of conversation context.
        - Redirect **medical images** to the appropriate AI analysis agent.

        ### Guidelines for Responding:
        1. **General Conversations:**
        - If the user engages in casual talk (e.g., greetings, small talk), respond in a friendly, engaging manner.
        - Keep responses **concise and engaging**, unless a detailed answer is needed.

        2. **Medical Questions:**
        - If you have **high confidence** in answering, provide a medically accurate response.
        - Ensure responses are **clear, concise, and factual**.

        3. **Follow-Up & Clarifications:**
        - Maintain conversation history for better responses.
        - If a query is unclear, ask **follow-up questions** before answering.

        4. **Handling Medical Image Analysis:**
        - Do **not** attempt to analyze images yourself.
        - If user speaks about analyzing or processing or detecting or segmenting or classifying any disease from any image, ask the user to upload the image so that in the next turn it is routed to the appropriate medical vision agents.
        - If an image was uploaded, it would have been routed to the medical computer vision agents. Read the history to know about the diagnosis results and continue conversation if user asks anything regarding the diagnosis.
        - After processing, **help the user interpret the results**.

        5. **Uncertainty & Ethical Considerations:**
        - If unsure, **never assume** medical facts.
        - Recommend consulting a **licensed healthcare professional** for serious medical concerns.
        - Avoid providing **medical diagnoses** or **prescriptions**—stick to general knowledge.

        ### Response Format:
        - Maintain a **conversational yet professional tone**.
        - Use **bullet points or numbered lists** for clarity when needed.
        - If pulling from external sources (RAG/Web Search), mention **where the information is from** (e.g., "According to Mayo Clinic...").
        - If a user asks for a diagnosis, remind them to **seek medical consultation**.

        ### Example User Queries & Responses:

        **User:** "Hey, how's your day going?"
        **You:** "I'm here and ready to help! How can I assist you today?"

        **User:** "I have a headache and fever. What should I do?"
        **You:** "I'm not a doctor, but headaches and fever can have various causes, from infections to dehydration. If your symptoms persist, you should see a medical professional."

        Conversational LLM Response:"""

        response = AgentConfig.llm.invoke(conversation_prompt)

        print("########### DEBUGGING #########: reponse from conversation agent llm:", response)

        # response = AIMessage(content="This would be handled by the conversation agent.")

        return {
            **state,
            "output": response,
            "agent_name": "CONVERSATION_AGENT"
        }
    
    def run_rag_agent(state: AgentState) -> AgentState:
        """Handle medical knowledge queries using RAG."""
        # Initialize the RAG agent

        print(f"Selected agent: RAG_AGENT")

        rag_agent = MedicalRAG(config, AgentConfig.llm, AgentConfig.embedding_model)
        
        # Process the query
        query = state["current_input"]
        # print("############### TEST 1 #################")
        # print("State messages:", state["messages"])
        # chat_history = [{"role": "user", "content": msg.content} if isinstance(msg, HumanMessage) else {"role": "assistant", "content": msg.content} for msg in state["messages"]]
        chat_history = []   ##################### Debugging
        for msg in state["messages"]:
            if isinstance(msg, dict):
                # If it's a dictionary, add it directly (assuming it has the right structure)
                chat_history.append(msg)
            else:
                # Otherwise, assume it's a HumanMessage or similar object
                role_type = "user" if isinstance(msg, HumanMessage) else "assistant"
                chat_history.append({"role": role_type, "content": msg.content})
        # print("############### TEST 2 #################")
        # print("State messages:", state["messages"])
        response = rag_agent.process_query(query, chat_history)
        # print("############### TEST 3 #################")
        # print("State messages:", state["messages"])
        retrieval_confidence = response.get("confidence", 0.0)  # Default to 0.0 if not provided
        # print("############### TEST 4 #################")

        print(f"Retrieval Confidence: {retrieval_confidence}")
        print(f"Sources: {len(response['sources'])}")

        # Store RAG output ONLY if confidence is high
        if retrieval_confidence >= config.rag.min_retrieval_confidence:
            temp_output = response["response"]
        else:
            temp_output = ""
        
        return {
            **state,
            "output": temp_output,
            "needs_human_validation": False,  # Assuming no validation needed for RAG responses
            "retrieval_confidence": retrieval_confidence,
            "agent_name": "RAG_AGENT"
        }

    # Web Search Processor Node
    def run_web_search_processor_agent(state: AgentState) -> AgentState:
        """Handles web search results, processes them with LLM, and generates a refined response."""

        print(f"Selected agent: WEB_SEARCH_PROCESSOR_AGENT")
        print("[WEB_SEARCH_PROCESSOR_AGENT] Processing Web Search Results...")

        web_search_processor = WebSearchProcessorAgent(config)
        processed_response = web_search_processor.process_web_search_results(query=state["current_input"])
        
        if state['agent_name'] != None:
            involved_agents = f"{state['agent_name']}, WEB_SEARCH_PROCESSOR_AGENT"
        else:
            involved_agents = "WEB_SEARCH_PROCESSOR_AGENT"

        # Overwrite any previous output with the processed Web Search response
        return {
            **state,
            # "output": "This would be handled by the web search agent, finding the latest information.",
            "output": processed_response,
            "agent_name": involved_agents
        }
    
    def run_brain_tumor_agent(state: AgentState) -> AgentState:
        """Handle brain MRI image analysis."""

        print(f"Selected agent: BRAIN_TUMOR_AGENT")

        response = AIMessage(content="This would be handled by the brain tumor agent, analyzing the MRI image.")

        return {
            **state,
            "output": response,
            "needs_human_validation": True,  # Medical diagnosis always needs validation
            "agent_name": "BRAIN_TUMOR_AGENT"
        }
    
    def run_chest_xray_agent(state: AgentState) -> AgentState:
        """Handle chest X-ray image analysis."""

        print(f"Selected agent: CHEST_XRAY_AGENT")

        response = AIMessage(content="This would be handled by the chest X-ray agent, analyzing the image.")

        return {
            **state,
            "output": response,
            "needs_human_validation": True,  # Medical diagnosis always needs validation
            "agent_name": "CHEST_XRAY_AGENT"
        }
    
    def run_skin_lesion_agent(state: AgentState) -> AgentState:
        """Handle skin lesion image analysis."""

        print(f"Selected agent: SKIN_LESION_AGENT")

        response = AIMessage(content="This would be handled by the skin lesion agent, analyzing the skin image.")

        return {
            **state,
            "output": response,
            "needs_human_validation": True,  # Medical diagnosis always needs validation
            "agent_name": "SKIN_LESION_AGENT"
        }
    
    def handle_human_validation(state: AgentState) -> Dict:
        """Prepare for human validation if needed."""
        if state.get("needs_human_validation", False):
            return {"agent_state": state, "next": "human_validation", "agent": "HUMAN_VALIDATION"}
        return {"agent_state": state, "next": END}
    
    def perform_human_validation(state: AgentState) -> AgentState:
        """Simulate human validation process."""

        print(f"Selected agent: HUMAN_VALIDATION")

        if state['agent_name'] != None:
            involved_agents = f"{state['agent_name']}, HUMAN_VALIDATION"
        else:
            involved_agents = "HUMAN_VALIDATION"
        
        response = AIMessage(content=f"[HUMAN VALIDATED]: {state['output'].content}")

        return {
            **state,
            "output": response,
            "agent_name": involved_agents
        }
    
    # Define Routing Logic
    def confidence_based_routing(state: AgentState) -> Dict[str, str]:
        """Route based on RAG confidence score."""
        if state.get("retrieval_confidence", 0.0) < config.rag.min_retrieval_confidence:
            print("Re-routed to Web Search Agent due to low confidence...")
            return "WEB_SEARCH_PROCESSOR_AGENT"  # Correct format
        return "check_validation"  # No transition needed if confidence is high

    
    # Create the workflow graph
    workflow = StateGraph(AgentState)
    
    # Add nodes for each step
    workflow.add_node("analyze_input", analyze_input)
    workflow.add_node("route_to_agent", route_to_agent)
    workflow.add_node("CONVERSATION_AGENT", run_conversation_agent)
    workflow.add_node("RAG_AGENT", run_rag_agent)
    workflow.add_node("WEB_SEARCH_PROCESSOR_AGENT", run_web_search_processor_agent)
    workflow.add_node("BRAIN_TUMOR_AGENT", run_brain_tumor_agent)
    workflow.add_node("CHEST_XRAY_AGENT", run_chest_xray_agent)
    workflow.add_node("SKIN_LESION_AGENT", run_skin_lesion_agent)
    workflow.add_node("check_validation", handle_human_validation)
    workflow.add_node("human_validation", perform_human_validation)
    
    # Define the edges (workflow connections)
    workflow.set_entry_point("analyze_input")
    workflow.add_edge("analyze_input", "route_to_agent")
    
    # Connect decision router to agents
    workflow.add_conditional_edges(
        "route_to_agent",
        lambda x: x["next"],
        {
            "CONVERSATION_AGENT": "CONVERSATION_AGENT",
            "RAG_AGENT": "RAG_AGENT",
            "WEB_SEARCH_PROCESSOR_AGENT": "WEB_SEARCH_PROCESSOR_AGENT",
            "BRAIN_TUMOR_AGENT": "BRAIN_TUMOR_AGENT",
            "CHEST_XRAY_AGENT": "CHEST_XRAY_AGENT",
            "SKIN_LESION_AGENT": "SKIN_LESION_AGENT",
            "needs_validation": "RAG_AGENT"  # Default to RAG if confidence is low
        }
    )
    
    # Connect agent outputs to validation check
    workflow.add_edge("CONVERSATION_AGENT", "check_validation")
    # workflow.add_edge("RAG_AGENT", "check_validation")
    workflow.add_edge("WEB_SEARCH_PROCESSOR_AGENT", "check_validation")
    workflow.add_conditional_edges("RAG_AGENT", confidence_based_routing)
    workflow.add_edge("BRAIN_TUMOR_AGENT", "check_validation")
    workflow.add_edge("CHEST_XRAY_AGENT", "check_validation")
    workflow.add_edge("SKIN_LESION_AGENT", "check_validation")
    
    # Connect validation nodes
    workflow.add_conditional_edges(
        "check_validation",
        lambda x: x["next"],
        {
            "human_validation": "human_validation",
            END: END
        }
    )
    
    workflow.add_edge("human_validation", END)
    
    # Compile the graph
    return workflow.compile()


def init_agent_state() -> AgentState:
    """Initialize the agent state with default values."""
    return {
        "messages": [],
        "agent_name": None,
        "current_input": None,
        "has_image": False,
        "image_type": None,
        "output": None,
        "needs_human_validation": False,
        "retrieval_confidence": 0.0
    }


def process_query(query: Union[str, Dict], conversation_history: List[BaseMessage] = None) -> str:
    """
    Process a user query through the agent decision system.
    
    Args:
        query: User input (text string or dict with text and image)
        conversation_history: Optional list of previous messages
        
    Returns:
        Response from the appropriate agent
    """
    # Initialize the graph
    graph = create_agent_graph()
    
    # Initialize state
    state = init_agent_state()
    if conversation_history:
        state["messages"] = conversation_history
    
    # Add the current query
    state["current_input"] = query

    result = graph.invoke(state)
    
    # Add the response to conversation history
    return result