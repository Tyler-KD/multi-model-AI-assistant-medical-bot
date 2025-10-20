"""
Test script for the Agent Decision System

This script demonstrates how the agent decision system works with different types of queries.
"""

import json
from agents.agent_decision import process_query
from langchain_core.messages import HumanMessage, AIMessage


def test_agent_decision():
    """Run tests with different types of queries to see agent selection logic."""
    
    # Test cases
    test_cases = [
        {
            "name": "General greeting",
            "query": "Hello, how are you today?",
            "expected_agent": "CONVERSATION_AGENT"
        },
        {
            "name": "Medical knowledge question",
            "query": "What are the symptoms of diabetes?",
            "expected_agent": "RAG_AGENT"
        },
        {
            "name": "Medical knowledge question",
            "query": "How is hypertension treated?",
            "expected_agent": "RAG_AGENT"
        },
        {
            "name": "Medical knowledge question",
            "query": "What is the connection between diabetes and hypertension?",
            "expected_agent": "RAG_AGENT"
        },
        {
            "name": "General medical question",
            "query": "What causes a fever?",
            "expected_agent": "WEB_SEARCH_AGENT"
        },
        {
            "name": "General medical question",
            "query": "How do I treat a cold?",
            "expected_agent": "WEB_SEARCH_AGENT"
        },
        {
            "name": "Recent medical development",
            "query": "Are there any new treatments for COVID-19 in clinical trials?",
            "expected_agent": "WEB_SEARCH_AGENT"
        },
        {
            "name": "Brain MRI image upload",
            "query": {
                "text": "I have this brain MRI scan. Can you check if there's a tumor?",
                "image": "mock_brain_mri.jpg"  # This is just a placeholder
            },
            "expected_agent": "BRAIN_TUMOR_AGENT"
        },
        {
            "name": "Chest X-ray image upload",
            "query": {
                "text": "Here's my chest X-ray. Is there anything abnormal?",
                "image": "mock_chest_xray.jpg"  # This is just a placeholder
            },
            "expected_agent": "CHEST_XRAY_AGENT"
        },
        {
            "name": "Skin lesion image upload",
            "query": {
                "text": "I have this mole on my arm. Does it look concerning?",
                "image": "mock_skin_lesion.jpg"  # This is just a placeholder
            },
            "expected_agent": "SKIN_LESION_AGENT"
        }
    ]
    
    # Run each test case
    conversation_history = []
    
    for test_case in test_cases:
        print(f"\n===== Testing: {test_case['name']} =====")
        print(f"Query: {test_case['query']}")
        
        # Process the query
        response = process_query(test_case["query"], conversation_history)
        

        try:
            response_text = response['output'].content
            # confidence_score = response['confidence']
            # sources = response['sources']
            # agent = response['agent']
        except:
            response_text = response['output']
            
        print(f"Response: {response_text}")
        print(f"Agent: {response['agent_name']}")
        
        # Update conversation history
        if isinstance(test_case["query"], str):
            conversation_history.append(HumanMessage(content=test_case["query"]))
        else:
            conversation_history.append(HumanMessage(content=test_case["query"]["text"]))
        
        try:
            conversation_history.append(AIMessage(content=response['output'].content))
        except:
            conversation_history.append(AIMessage(content=response['output']))
        
        # Keep conversation history to a reasonable size
        if len(conversation_history) > 10:
            conversation_history = conversation_history[-10:]
        
        print("=" * 50)


if __name__ == "__main__":
    test_agent_decision()
