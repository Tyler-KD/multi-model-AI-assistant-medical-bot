from .web_search_agent import WebSearchAgent
from langchain_openai import AzureChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv()

class WebSearchProcessor:
    """
    Processes web search results and routes them to the appropriate LLM for response generation.
    """
    
    def __init__(self, config):
        self.web_search_agent = WebSearchAgent(config)
        
        # Initialize LLM for processing web search results
        self.llm = AzureChatOpenAI(
            deployment_name=os.getenv("deployment_name"),
            model_name=os.getenv("model_name"),
            azure_endpoint=os.getenv("azure_endpoint"),
            openai_api_key=os.getenv("openai_api_key"),
            openai_api_version=os.getenv("openai_api_version"),
            temperature=0.3  # Slightly creative but factual
        )
    
    def process_web_results(self, query: str) -> str:
        """
        Fetches web search results, processes them using LLM, and returns a user-friendly response.
        """
        # print(f"[WebSearchProcessor] Fetching web search results for: {query}")
        
        # Retrieve web search results
        web_results = self.web_search_agent.search(query)
        
        # Construct prompt to LLM for processing the results
        llm_prompt = (
            "You are an AI assistant specialized in medical information. Below are web search results "
            "retrieved for a user query. Summarize and generate a helpful, concise response. "
            "Use reliable sources only and ensure medical accuracy.\n\n"
            f"Query: {query}\n\nWeb Search Results:\n{web_results}\n\nResponse:"
        )
        
        # Invoke the LLM to process the results
        response = self.llm.invoke(llm_prompt)
        
        return response
