import requests
from typing import Dict

from .pubmed_search import PubmedSearchAgent
from .tavily_search import TavilySearchAgent

class WebSearchAgent:
    """
    Agent responsible for retrieving real-time medical information from web sources.
    """
    
    def __init__(self, config):
        self.tavily_search_agent = TavilySearchAgent()
        
        # self.pubmed_search_agent = PubmedSearchAgent()
        # self.pubmed_api_url = config.pubmed_api_url
    
    def search(self, query: str) -> str:
        """
        Perform both general and medical-specific searches.
        """
        # print(f"[WebSearchAgent] Searching for: {query}")
        
        tavily_results = self.tavily_search_agent.search_tavily(query=query)
        # pubmed_results = self.pubmed_search_agent.search_pubmed(self.pubmed_api_url, query)
        
        return f"Tavily Results:\n{tavily_results}\n"
        # \nPubMed Results:\n{pubmed_results}"
