import requests
from langchain_community.tools.tavily_search import TavilySearchResults

class TavilySearchAgent:
    """
    Processes general documents for the RAG system with context-aware chunking.
    """
    def __init__(self):
        """
        Initialize the Tavily search agent.
        
        Args:
            query: User query
        """
        pass

    def search_tavily(self, tavily_api_key, query: str) -> str:
        """Perform a general web search using Tavily API."""

        tavily_search = TavilySearchResults(max_results = 3)

        # url = "https://api.tavily.com/search"
        # params = {
        #     "api_key": tavily_api_key,
        #     "query": query,
        #     "num_results": 5
        # }
        
        try:
            # response = requests.get(url, params=params)
            search_docs = tavily_search.invoke(query)
            # data = response.json()
            # if "results" in data:
            if len(search_docs):
                return "\n".join(["title: " + str(res["title"]) + " - " + 
                                  "url: " + str(res["url"]) + " - " + 
                                  "content: " + str(res["content"]) + " - " + 
                                  "score: " + str(res["score"]) for res in search_docs])
            return "No relevant results found."
        except Exception as e:
            return f"Error retrieving web search results: {e}"