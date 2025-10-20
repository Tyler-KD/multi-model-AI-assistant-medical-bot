import requests

class PubmedSearchAgent:
    """
    Processes medical documents for the RAG system with context-aware chunking.
    """
    def __init__(self):
        """
        Initialize the Pubmed search agent.
        
        Args:
            query: User query
        """
        pass

    def search_pubmed(self, pubmed_api_url, query: str) -> str:
        """Search PubMed for relevant medical articles."""
        params = {
            "db": "pubmed",
            "term": query,
            "retmode": "json",
            "retmax": 5
        }
        
        try:
            response = requests.get(pubmed_api_url, params=params)
            data = response.json()
            article_ids = data.get("esearchresult", {}).get("idlist", [])
            if not article_ids:
                return "No relevant PubMed articles found."
            
            article_links = [f"https://pubmed.ncbi.nlm.nih.gov/{article_id}/" for article_id in article_ids]
            return "\n".join(article_links)
        except Exception as e:
            return f"Error retrieving PubMed articles: {e}"