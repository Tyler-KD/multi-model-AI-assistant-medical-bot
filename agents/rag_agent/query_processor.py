import logging
import re
from typing import List, Dict, Any, Optional, Tuple

class QueryProcessor:
    """
    Processes queries for optimal retrieval from the medical knowledge base.
    """
    def __init__(self, config, embedding_model):
        """
        Initialize the query processor.
        
        Args:
            config: Configuration object
            embedding_model: Model to generate embeddings
        """
        self.logger = logging.getLogger(__name__)
        self.embedding_model = embedding_model
        
        # Medical entity pattern (simplified - in production would use a medical NER model)
        self.medical_entity_pattern = re.compile(
            r"(?i)(diabetes|hypertension|cancer|asthma|covid-19|stroke|"
            r"alzheimer's|parkinson's|arthritis|obesity|heart disease|hepatitis|"
            r"influenza|pneumonia|tuberculosis|hiv/aids|malaria|cholera|"
            r"diabetes mellitus|chronic kidney disease|copd)"
        )
        
        # Medical specialty keywords
        self.specialty_keywords = {
            "cardiology": ["heart", "cardiac", "cardiovascular", "arrhythmia", "hypertension"],
            "neurology": ["brain", "neural", "stroke", "alzheimer", "seizure", "parkinson"],
            "oncology": ["cancer", "tumor", "chemotherapy", "radiation", "oncology"],
            "pediatrics": ["child", "infant", "pediatric", "childhood", "adolescent"],
            "psychiatry": ["mental health", "depression", "anxiety", "psychiatric", "disorder"],
            "orthopedics": ["bone", "joint", "fracture", "orthopedic", "arthritis"],
            "dermatology": ["skin", "dermatological", "rash", "acne", "eczema"],
            "endocrinology": ["hormone", "diabetes", "thyroid", "endocrine", "insulin"],
            "gastroenterology": ["stomach", "intestine", "liver", "digestive", "gastric"],
            "ophthalmology": ["eye", "vision", "retina", "cataract", "glaucoma"]
        }
    
    # def process_query(self, query: str) -> Tuple[List[float], Dict[str, Any]]:
    #     """
    #     Process the query to generate embedding and extract metadata filters.
        
    #     Args:
    #         query: User query string
            
    #     Returns:
    #         Tuple of (query_embedding, extracted_filters)
    #     """
    #     try:
    #         # Analyze query
    #         expanded_query = self._expand_query(query)
    #         medical_entities = self._extract_medical_entities(query)
    #         specialty = self._detect_specialty(query)
            
    #         # Generate embedding
    #         query_embedding = self.embedding_model.embed_query(expanded_query)#.tolist()
            
    #         # Create filter dict for retrieval
    #         filters = {
    #             "medical_entities": medical_entities if medical_entities else None,
    #             "specialty": specialty if specialty else None
    #         }
    #         # filters = {
    #         #     "medical_entities": [doc_metadata["medical_entities"]] if isinstance(doc_metadata["medical_entities"], str) else doc_metadata.get("medical_entities", []),
    #         #     "specialty": doc_metadata.get("specialty", ""),
    #         # }

            
    #         # Only keep non-None values
    #         filters = {k: v for k, v in filters.items() if v is not None}
            
    #         self.logger.info(f"Processed query with filters: {filters}")
    #         # DEBUG
    #         print(f"Filters being sent: {filters}")
    #         return query_embedding, filters
            
    #     except Exception as e:
    #         self.logger.error(f"Error processing query: {e}")
    #         # Return just the embedding with no filters as fallback
    #         return self.embedding_model.embed_query(query), {}#.tolist(), {}

    from typing import List, Tuple, Dict, Any

    def process_query(self, query: str) -> Tuple[List[float], Dict[str, Any]]:
        """
        Process the query to generate embedding and extract metadata filters.
        
        Args:
            query: User query string
            
        Returns:
            Tuple of (query_embedding, extracted_filters)
        """
        try:
            # Analyze query
            expanded_query = self._expand_query(query)
            medical_entities = self._extract_medical_entities(query) or []
            specialty = self._detect_specialty(query) or ""

            # Ensure medical_entities is a list
            if isinstance(medical_entities, str):
                medical_entities = [medical_entities]

            # Generate embedding
            query_embedding = self.embedding_model.embed_query(expanded_query)
            if query_embedding is None:
                raise ValueError("Embedding model returned None for query.")

            query_embedding = query_embedding.tolist() if hasattr(query_embedding, "tolist") else query_embedding

            # Create filter dict for retrieval
            filters = {
                "medical_entities": medical_entities if medical_entities else None,
                "specialty": specialty if specialty else None
            }
            
            # Remove None values
            filters = {k: v for k, v in filters.items() if v is not None}

            self.logger.info(f"Processed query with filters: {filters}")
            # print(f"Filters being sent: {filters}")  # DEBUG
            
            return query_embedding, filters
            
        except Exception as e:
            self.logger.error(f"Error processing query: {e}")
            
            # Fallback: return embedding of original query with empty filters
            fallback_embedding = self.embedding_model.embed_query(query)
            if fallback_embedding is None:
                fallback_embedding = []
            
            fallback_embedding = fallback_embedding.tolist() if hasattr(fallback_embedding, "tolist") else fallback_embedding

            return fallback_embedding, {}

    
    def _expand_query(self, query: str) -> str:
        """
        Expand query with medical synonyms or related terms.
        
        Args:
            query: Original query
            
        Returns:
            Expanded query
        """
        # In production, this would use a medical knowledge graph or ontology
        # For now, implementing a simple rule-based expansion
        expansions = {
            "heart attack": "myocardial infarction cardiac arrest coronary thrombosis",
            "high blood pressure": "hypertension elevated blood pressure",
            "diabetes": "diabetes mellitus hyperglycemia glucose intolerance",
            "stroke": "cerebrovascular accident brain attack cerebral infarction",
            "cancer": "malignancy neoplasm tumor carcinoma",
            "kidney disease": "renal disease nephropathy kidney failure",
            "alzheimer's": "dementia neurocognitive disorder memory loss",
            "flu": "influenza viral infection respiratory infection"
        }
        
        expanded = query
        for term, expansion in expansions.items():
            if term.lower() in query.lower():
                expanded = f"{expanded} {expansion}"
        
        return expanded
    
    def _extract_medical_entities(self, text: str) -> List[str]:
        """
        Extract medical entities from text using regex pattern.
        In production, this would be replaced with a medical NER model.
        
        Args:
            text: Input text
            
        Returns:
            List of extracted medical entities
        """
        entities = set()
        for match in self.medical_entity_pattern.finditer(text.lower()):
            entities.add(match.group(0))
        
        return list(entities)
    
    def _detect_specialty(self, text: str) -> Optional[str]:
        """
        Detect medical specialty most relevant to the query.
        
        Args:
            text: Input text
            
        Returns:
            Detected specialty or None
        """
        text_lower = text.lower()
        specialty_scores = {}
        
        # Score each specialty based on keyword matches
        for specialty, keywords in self.specialty_keywords.items():
            score = 0
            for keyword in keywords:
                if keyword.lower() in text_lower:
                    score += 1
            if score > 0:
                specialty_scores[specialty] = score
        
        # Return highest scoring specialty or None
        if specialty_scores:
            return max(specialty_scores.items(), key=lambda x: x[1])[0]
        return None
