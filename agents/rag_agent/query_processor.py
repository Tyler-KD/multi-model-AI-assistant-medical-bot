import logging
import re
import uuid
from typing import List, Dict, Any, Optional, Tuple, Union
from datetime import datetime

class QueryProcessor:
    """
    Advanced processor for medical queries with enhanced entity extraction and specialty detection.
    """
    def __init__(self, config, embedding_model):
        """
        Initialize the query processor with expanded capabilities.
        
        Args:
            config: Configuration object
            embedding_model: Model to generate embeddings
        """
        self.logger = logging.getLogger(__name__)
        self.embedding_model = embedding_model
        
        # Enhanced medical entities detection - aligned with document processor
        # Fixed by removing inline (?i) flags and using re.IGNORECASE parameter
        self.medical_entity_categories = {
            "diseases": r"(diabetes|hypertension|cancer|asthma|covid-19|stroke|"
                      r"alzheimer's|parkinson's|arthritis|obesity|heart disease|hepatitis|"
                      r"influenza|pneumonia|tuberculosis|hiv/aids|malaria|cholera|"
                      r"diabetes mellitus|chronic kidney disease|copd)",
            
            "medications": r"(aspirin|ibuprofen|acetaminophen|lisinopril|metformin|"
                         r"atorvastatin|omeprazole|amoxicillin|prednisone|insulin|"
                         r"albuterol|levothyroxine|warfarin|clopidogrel|metoprolol)",
            
            "procedures": r"(surgery|biopsy|endoscopy|colonoscopy|mri|ct scan|x-ray|"
                        r"ultrasound|echocardiogram|ekg|ecg|angiography|mammography|"
                        r"vaccination|immunization|blood test|urinalysis)",
            
            "anatomy": r"(heart|lung|liver|kidney|brain|stomach|intestine|colon|"
                     r"pancreas|spleen|thyroid|adrenal|pituitary|bone|muscle|nerve|"
                     r"artery|vein|capillary|joint|skin)"
        }
        
        # Combine all entity patterns
        all_patterns = []
        for category, pattern in self.medical_entity_categories.items():
            all_patterns.append(f"(?P<{category}>{pattern})")
        
        self.medical_entity_pattern = re.compile("|".join(all_patterns), re.IGNORECASE)
        
        # Expanded medical specialty keywords
        self.specialty_keywords = {
            "cardiology": ["heart", "cardiac", "cardiovascular", "arrhythmia", "hypertension", 
                         "coronary", "myocardial", "angina", "pacemaker", "valve"],
            "neurology": ["brain", "neural", "stroke", "alzheimer", "seizure", "parkinson", 
                        "headache", "migraine", "neuropathy", "multiple sclerosis", "epilepsy"],
            "oncology": ["cancer", "tumor", "chemotherapy", "radiation", "oncology", 
                       "metastasis", "lymphoma", "leukemia", "carcinoma", "biopsy"],
            "pediatrics": ["child", "infant", "pediatric", "childhood", "adolescent", 
                         "newborn", "toddler", "vaccination", "growth", "developmental"],
            "psychiatry": ["mental health", "depression", "anxiety", "psychiatric", "disorder", 
                         "schizophrenia", "bipolar", "therapy", "behavioral", "psychological"],
            "orthopedics": ["bone", "joint", "fracture", "orthopedic", "arthritis", 
                          "spine", "knee", "hip", "shoulder", "cartilage", "tendon"],
            "dermatology": ["skin", "dermatological", "rash", "acne", "eczema", 
                          "melanoma", "psoriasis", "lesion", "mole", "dermatitis"],
            "endocrinology": ["hormone", "diabetes", "thyroid", "endocrine", "insulin", 
                            "pituitary", "adrenal", "metabolism", "glucose", "hyperthyroidism"],
            "gastroenterology": ["stomach", "intestine", "liver", "digestive", "gastric", 
                               "colon", "ulcer", "gallbladder", "pancreas", "hepatitis"],
            "ophthalmology": ["eye", "vision", "retina", "cataract", "glaucoma", 
                            "cornea", "ophthalmologic", "lens", "macular", "blindness"],
            "pulmonology": ["lung", "respiratory", "pulmonary", "asthma", "copd", 
                          "bronchitis", "pneumonia", "oxygen", "breathing", "emphysema"],
            "nephrology": ["kidney", "renal", "dialysis", "nephritis", "urinary", 
                         "nephrology", "proteinuria", "creatinine", "transplant", "glomerular"],
            "infectious_disease": ["infection", "bacteria", "viral", "antibiotic", "fungal", 
                                 "hiv", "sepsis", "infectious", "immune", "vaccination"]
        }
        
        # Document type detection patterns - Fixed by keeping the inline (?i) at the start only
        self.document_type_patterns = {
            "research_paper": re.compile(r"(?i)(abstract|introduction|methods|results|discussion|conclusion|references)"),
            "clinical_note": re.compile(r"(?i)(chief complaint|history of present illness|past medical history|medications|assessment|plan)"),
            "patient_record": re.compile(r"(?i)(patient information|vital signs|allergies|family history|social history)"),
            "medical_guideline": re.compile(r"(?i)(recommendations|guidelines|protocols|indications|contraindications)"),
            "drug_information": re.compile(r"(?i)(mechanism of action|pharmacokinetics|dosage|side effects|interactions)")
        }

    def process_query(self, query: str) -> Tuple[List[float], Dict[str, Any]]:
        """
        Process the query to generate embedding and extract metadata filters.
        
        Args:
            query: User query string
            
        Returns:
            Tuple of (query_embedding, extracted_filters)
        """
        try:
            # Generate a query ID for tracking
            query_id = str(uuid.uuid4())
            
            # Analyze query
            expanded_query = self._expand_query(query)
            
            # Extract medical entities with categories
            medical_entities = self._extract_medical_entities(query)
            
            # Detect medical specialty
            specialty = self._detect_specialty(query)
            
            # Detect potential document type interest
            doc_type = self._detect_document_type_interest(query)
            
            # Extract temporal context (if any)
            temporal_context = self._extract_temporal_context(query)
            
            # Determine query intent
            query_intent = self._determine_query_intent(query)
            
            # Generate embedding
            query_embedding = self.embedding_model.embed_query(expanded_query)
            if query_embedding is None:
                raise ValueError("Embedding model returned None for query.")

            query_embedding = query_embedding.tolist() if hasattr(query_embedding, "tolist") else query_embedding

            # Create comprehensive filter dict for retrieval
            filters = {
                "query_id": query_id,
                "timestamp": datetime.now().isoformat(),
                "query_intent": query_intent,
                "medical_entities": medical_entities if medical_entities else None,
                "specialty": specialty if specialty else None,
                "document_type": doc_type if doc_type else None,
                "temporal_context": temporal_context if temporal_context else None
            }
            
            # Remove None values
            filters = {k: v for k, v in filters.items() if v is not None}

            self.logger.info(f"Processed query with filters: {filters}")
            
            return query_embedding, filters
            
        except Exception as e:
            self.logger.error(f"Error processing query: {e}")
            
            # Fallback: return embedding of original query with minimal filters
            fallback_embedding = self.embedding_model.embed_query(query)
            if fallback_embedding is None:
                fallback_embedding = []
            
            fallback_embedding = fallback_embedding.tolist() if hasattr(fallback_embedding, "tolist") else fallback_embedding
            
            # Include at least a query ID in the fallback filters
            fallback_filters = {
                "query_id": str(uuid.uuid4()),
                "timestamp": datetime.now().isoformat(),
                "is_fallback": True
            }

            return fallback_embedding, fallback_filters
    
    def _expand_query(self, query: str) -> str:
        """
        Expand query with medical synonyms or related terms.
        
        Args:
            query: Original query
            
        Returns:
            Expanded query
        """
        # In production, this would use a medical knowledge graph or ontology
        # Expanded with more medical terms and conditions
        expansions = {
            "heart attack": "myocardial infarction cardiac arrest coronary thrombosis acute coronary syndrome",
            "high blood pressure": "hypertension elevated blood pressure hypertensive cardiovascular disease",
            "diabetes": "diabetes mellitus hyperglycemia glucose intolerance type 1 diabetes type 2 diabetes",
            "stroke": "cerebrovascular accident brain attack cerebral infarction hemorrhagic stroke ischemic stroke",
            "cancer": "malignancy neoplasm tumor carcinoma oncology metastasis",
            "kidney disease": "renal disease nephropathy kidney failure chronic kidney disease acute kidney injury",
            "alzheimer's": "dementia neurocognitive disorder memory loss cognitive decline alzheimer disease",
            "flu": "influenza viral infection respiratory infection seasonal flu influenza virus",
            "arthritis": "joint inflammation rheumatoid arthritis osteoarthritis inflammatory arthritis",
            "asthma": "respiratory condition bronchial hyperresponsiveness wheezing bronchoconstriction",
            "pneumonia": "lung infection pulmonary inflammation bronchopneumonia respiratory infection",
            "heart failure": "cardiac failure congestive heart failure CHF cardiomyopathy"
        }
        
        expanded = query
        for term, expansion in expansions.items():
            if re.search(r"\b" + re.escape(term) + r"\b", query.lower()):
                expanded = f"{expanded} {expansion}"
        
        return expanded
    
    def _extract_medical_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extract medical entities from text by category.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary of categorized medical entities
        """
        categorized_entities = {}
        
        for category, pattern in self.medical_entity_categories.items():
            category_pattern = re.compile(pattern)
            matches = set(m.group(0).lower() for m in category_pattern.finditer(text))
            if matches:
                categorized_entities[category] = list(matches)
        
        return categorized_entities
    
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
                # Use word boundary search for more accurate matching
                if re.search(r"\b" + re.escape(keyword.lower()) + r"\b", text_lower):
                    score += 1
            if score > 0:
                specialty_scores[specialty] = score
        
        # Return highest scoring specialty or None
        if specialty_scores:
            return max(specialty_scores.items(), key=lambda x: x[1])[0]
        return None
    
    def _detect_document_type_interest(self, text: str) -> Optional[str]:
        """
        Detect potential interest in specific document types.
        
        Args:
            text: Input text
            
        Returns:
            Detected document type interest or None
        """
        # Check for explicit mention of document types
        if re.search(r"\b(research|study|paper|publication|article)\b", text, re.IGNORECASE):
            return "research_paper"
        elif re.search(r"\b(clinical note|doctor'?s note|medical note)\b", text, re.IGNORECASE):
            return "clinical_note"
        elif re.search(r"\b(patient record|medical record|health record)\b", text, re.IGNORECASE):
            return "patient_record"
        elif re.search(r"\b(guideline|protocol|recommendation|best practice)\b", text, re.IGNORECASE):
            return "medical_guideline"
        elif re.search(r"\b(drug|medication|dosage|side effect|contraindication)\b", text, re.IGNORECASE):
            return "drug_information"
        
        # If no explicit mention, check patterns as in document processor
        doc_type_scores = {}
        for doc_type, pattern in self.document_type_patterns.items():
            matches = pattern.findall(text)
            doc_type_scores[doc_type] = len(matches)
        
        if max(doc_type_scores.values(), default=0) > 0:
            return max(doc_type_scores.items(), key=lambda x: x[1])[0]
        
        return None
    
    def _extract_temporal_context(self, text: str) -> Optional[str]:
        """
        Extract temporal context from the query.
        
        Args:
            text: Input text
            
        Returns:
            Temporal context or None
        """
        # Check for recent/latest mention
        if re.search(r"\b(recent|latest|new|current|update|today)\b", text, re.IGNORECASE):
            return "recent"
        # Check for historical mention
        elif re.search(r"\b(historical|history|past|old|previous|before|earlier)\b", text, re.IGNORECASE):
            return "historical"
        # Check for specific time periods
        elif re.search(r"\b(last|within|past)\s+(\d+)\s+(year|month|week|day)s?\b", text, re.IGNORECASE):
            return "specific_timeframe"
        
        return None
    
    def _determine_query_intent(self, text: str) -> str:
        """
        Determine the intent behind the query.
        
        Args:
            text: Input text
            
        Returns:
            Query intent category
        """
        text_lower = text.lower()
        
        # Check for definition/explanation intent
        if re.search(r"\b(what is|define|explain|describe|meaning of)\b", text_lower):
            return "definition"
        
        # Check for treatment intent
        elif re.search(r"\b(treat|therapy|medication|cure|manage|drug|prescription)\b", text_lower):
            return "treatment"
        
        # Check for diagnosis intent
        elif re.search(r"\b(diagnose|diagnostic|symptom|sign|identify|determine)\b", text_lower):
            return "diagnosis"
        
        # Check for prognosis/outcome intent
        elif re.search(r"\b(prognosis|outcome|survival|mortality|chance|likelihood|risk)\b", text_lower):
            return "prognosis"
        
        # Check for prevention intent
        elif re.search(r"\b(prevent|preventive|avoid|risk factor|reduction)\b", text_lower):
            return "prevention"
        
        # Check for epidemiology intent
        elif re.search(r"\b(prevalence|incidence|statistics|common|rate|population)\b", text_lower):
            return "epidemiology"
        
        # Check for procedure intent
        elif re.search(r"\b(procedure|surgery|operation|technique|method)\b", text_lower):
            return "procedure"
        
        # Default to information seeking
        return "general_information"