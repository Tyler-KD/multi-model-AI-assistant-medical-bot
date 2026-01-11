import re
import logging
from typing import List, Dict, Any, Optional, Tuple

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI

class ContentProcessor:
    """
    Processes the parsed content - summarizes images, creates llm based semantic chunks
    """
    def __init__(self, config):
        """
        Initialize the response generator.
        
        Args:
            llm: Large language model for image summarization
        """
        self.logger = logging.getLogger(__name__)
        self.summarizer_model = config.rag.summarizer_model     # temperature 0.5
        self.chunker_model = config.rag.chunker_model     # temperature 0.0
    
    def summarize_images(self, images: List[str]) -> List[str]:
        """
        Summarize images using the provided model, with error handling.
        
        Args:
            images: List of image paths
            
        Returns:
            List of image summaries, with placeholders for failed images
        """
        
        prompt_template = """Describe the image in detail while keeping it concise and to the point. 
                        For context, the image is part of either a medical research paper or a research paper
                        demonstrating the use of artificial intelligence techniques like
                        machine learning and deep learning in diagnosing diseases or a medical report.
                        Be specific about graphs, such as bar plots if they are present in the image.
                        Only summarize what is present in the image, without adding any extra detail or comment.
                        Summarize the image only if it is related to the context, return 'non-informative' explicitly 
                        if the image is of some button not relevant to the context."""

        messages = [
            (
                "user",
                [
                    {"type": "text", "text": prompt_template},
                    {
                        "type": "image_url",
                        "image_url": {"url": "{image}"},
                    },
                ],
            )
        ]

        prompt = ChatPromptTemplate.from_messages(messages)
        summary_chain = prompt | self.summarizer_model | StrOutputParser()
        
        results = []
        for image in images:
            try:
                summary = summary_chain.invoke({"image": image})
                results.append(summary)
            except Exception as e:
                # Log the error if needed
                print(f"Error processing image: {str(e)}")
                # Add placeholder for the failed image
                results.append("no image summary")
        
        return results
    
    def format_document_with_images(self, parsed_document: Any, image_summaries: List[str]) -> str:
        """
        Format the parsed document by replacing image placeholders with image summaries.
        
        Args:
            parsed_document: Parsed document from doc_parser
            image_summaries: List of image summaries
            
        Returns:
            Formatted document text with image summaries
        """
        IMAGE_PLACEHOLDER = "<!-- image_placeholder -->"
        PAGE_BREAK_PLACEHOLDER = "<!-- page_break -->"
        
        formatted_parsed_document = parsed_document.export_to_markdown(
            page_break_placeholder=PAGE_BREAK_PLACEHOLDER, 
            image_placeholder=IMAGE_PLACEHOLDER
        )
        
        formatted_document = self._replace_occurrences(
            formatted_parsed_document, 
            IMAGE_PLACEHOLDER, 
            image_summaries
        )
        
        return formatted_document
    
    def _replace_occurrences(self, text: str, target: str, replacements: List[str]) -> str:
        """
        Replace occurrences of a target placeholder with corresponding replacements.
        
        Args:
            text: Text containing placeholders
            target: Placeholder to replace
            replacements: List of replacements for each occurrence
            
        Returns:
            Text with replacements
        """
        result = text
        for counter, replacement in enumerate(replacements):
            if target in result:
                if replacement.lower() != 'non-informative':
                    result = result.replace(
                        target, 
                        f'picture_counter_{counter}' + ' ' + replacement, 
                        1
                    )
                else:
                    result = result.replace(target, '', 1)
            else:
                # Instead of raising an error, just break the loop when no more occurrences are found
                break
        
        return result

    def chunk_document(self, formatted_document: str) -> List[str]:
        """
        Split the document into semantic chunks.
        
        Args:
            formatted_document: Formatted document text
            model: AzureChatOpenAI model instance (will create one if not provided)
            
        Returns:
            List of document chunks
        """
        
        # Split by section boundaries
        SPLIT_PATTERN = "\n#"
        chunks = formatted_document.split(SPLIT_PATTERN)
        
        chunked_text = ""
        for i, chunk in enumerate(chunks):
            if chunk.startswith("#"):
                chunk = f"#{chunk}"  # add the # back to the chunk
            chunked_text += f"<|start_chunk_{i}|>\n{chunk}\n<|end_chunk_{i}|>\n"
        
        # LLM-based semantic chunking
        CHUNKING_PROMPT = """
        You are an assistant specialized in splitting text into semantically consistent sections. 
        
        Following is the document text:
        <document>
        {document_text}
        </document>
        
        <instructions>
        Instructions:
            1. The text has been divided into chunks, each marked with <|start_chunk_X|> and <|end_chunk_X|> tags, where X is the chunk number.
            2. Identify points where splits should occur, such that consecutive chunks of similar themes stay together.
            3. Each chunk must be between 256 and 512 words.
            4. If chunks 1 and 2 belong together but chunk 3 starts a new topic, suggest a split after chunk 2.
            5. The chunks must be listed in ascending order.
            6. Provide your response in the form: 'split_after: 3, 5'.
        </instructions>
        
        Respond only with the IDs of the chunks where you believe a split should occur.
        YOU MUST RESPOND WITH AT LEAST ONE SPLIT.
        """.strip()
        
        formatted_chunking_prompt = CHUNKING_PROMPT.format(document_text=chunked_text)
        chunking_response = self.chunker_model.invoke(formatted_chunking_prompt).content
        
        return self._split_text_by_llm_suggestions(chunked_text, chunking_response)
    
    def _split_text_by_llm_suggestions(self, chunked_text: str, llm_response: str) -> List[str]:
        """
        Split text according to LLM suggested split points.
        
        Args:
            chunked_text: Text with chunk markers
            llm_response: LLM response with split suggestions
            
        Returns:
            List of document chunks
        """
        # Extract split points from LLM response
        split_after = [] 
        if "split_after:" in llm_response:
            split_points = llm_response.split("split_after:")[1].strip()
            split_after = [int(x.strip()) for x in split_points.replace(',', ' ').split()] 

        # If no splits were suggested, return the whole text as one section
        if not split_after:
            return [chunked_text]

        # Find all chunk markers in the text
        chunk_pattern = r"<\|start_chunk_(\d+)\|>(.*?)<\|end_chunk_\1\|>"
        chunks = re.findall(chunk_pattern, chunked_text, re.DOTALL)

        # Group chunks according to split points
        sections = []
        current_section = [] 

        for chunk_id, chunk_text in chunks:
            current_section.append(chunk_text)
            if int(chunk_id) in split_after:
                sections.append("".join(current_section).strip())
                current_section = [] 
        
        # Add the last section if it's not empty
        if current_section:
            sections.append("".join(current_section).strip())

        return sections