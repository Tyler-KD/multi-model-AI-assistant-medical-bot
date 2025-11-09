import os
import json
import logging
from pathlib import Path
import pandas as pd
from typing import List, Dict, Any, Optional, Union
from unstructured.partition.pdf import partition_pdf
from unstructured.chunking.title import chunk_by_title
from PyPDF2 import PdfReader

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MedicalDataIngestion:
    """
    Handles ingestion of various medical data formats into the RAG system.
    """
    def __init__(self):
        """
        Initialize the data ingestion pipeline.
        
        Args:
            config_path: Optional path to configuration file
        """
        # Initialize stats
        self.stats = {
            "files_processed": 0,
            "documents_ingested": 0,
            "errors": 0
        }
        
        # For simplicity, we'll just log instead of integrating with RAG system
        logger.info("MedicalDataIngestion initialized")
    
    def ingest_directory(self, directory_path: str, file_extension: Optional[str] = None) -> Dict[str, Any]:
        """
        Ingest all files in a directory.
        
        Args:
            directory_path: Path to directory containing files
            file_extension: Optional file extension filter (e.g., ".txt", ".pdf")
            
        Returns:
            Dictionary with ingestion statistics
        """
        logger.info(f"Processing directory: {directory_path}")
        
        try:
            directory = Path(directory_path)
            if not directory.exists() or not directory.is_dir():
                raise ValueError(f"Directory does not exist: {directory_path}")
            
            # Get all files with the specified extension
            if file_extension:
                files = list(directory.glob(f"*{file_extension}"))
            else:
                files = [f for f in directory.iterdir() if f.is_file()]
            
            logger.info(f"Found {len(files)} files to process")
            
            for file_path in files:
                try:
                    self.ingest_file(str(file_path))
                    self.stats["files_processed"] += 1
                except Exception as e:
                    logger.error(f"Error processing file {file_path}: {e}")
                    self.stats["errors"] += 1
            
            return self.stats
            
        except Exception as e:
            logger.error(f"Error processing directory: {e}")
            return self.stats
    
    def ingest_file(self, file_path: str) -> Dict[str, Any]:
        """
        Ingest a single file based on its extension.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Dictionary with ingestion results
        """
        logger.info(f"Processing file: {file_path}")
        
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Handle different file formats
        if file_path.suffix.lower() == '.txt':
            return self._ingest_text_file(file_path)
        elif file_path.suffix.lower() == '.csv':
            return self._ingest_csv_file(file_path)
        elif file_path.suffix.lower() == '.json':
            return self._ingest_json_file(file_path)
        elif file_path.suffix.lower() == '.pdf':
            return self._ingest_pdf_file(file_path)
        else:
            logger.warning(f"Unsupported file format: {file_path.suffix}")
            return {"success": False, "error": "Unsupported file format"}
    
    def _ingest_text_file(self, file_path: Path) -> Dict[str, Any]:
        """
        Ingest a plain text file.
        
        Args:
            file_path: Path to the text file
            
        Returns:
            Dictionary with ingestion results
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Extract metadata from filename
            metadata = {
                "source": file_path.name,
                "file_type": "txt"
            }
            
            # Create document object
            document = {
                "content": content,
                "metadata": metadata
            }
            
            logger.info(f"Successfully ingested text file: {file_path}")
            self.stats["documents_ingested"] += 1
            
            return {"success": True, "document": document}
            
        except Exception as e:
            logger.error(f"Error ingesting text file: {e}")
            return {"success": False, "error": str(e)}
    
    def _ingest_csv_file(self, file_path: Path) -> Dict[str, Any]:
        """
        Ingest a CSV file, treating each row as a separate document.
        
        Args:
            file_path: Path to the CSV file
            
        Returns:
            Dictionary with ingestion results
        """
        try:
            df = pd.read_csv(file_path)
            
            # Find the column with the most text content
            text_column = self._identify_content_column(df)
            
            documents = []
            for _, row in df.iterrows():
                # Extract content from identified column
                content = str(row[text_column])
                
                # Extract metadata from other columns
                metadata = {
                    "source": file_path.name,
                    "file_type": "csv"
                }
                
                # Add all other columns as metadata
                for col in df.columns:
                    if col != text_column and not pd.isna(row[col]):
                        metadata[col] = str(row[col])
                
                documents.append({
                    "content": content,
                    "metadata": metadata
                })
            
            logger.info(f"Successfully ingested CSV file with {len(documents)} entries: {file_path}")
            self.stats["documents_ingested"] += len(documents)
            
            return {"success": True, "documents": documents}
            
        except Exception as e:
            logger.error(f"Error ingesting CSV file: {e}")
            return {"success": False, "error": str(e)}
    
    def _ingest_json_file(self, file_path: Path) -> Dict[str, Any]:
        """
        Ingest a JSON file.
        
        Args:
            file_path: Path to the JSON file
            
        Returns:
            Dictionary with ingestion results
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            documents = []
            
            # Handle different JSON structures
            if isinstance(data, list):
                # List of documents
                for item in data:
                    # Check if item has required fields
                    if isinstance(item, dict):
                        # Try to identify content field
                        content_field = self._identify_json_content_field(item)
                        if content_field:
                            content = item[content_field]
                            
                            # Use remaining fields as metadata
                            metadata = {
                                "source": file_path.name,
                                "file_type": "json"
                            }
                            
                            for key, value in item.items():
                                if key != content_field and isinstance(value, (str, int, float, bool)):
                                    metadata[key] = value
                            
                            documents.append({
                                "content": content,
                                "metadata": metadata
                            })
            elif isinstance(data, dict):
                # Single document or dictionary of documents
                for key, value in data.items():
                    if isinstance(value, str) and len(value) > 100:
                        # This looks like content
                        documents.append({
                            "content": value,
                            "metadata": {
                                "source": file_path.name,
                                "file_type": "json",
                                "key": key
                            }
                        })
                    elif isinstance(value, dict):
                        # Nested document
                        content_field = self._identify_json_content_field(value)
                        if content_field:
                            content = value[content_field]
                            
                            # Use remaining fields as metadata
                            metadata = {
                                "source": file_path.name,
                                "file_type": "json",
                                "document_id": key
                            }
                            
                            for k, v in value.items():
                                if k != content_field and isinstance(v, (str, int, float, bool)):
                                    metadata[k] = v
                            
                            documents.append({
                                "content": content,
                                "metadata": metadata
                            })
            
            if not documents:
                logger.warning(f"No valid documents found in JSON file: {file_path}")
                return {"success": False, "error": "No valid documents found"}
            
            logger.info(f"Successfully ingested JSON file with {len(documents)} entries: {file_path}")
            self.stats["documents_ingested"] += len(documents)
            
            return {"success": True, "documents": documents}
            
        except Exception as e:
            logger.error(f"Error ingesting JSON file: {e}")
            return {"success": False, "error": str(e)}
    
    def _ingest_pdf_file(self, file_path: Path) -> Dict[str, Any]:
        """
        Ingest a PDF file using unstructured.io, which provides advanced document parsing capabilities.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Dictionary with ingestion results
        """
        logger.info(f"Processing PDF with: {file_path}")

        try:
            # Extract elements from PDF
            elements = partition_pdf(
                file_path,
                # Additional parameters:
                extract_images_in_pdf=False,  # Set to True to extract images
                extract_tables=True,  # Extract tables from the PDF
                infer_table_structure=True,  # Try to infer structure of tables
                chunking_strategy="by_title"  # Chunk elements by title hierarchy
            )
            
            # Process different element types
            content_parts = []
            tables = []
            images = []
            metadata_parts = {}
            
            for element in elements:
                if hasattr(element, "category"):
                    # Add text with category context
                    element_text = str(element)
                    element_category = element.category
                    
                    if element_category == "Title":
                        content_parts.append(f"\n## {element_text}\n")
                    elif element_category == "NarrativeText":
                        content_parts.append(element_text)
                    elif element_category == "ListItem":
                        content_parts.append(f"- {element_text}")
                    elif element_category == "Table":
                        # Format the table as markdown table for better structure
                        formatted_table_text = self._format_table_as_markdown(element_text)
                        content_parts.append(f"\n{formatted_table_text}\n")
                        
                        if hasattr(element, "metadata") and element.metadata:
                            # Collect table metadata and save the table separately
                            table_metadata = element.metadata.__dict__ if hasattr(element.metadata, "__dict__") else element.metadata
                            tables.append({
                                "text": formatted_table_text,
                                "raw_text": element_text,
                                "metadata": table_metadata
                            })
                    elif element_category == "Image":
                        content_parts.append(f"\n[IMAGE: {element_text}]\n")
                        if hasattr(element, "metadata") and element.metadata:
                            images.append({
                                "text": element_text,
                                "metadata": element.metadata.__dict__ if hasattr(element.metadata, "__dict__") else element.metadata
                            })
                    else:
                        content_parts.append(element_text)
                    
                    # Collect metadata from elements
                    if hasattr(element, "metadata") and element.metadata:
                        metadata = element.metadata.__dict__ if hasattr(element.metadata, "__dict__") else element.metadata
                        for key, value in metadata.items():
                            if key not in metadata_parts:
                                metadata_parts[key] = value
            
            # Combine content parts
            content = "\n".join(content_parts)
            
            # Create consolidated metadata
            metadata = {
                "source": file_path.name,
                "file_type": "pdf",
                "has_tables": len(tables) > 0,
                "table_count": len(tables),
                "has_images": len(images) > 0,
                "image_count": len(images)
            }
            
            # Add extracted metadata
            metadata.update(metadata_parts)
            
            # Create document object with structured elements
            document = {
                "content": content,
                "metadata": metadata,
                "tables": tables,
                "images": images,
                "elements": [{"category": getattr(e, "category", "Unknown"), "text": str(e)} for e in elements]
            }
            
            logger.info(f"Successfully ingested PDF file using unstructured.io: {file_path}")
            self.stats["documents_ingested"] += 1
            
            return {"success": True, "document": document}
                
        except Exception as e:
            logger.error(f"Error ingesting PDF file with unstructured.io: {e}")
            return {"success": False, "error": str(e)}

    def _format_table_as_markdown(self, table_text: str) -> str:
        """
        Format a table as markdown for better structure and readability.
        
        Args:
            table_text: Raw table text extracted from unstructured.io
            
        Returns:
            Formatted markdown table
        """
        try:
            lines = table_text.strip().split('\n')
            if not lines:
                return table_text
            
            # Check if this is already formatted or just raw text
            if '|' in table_text:
                # Try to detect and clean up an existing table structure
                rows = [line.strip() for line in lines]
                
                # Add headers if missing
                if not rows[0].startswith('|'):
                    rows[0] = '| ' + rows[0] + ' |'
                
                # Add separator row after header if missing
                if len(rows) > 1 and not rows[1].startswith('|---'):
                    cols = rows[0].count('|') - 1
                    separator = '|' + '|'.join(['---' for _ in range(cols)]) + '|'
                    rows.insert(1, separator)
                
                # Format all rows consistently
                for i in range(len(rows)):
                    if i > 1 and not rows[i].startswith('|'):
                        rows[i] = '| ' + rows[i] + ' |'
                
                return '\n'.join(rows)
            else:
                # This seems to be raw text, try to infer table structure
                # Split by whitespace and hope for the best
                rows = []
                for line in lines:
                    if line.strip():
                        rows.append(line.split())
                
                if not rows:
                    return table_text
                
                # Determine the number of columns based on the first row
                num_cols = len(rows[0]) if rows else 0
                if num_cols == 0:
                    return table_text
                
                # Normalize all rows to have the same number of columns
                for i in range(len(rows)):
                    while len(rows[i]) < num_cols:
                        rows[i].append('')
                    rows[i] = rows[i][:num_cols]  # Truncate if too long
                
                # Build markdown table
                header = '| ' + ' | '.join(rows[0]) + ' |'
                separator = '|' + '|'.join(['---' for _ in range(num_cols)]) + '|'
                
                formatted_rows = [header, separator]
                for row in rows[1:]:
                    formatted_rows.append('| ' + ' | '.join(row) + ' |')
                
                return '\n'.join(formatted_rows)
        except Exception as e:
            logger.warning(f"Error formatting table as markdown: {e}")
            # Return original text if formatting fails
            return f"Table:\n{table_text}"
    
    def _identify_content_column(self, df: pd.DataFrame) -> str:
        """
        Identify which column in a DataFrame contains the main content.
        
        Args:
            df: Pandas DataFrame
            
        Returns:
            Name of the content column
        """
        # Look for columns with these names
        content_column_names = ["content", "text", "description", "abstract", "body"]
        
        for name in content_column_names:
            if name in df.columns:
                return name
        
        # If no standard content column found, look for the column with longest strings
        avg_lengths = {}
        for col in df.columns:
            if df[col].dtype == 'object':  # Only check string columns
                # Calculate average string length
                avg_length = df[col].astype(str).apply(len).mean()
                avg_lengths[col] = avg_length
        
        if avg_lengths:
            # Return column with longest average string length
            return max(avg_lengths.items(), key=lambda x: x[1])[0]
        
        # Fallback to first column
        return df.columns[0]
    
    def _identify_json_content_field(self, item: Dict) -> Optional[str]:
        """
        Identify which field in a JSON object contains the main content.
        
        Args:
            item: Dictionary representing a JSON object
            
        Returns:
            Name of the content field or None if not found
        """
        # Look for fields with these names
        content_field_names = ["content", "text", "description", "abstract", "body"]
        
        for name in content_field_names:
            if name in item and isinstance(item[name], str):
                return name
        
        # If no standard content field found, look for the field with longest string
        text_fields = {}
        for key, value in item.items():
            if isinstance(value, str) and len(value) > 50:
                text_fields[key] = len(value)
        
        if text_fields:
            # Return field with longest text
            return max(text_fields.items(), key=lambda x: x[1])[0]
        
        return None