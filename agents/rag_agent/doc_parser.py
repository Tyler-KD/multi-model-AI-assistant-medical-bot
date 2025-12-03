import os
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions, 
    TableFormerMode, 
    RapidOcrOptions, 
    smolvlm_picture_description
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc import PictureItem, TableItem

class MedicalDocParser:
    """
    Handles parsing of medical research documents using docling.
    """
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.logger.info("Medical Document Parser initialized!")

    def parse_document(
            self,
            document_path: str,
            output_dir: str,
            image_resolution_scale: float = 2.0,
            do_ocr: bool = True,
            do_tables: bool = True,
            do_formulas: bool = True,
            do_picture_desc: bool = False
        ) -> Tuple[Any, List[str]]:
        """
        Parse the document and extract structured content and images.
        
        Args:
            document_path: Path to the document to parse
            output_dir: Directory to save extracted images
            image_resolution_scale: Resolution scale for extracted images
            do_ocr: Enable OCR processing
            do_tables: Enable table structure extraction
            do_formulas: Enable formula enrichment
            do_picture_desc: Enable picture description generation
            
        Returns:
            Tuple containing (parsed_document, list_of_image_paths)
        """
        # Create output directory if it doesn't exist
        output_dir_path = Path(output_dir)
        output_dir_path.mkdir(parents=True, exist_ok=True)
        
        # Configure pipeline options
        pipeline_options = PdfPipelineOptions(
            generate_page_images=True,
            generate_picture_images=True,
            images_scale=image_resolution_scale,
            do_ocr=do_ocr,
            do_table_structure=do_tables,
            do_formula_enrichment=do_formulas,
            do_picture_description=do_picture_desc
        )
        
        # Set table structure mode
        pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE    # Can choose between FAST and ACCURATE
        
        # Initialize document converter
        converter = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
        )
        
        # Convert document
        conversion_res = converter.convert(document_path)
        
        # Get document filename
        doc_filename = conversion_res.input.file.stem
        
        # Save page images
        for page_no, page in conversion_res.document.pages.items():
            page_image_filename = output_dir_path / f"{doc_filename}-{page_no}.png"
            with page_image_filename.open("wb") as fp:
                page.image.pil_image.save(fp, format="PNG")
        
        # Save images of figures and tables
        table_counter = 0
        picture_counter = 0
        image_paths = []
        
        for element, _level in conversion_res.document.iterate_items():
            if isinstance(element, TableItem):
                table_counter += 1
                element_image_filename = output_dir_path / f"{doc_filename}-table-{table_counter}.png"
                with element_image_filename.open("wb") as fp:
                    element.get_image(conversion_res.document).save(fp, "PNG")
                    
            if isinstance(element, PictureItem):
                picture_path = f"{doc_filename}-picture-{picture_counter}.png"
                element_image_filename = output_dir_path / picture_path
                with element_image_filename.open("wb") as fp:
                    element.get_image(conversion_res.document).save(fp, "PNG")
                
                # Add path to the list of images
                image_paths.append(str(element_image_filename))
                picture_counter += 1
        
        # Extract images for summarization
        images = []
        for picture in conversion_res.document.pictures:
            ref = picture.get_ref().cref
            image = picture.image
            if image:
                images.append(str(image.uri))
        
        return conversion_res.document, images