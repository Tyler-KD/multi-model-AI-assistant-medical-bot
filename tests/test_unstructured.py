import logging
from pathlib import Path
from unstructured.partition.pdf import partition_pdf

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_pdf_processing(pdf_path):
    """
    Test PDF processing with unstructured.io
    
    Args:
        pdf_path: Path to a PDF file for testing
    """
    logger.info(f"Testing PDF processing with: {pdf_path}")
    
    try:
        # Process the PDF
        logger.info("Calling partition_pdf...")
        elements = partition_pdf(
            pdf_path,
            extract_images_in_pdf=False,
            extract_tables=True,
            infer_table_structure=True,
            chunking_strategy="by_title"
        )
        
        # Print results
        logger.info(f"Successfully extracted {len(elements)} elements from the PDF")
        logger.info("Sample of extracted elements:")
        for i, element in enumerate(elements[:5]):  # Show first 5 elements
            logger.info(f"Element {i+1}, Type: {type(element).__name__}, Category: {getattr(element, 'category', 'Unknown')}")
            logger.info(f"Content: {str(element)[:100]}...")  # Show first 100 chars
        
        logger.info("PDF processing test completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error processing PDF: {e}")
        return False

if __name__ == "__main__":
    # Replace with path to a sample PDF on your system
    pdf_path = "data/raw/deep_learning_based_brain_tumor_segmentation_a_survey.pdf"
    test_pdf_processing(pdf_path)