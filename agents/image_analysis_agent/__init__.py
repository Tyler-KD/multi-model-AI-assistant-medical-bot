from .image_analyzer import ImageAnalyzer

class ImageAnalysisAgent:
    """
    Agent responsible for processing image uploads and classifying them as medical or non-medical, and determining their type.
    """
    
    def __init__(self):
        self.image_analyzer = ImageAnalyzer()
    
    def process_image(self, image_path: str) -> str:
        """Classifies images as medical or non-medical and determines their type."""
        return self.image_analyzer.classify_image(image_path)