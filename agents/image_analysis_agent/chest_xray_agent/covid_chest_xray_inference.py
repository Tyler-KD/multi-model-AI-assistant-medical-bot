import logging
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

class ChestXRayClassification:
    def __init__(self, model_path, device=None):
        # Configure logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)

        self.class_names = ['covid19', 'normal']
        self.device = device if device else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # print(f"Using device: {self.device}")
        self.logger.info(f"Using device: {self.device}")
        
        # Load model
        self.model = self._build_model(weights=None)
        self._load_model_weights(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        # Image transformations
        self.mean_nums = [0.485, 0.456, 0.406]
        self.std_nums = [0.229, 0.224, 0.225]
        self.transform = transforms.Compose([
            transforms.Resize((150, 150)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean_nums, std=self.std_nums)
        ])
    
    def _build_model(self, weights=None):
        """Initialize the DenseNet model with custom classification layer."""
        model = models.densenet121(weights=None)
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, len(self.class_names))
        return model
    
    def _load_model_weights(self, model_path):
        """Load pre-trained model weights."""
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            # print(f"Model loaded successfully from {model_path}")
            self.logger.info(f"Model loaded successfully from {model_path}")
        except Exception as e:
            # print(f"Error loading model: {e}")
            self.logger.error(f"Error loading model: {e}")
            raise e
    
    def predict(self, img_path):
        """Predict the class of a given image."""
        try:
            image = Image.open(img_path).convert("RGB")
            image_tensor = self.transform(image).unsqueeze(0)
            input_tensor = Variable(image_tensor).to(self.device)
            
            with torch.no_grad():
                out = self.model(input_tensor)
                _, preds = torch.max(out, 1)
                idx = preds.cpu().numpy()[0]
                pred_class = self.class_names[idx]
                
            # # Display Image
            # plt.imshow(np.array(image))
            # plt.title(f"Predicted: {pred_class}")
            # plt.show()

            self.logger.info(f"Predicted Class: {pred_class}")
            
            return pred_class
        except Exception as e:
            self.logger.error(f"Error during prediction Covid Chest X-ray: {str(e)}")
            return None

# if __name__ == "__main__":
#     classifier = ChestXRayClassification('./models/covid_chest_xray_model.pth')
#     predicted_class = classifier.predict('./images/NORMAL2-IM-0362-0001.jpeg')
#     print(f"PREDICTED CLASS: {predicted_class}")
