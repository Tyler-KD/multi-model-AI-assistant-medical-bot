import os
import cv2
import torch
import logging
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from .model_download import download_model_checkpoint

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Device setup
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {DEVICE}")

class UNet(nn.Module):
    """U-Net model for image segmentation."""
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        # Contracting path (encoder)
        self.conv1 = nn.Conv2d(self.n_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(512, 1024, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Expansive path (decoder)
        self.upconv1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.conv6 = nn.Conv2d(1024, 512, kernel_size=3, padding=1)
        self.upconv2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv7 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv8 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.upconv4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv9 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.conv10 = nn.Conv2d(64, self.n_classes, kernel_size=1)

    def forward(self, x):
        """Forward pass of U-Net."""
        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.conv2(self.pool(x1)))
        x3 = F.relu(self.conv3(self.pool(x2)))
        x4 = F.relu(self.conv4(self.pool(x3)))
        x5 = F.relu(self.conv5(self.pool(x4)))

        x6 = F.relu(self.upconv1(x5))
        x6 = torch.cat([x4, x6], dim=1)
        x6 = F.relu(self.conv6(x6))
        x7 = F.relu(self.upconv2(x6))
        x7 = torch.cat([x3, x7], dim=1)
        x7 = F.relu(self.conv7(x7))
        x8 = F.relu(self.upconv3(x7))
        x8 = torch.cat([x2, x8], dim=1)
        x8 = F.relu(self.conv8(x8))
        x9 = F.relu(self.upconv4(x8))
        x9 = torch.cat([x1, x9], dim=1)
        x9 = F.relu(self.conv9(x9))
        x10 = self.conv10(x9)

        return x10


class SkinLesionSegmentation:
    """Handles skin lesion segmentation using a trained U-Net model."""
    
    def __init__(self, model_path):
        self.model_path = model_path
        self.device = DEVICE
        self.model = self._load_model()

    def _load_model(self):
        """Load the trained U-Net model."""
        try:
            # with safe_globals([UNet]):
            #     model = torch.load(self.model_path, weights_only=False, map_location=self.device)
            # Call this before using the model
            download_model_checkpoint('1rvn4ucOH6UBoNk-GB9bUWuGTLkNIVUf0', self.model_path)
            model = UNet(n_channels=3, n_classes=1).to(self.device)  # Explicitly initialize UNet
            # model.load_state_dict(torch.load(self.model_path, weights_only=False, map_location=self.device), strict=False)
            model.load_state_dict(torch.load(self.model_path, map_location=torch.device(self.device))['state_dict'])
            # model = torch.load(self.model_path, map_location=torch.device(DEVICE))
            model.eval()
            logger.info(f"Model loaded successfully from {self.model_path}")
            return model
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise e

    def _overlay_mask(self, img, mask, output_path):
        """Overlay the segmentation mask on the original image."""
        try:
            mask_stacked = np.stack((mask,) * 3, axis=-1)
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.axis("off")
            ax.imshow(img)
            ax.imshow(mask_stacked, alpha=0.4)
            # plt.savefig("overlayed_plot.png", bbox_inches="tight")
            plt.savefig(output_path, bbox_inches="tight")
            logger.info("Overlayed segmentation mask saved as 'overlayed_plot.png'")
            # return "overlayed_plot.png"
            return True
        except Exception as e:
            logger.error(f"Error generating overlay: {e}")
            raise e
    
    def predict(self, image_path, output_path):
        """Segment lesion in an image and return overlaid visualization."""
        try:
            img = cv2.imread(image_path, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0  # Normalize to [0,1]
            img_resized = cv2.resize(img, (256, 256))
            img_tensor = torch.Tensor(img_resized).unsqueeze(0).permute(0, 3, 1, 2).to(self.device)

            with torch.no_grad():
                generated_mask = self.model(img_tensor).squeeze().cpu().numpy()

            # Resize mask to match original image dimensions
            generated_mask_resized = cv2.resize(generated_mask, (img.shape[1], img.shape[0]))
            return self._overlay_mask(img, generated_mask_resized, output_path)

        except Exception as e:
            logger.error(f"Error during segmentation: {e}")
            raise e


# # Example Usage
# if __name__ == "__main__":
#     segmenter = SkinLesionSegmentation(model_path="./models/skin_lesion_segmentation.pth")
#     segmented_image = segmenter.predict("./images/ISIC_0020840.jpg", "./segmentation_plot.png")
#     logger.info(f"Segmentation completed. Output saved at: {segmented_image}")
