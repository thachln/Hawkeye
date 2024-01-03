import os
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import logging
from config import setup_config
from model.registry import MODEL
from utils import TqdmHandler
from torchvision import transforms


class Predictor():
    def __init__(self):
        self.config = setup_config()
        self.logger = self.get_logger()  # Assuming you have a logger setup function

        # set device. `config.experiment.cuda` should be a list of gpu device ids, None or [] for cpu only.
        self.device = self.config.experiment.cuda if isinstance(self.config.experiment.cuda, list) else []
        if len(self.device) > 0:
            os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(i) for i in self.device])
            self.logger.info(f'Using GPU: {self.device}')
        else:
            self.logger.info(f'Using CPU!')

        self.logger.info(f'Predict model {self.config.model.name} ...')
        self.model = self.get_model(self.config.model)
        self.model = self.to_device(self.model, parallel=False)
        self.logger.info(f'Predict model {self.config.model.name} OK!')
        self.model.eval()

        self.transformer = self.get_transformer(self.config.dataset.transformer)
        self.labels = ['Aedes', 'Anopheles', 'Culex']

    def get_logger(self):
        logger = logging.getLogger()
        logger.handlers = []
        logger.setLevel(logging.INFO)

        screen_handler = TqdmHandler()
        screen_handler.setFormatter(logging.Formatter('[%(asctime)s] %(message)s'))
        logger.addHandler(screen_handler)
        return logger
    
    def get_model(self, config):
        """Build and load model in config
        """
        name = config.name
        model = MODEL.get(name)(config)

        assert 'load' in config and config.load != '', 'There is no valid `load` in config[model.load]!'
        self.logger.info(f'Loading model from {config.load}')
        state_dict = torch.load(config.load, map_location='cpu')
        model.load_state_dict(state_dict)
        self.logger.info(f'OK! Model loaded from {config.load}')
        return model
    
    def to_device(self, m, parallel=False):
        if len(self.device) == 0:
            m = m.to('cpu')
        elif len(self.device) == 1 or not parallel:
            m = m.to(f'cuda:{self.device[0]}')
        else:
            m = m.cuda(self.device[0])
            m = torch.nn.DataParallel(m, device_ids=self.device)
        return m
    
    def get_transformer(self, config):
        return transforms.Compose([
            transforms.Resize(size=config.resize_size),
            transforms.CenterCrop(size=config.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            ])
        
    #
    
    def predict(self, image_path):
        image = Image.open(image_path).convert('RGB')
        image = self.transformer(image).unsqueeze(0)  # Add batch dimension
        image = self.to_device(image)
        # print(np.max(image))
        # print(np.min(image))
        # image = (image / 255.0)

        with torch.no_grad():
            logits = self.model(image)
            # Get probabilities for each class
            probabilities = torch.nn.functional.softmax(logits, dim=0).cpu().numpy()
            predicted_label = torch.argmax(logits, dim=-1).item()

        # Display the image
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.imshow(image.squeeze(0).permute(2,1,0).cpu().numpy())  # Rearrange for Matplotlib
        # )
        plt.title(f"Predicted class: {self.labels[predicted_label]} (Probability: {probabilities[predicted_label]:.2%}) ")

        # Display a bar chart of the prediction probabilities
        plt.subplot(1, 2, 2)
        plt.bar(np.arange(len(probabilities)), probabilities)
        plt.xticks(np.arange(len(probabilities)), self.labels, rotation='vertical')
        plt.title("Prediction probabilities for each class")

        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    predictor = Predictor()
    image_path = 'predict_image/1.jpg'  # Replace with the actual image path
    predictor.predict(image_path)