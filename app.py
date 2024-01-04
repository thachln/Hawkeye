import os
import logging
import streamlit as st
from PIL import Image
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms


from config import setup_config
from model.registry import MODEL
from utils import TqdmHandler

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


def main():
    st.title("Image Predictor App")

    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)

        # Make predictions
        predictor = Predictor()
        image = Image.open(uploaded_file).convert('RGB')
        image = predictor.transformer(image).unsqueeze(0)  # Add batch dimension
        image = predictor.to_device(image)

        with torch.no_grad():
            logits = predictor.model(image)
            probabilities = torch.nn.functional.softmax(logits, dim=0).cpu().numpy()
            predicted_label = torch.argmax(logits, dim=-1).item()

        # Display predictions
        st.subheader("Prediction:")
        st.write(f"Predicted class: {predictor.labels[predicted_label]}")
        st.write(f"Probability: {probabilities[predicted_label]:.2%}")

        # Display the image and bar chart
        st.subheader("Image and Prediction Probabilities:")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

        ax1.imshow(image.squeeze(0).permute(2, 1, 0).cpu().numpy())
        ax1.set_title(f"Predicted class: {predictor.labels[predicted_label]}")

        ax2.bar(np.arange(len(probabilities)), probabilities)
        ax2.set_xticks(np.arange(len(probabilities)))
        ax2.set_xticklabels(predictor.labels, rotation='vertical')
        ax2.set_title("Prediction probabilities for each class")

        st.pyplot(fig)

if __name__ == "__main__":
    main()