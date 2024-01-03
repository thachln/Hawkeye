import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging

from config import setup_config
from dataset.dataset import FGDataset
from torchvision import transforms
from model.registry import MODEL
from utils import PerformanceMeter, TqdmHandler, AverageMeter, accuracy
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

class Tester(object):
    """Test a model from a config which could be a training config.
    """

    def __init__(self):
        self.config = setup_config()
        self.report_one_line = True
        self.logger = self.get_logger()

        # set device. `config.experiment.cuda` should be a list of gpu device ids, None or [] for cpu only.
        self.device = self.config.experiment.cuda if isinstance(self.config.experiment.cuda, list) else []
        if len(self.device) > 0:
            os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(i) for i in self.device])
            self.logger.info(f'Using GPU: {self.device}')
        else:
            self.logger.info(f'Using CPU!')
        # build dataloader and model
        self.transformer = self.get_transformer(self.config.dataset.transformer)
        self.collate_fn = self.get_collate_fn()
        self.dataset = self.get_dataset(self.config.dataset)
        self.dataloader = self.get_dataloader(self.config.dataset)
        self.logger.info(f'Building model {self.config.model.name} ...')
        self.model = self.get_model(self.config.model)
        self.model = self.to_device(self.model, parallel=True)
        self.logger.info(f'Building model {self.config.model.name} OK!')
        
        # build meters
        self.performance_meters = self.get_performance_meters()
        self.average_meters = self.get_average_meters()
        
        # Confusion matrix
        self.all_predicted_labels = []
        self.all_true_labels = []
        self.labels = ['Aedes', 'Anopheles', 'Culex']

    def get_logger(self):
        logger = logging.getLogger()
        logger.handlers = []
        logger.setLevel(logging.INFO)

        screen_handler = TqdmHandler()
        screen_handler.setFormatter(logging.Formatter('[%(asctime)s] %(message)s'))
        logger.addHandler(screen_handler)
        return logger

    def get_performance_meters(self):
        return {
            metric: PerformanceMeter() for metric in ['acc']
        }

    def get_average_meters(self):
        return {
            meter: AverageMeter() for meter in ['acc']
        }

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

    def get_transformer(self, config):
        return transforms.Compose([
            transforms.Resize(size=config.resize_size),
            transforms.CenterCrop(size=config.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])

    def get_collate_fn(self):
        return None

    def get_dataset(self, config):
        path = os.path.join(config.meta_dir, 'val.txt')
        return FGDataset(config.root_dir, path, transform=self.transformer)

    def get_dataloader(self, config):
        return DataLoader(self.dataset, config.batch_size, num_workers=config.num_workers, pin_memory=False,
                          shuffle=False, collate_fn=self.collate_fn)

    def to_device(self, m, parallel=False):
        if len(self.device) == 0:
            m = m.to('cpu')
        elif len(self.device) == 1 or not parallel:
            m = m.to(f'cuda:{self.device[0]}')
        else:
            m = m.cuda(self.device[0])
            m = torch.nn.DataParallel(m, device_ids=self.device)
        return m

    def get_model_module(self, model=None):
        """get `model` in single-gpu mode or `model.module` in multi-gpu mode.
        """
        if model is None:
            model = self.model
        if isinstance(model, torch.nn.DataParallel):
            return model.module
        else:
            return model

    def test(self):
        self.logger.info(f'Testing model from {self.config.model.load}')
        self.validate()
        self.performance_meters['acc'].update(self.average_meters['acc'].avg)
        self.report()
    
    def validate(self):
        self.model.train(False)
        with torch.no_grad():
            val_bar = tqdm(self.dataloader, ncols=100)
            for data in val_bar:
                
                logits, labels = self.batch_validate(data)
                predicted_labels = torch.argmax(logits, dim=1).cpu().numpy()  # Assuming logits are probabilities
                true_labels = labels.cpu().numpy()
                self.all_predicted_labels.extend(predicted_labels)
                self.all_true_labels.extend(true_labels)
                # calculate_and_plot_metrics(logits, labels)
                # Generate and plot confusion matrix after all predictions
                if len(self.all_predicted_labels) == len(self.dataset):  # Check if all data is processed
                    cm = confusion_matrix(self.all_true_labels, self.all_predicted_labels)
                    plt.figure(figsize=(8, 6))
                    plt.imshow(cm, cmap='Blues', vmin=0, vmax=len(self.dataset))

                    for i in range(cm.shape[0]):
                        for j in range(cm.shape[1]):
                            plt.text(j, i, str(cm[i, j]), ha='center', va='center')
                    # Set labels for x-axis and y-axis
                    plt.xticks(list(range(len(self.labels))), self.labels, rotation=45)
                    plt.yticks(list(range(len(self.labels))), self.labels)
                    plt.ylabel("True label",fontsize=12)
                    plt.title("Predict label")
                    
                    plt.show() 
                val_bar.set_description(f'Testing')
                val_bar.set_postfix(acc=self.average_meters['acc'].avg)

    def batch_validate(self, data):

        images, labels = self.to_device(data['img']), self.to_device(data['label'])
        logits = self.model(images)

        acc = accuracy(logits, labels, 1)
        
        self.average_meters['acc'].update(acc, images.size(0))
        return logits,labels

    def report(self):
        metric_str = '  '.join([f'{metric}: {self.performance_meters[metric].current_value:.2f}'
                                for metric in self.performance_meters])
        self.logger.info(metric_str)


if __name__ == '__main__':
    tester = Tester()
    tester.test()