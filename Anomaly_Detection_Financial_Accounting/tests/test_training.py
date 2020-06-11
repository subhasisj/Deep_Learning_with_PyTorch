import unittest
import pandas as pd
from src.training import Training
from src.dataset import AnomalyDataset
from src.model import AutoEncoder
from src import config
import os

import torch

class TestTraining(unittest.TestCase):

    def setUp(self):
        self.df = pd.read_csv(os.path.join('tests','./data_for_testing.csv'))
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = AutoEncoder(self.df.shape[1])


    def test_training(self):
        print(self.df.head())
        train_dataset = AnomalyDataset(self.df.values)
        train_data_loader = torch.utils.data.DataLoader(train_dataset,batch_size=config.BATCH_SIZE,shuffle=True,num_workers=0)
        trainer = Training(train_data_loader,self.model,self.device)

        for i in range(5):
            reconstruction_loss_for_epoch = trainer.start()
            self.assertIsNotNone(reconstruction_loss_for_epoch)


if __name__ == '__main__':
    unittest.main()