import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from datetime import datetime



class Training:

    def __init__(self,dataloader,model,device):
        
        self.dataloader = dataloader
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(),lr = 1e-3)
        self.device = device
        self.loss = nn.BCEWithLogitsLoss(reduction='mean').to(device)

    def start(self):
        
        self.model.train()
        reconstruction_losses = []
        # start timer
        start_time = datetime.now()
        for batch,data in tqdm(enumerate(self.dataloader),total=len(self.dataloader)):
            
            batch_reconstruction = self.model(data)
            loss_for_batch = self.loss(batch_reconstruction,data)
            reconstruction_losses.append(np.round(loss_for_batch.item(),4))

            # reset graph gradients
            self.optimizer.zero_grad()

            # Backward pass: compute gradient of the loss with respect to model parameters
            loss_for_batch.backward()

            nn.utils.clip_grad_norm_(self.model.parameters(),max_norm = 1.0)

            # update network parameters
            self.optimizer.step()

            # print training progress each 1'000 mini-batches
            if batch % 1000 == 0:
                print(f'Batch {batch}, Loss {np.round(loss_for_batch.item(),4)}')
        return np.mean(reconstruction_losses)
        



