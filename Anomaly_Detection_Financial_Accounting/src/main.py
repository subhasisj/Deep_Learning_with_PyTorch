import torch
import torch.nn as nn
from model import AutoEncoder
import config
from training import Training
from dataset import AnomalyDataset
import pandas as pd 
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt 

def main():

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # read data
    df = pd.read_csv(os.path.join('Data','categorical_and_numericals.csv'))
    df_train,df_test = train_test_split(df,test_size=0.1)
    df_train = df_train.reset_index(drop=True)

    train_dataset = AnomalyDataset(df_train.values)
    test_dataset = AnomalyDataset(df_test.values)
    train_data_loader = torch.utils.data.DataLoader(train_dataset,batch_size=config.BATCH_SIZE,shuffle=True,num_workers=0)
    test_data_loader = torch.utils.data.DataLoader(test_dataset,batch_size=config.BATCH_SIZE,shuffle=True,num_workers=0)


    model = AutoEncoder(df.shape[1])
    model.to(device)

    trainer = Training(train_data_loader,model,device)
    epoch_losses = []
    for epoch in range(config.EPOCHS):
        print(f'Epoch: {epoch+1} / {config.EPOCHS}:')
        try:
            reconstruction_loss_for_epoch = trainer.start()
            epoch_losses.append(reconstruction_loss_for_epoch)
        except Exception as e:
            print(f'Exception occurred during training at epoch {epoch+1} because {str(e)}')
            break
       
    # Save the loss history plot
    plt.plot(range(0, len(epoch_losses)), epoch_losses)        
    plt.xlabel('[training epoch]')
    plt.xlim([0, len(epoch_losses)])
    plt.ylabel('[reconstruction-error]')
    plt.title('AENN training performance')
    plt.savefig('./Plots/training_loss.png')

if __name__ == "__main__":
    main()