import os

DATASET = os.path.join('Anomaly_Detection_Financial_Accounting','Data','categorical_and_numericals.csv') 
EPOCHS = 30
MODEL_SAVE_PATH = os.path.join('Anomaly_Detection_Financial_Accounting','Saved_Models','Autoencoder.pth')
BATCH_SIZE = 128