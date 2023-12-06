import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import precision_score, precision_recall_fscore_support, accuracy_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, MultiStepLR
from torch.utils.data import DataLoader, ConcatDataset, TensorDataset
from task1_2 import GetDataset, MultiplePCA

from task3 import Net  # Replace with the actual module where your model class is defined
from task1_2 import GetDataset
# Define your model architecture
model = Net()  # Replace with the actual class name of your model

# Load the saved model state dictionary
fold = 1  # Replace with the actual fold value
#30x300x300x1 sigmoid batch dropout 0.5
save_path = f'./best_models/model-100.pth'
model.load_state_dict(torch.load(save_path))

# If you're using GPU, move the model to the GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Set the model in evaluation mode (if needed)
model.eval()
X, y = GetDataset(False)
scaler = StandardScaler()
X = scaler.fit_transform(X)
#X = MultiplePCA(X, [10])[10]
X = Variable(torch.from_numpy(np.array(X).astype('float32')))
y = Variable(torch.from_numpy(np.array(y).astype('float32')))
test_outputs = model(X)
predicted_labels = (test_outputs > 0.5).float()

# Calculate accuracy
correct_predictions = (predicted_labels == y.view(-1, 1)).sum().item()
total_samples = len(y)
accuracy = correct_predictions / total_samples
print(f"Accuracy: {accuracy}")