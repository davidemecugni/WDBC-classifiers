import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
from task1_2 import GetDataset
# Generate a synthetic dataset with 30 features
X, y = GetDataset(False)
X = np.array(X)
y = np.array(y)
#X, y = make_classification(n_samples=1000, n_features=30, n_classes=2, random_state=42)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the input features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# Convert to PyTorch tensors
X_train_tensor = torch.from_numpy(X_train.astype(np.float32))
y_train_tensor = torch.from_numpy(y_train.astype(np.float32))
X_test_tensor = torch.from_numpy(X_test.astype(np.float32))
y_test_tensor = torch.from_numpy(y_test.astype(np.float32))

# Define the neural network
class BinaryClassifier(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(BinaryClassifier, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size1)
        self.relu1 = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size1, hidden_size2)
        self.relu2 = nn.ReLU()
        self.layer3 = nn.Linear(hidden_size2, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu1(x)
        x = self.layer2(x)
        x = self.relu2(x)
        x = self.layer3(x)
        x = self.sigmoid(x)
        return x

# Set random seed for reproducibility
torch.manual_seed(42)

# Define hyperparameters
input_size = 30
hidden_size1 = 300
hidden_size2 = 300
output_size = 1
learning_rate = 0.05
epochs = 1000

# Instantiate the model
model = BinaryClassifier(input_size, hidden_size1, hidden_size2, output_size)

# Define loss function and optimizer
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(epochs):
    # Forward pass
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor.view(-1, 1))

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Print progress
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# Test the model
with torch.no_grad():
    model.eval()
    test_outputs = model(X_test_tensor)
    predicted_labels = (test_outputs > 0.5).float()

    # Calculate accuracy
    correct_predictions = (predicted_labels == y_test_tensor.view(-1, 1)).sum().item()
    total_samples = len(y_test)
    accuracy = correct_predictions / total_samples
    print(f"Accuracy: {accuracy}")
