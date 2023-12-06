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

def reset_weights(m):
  '''
    Try resetting model weights to avoid
    weight leakage.
  '''
  for layer in m.children():
    if hasattr(layer, 'reset_parameters'):
        layer.reset_parameters()

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(30, 600),
            nn.Sigmoid(),
            nn.Dropout(p=0.50),
            nn.BatchNorm1d(600),
            nn.Linear(600, 1200),
            nn.Sigmoid(),
            nn.Dropout(p=0.50),
            nn.BatchNorm1d(1200),
            nn.Linear(1200, 1),
            nn.Sigmoid(),
        )
    def forward(self, x):
        return self.layers(x)
    def num_flat_features(self, x):
        # all dimensions except the batch dimension
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
    
K_FOLDS = 5
BATCH_SIZE = 40
#50 94.796
NUM_EPOCHS = 200
LR = 0.06
MOMENTUM = 0
FACTOR = 0.5
PATIENCE = (BATCH_SIZE * NUM_EPOCHS) / 10
DECIMALS = 4
if __name__ == '__main__':
    X, y = GetDataset(False)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    #X = MultiplePCA(X, [10])[10]
    X = Variable(torch.from_numpy(np.array(X).astype('float32')))
    y = Variable(torch.from_numpy(np.array(y).astype('float32')))
    dataset = TensorDataset(X,y)
    # Configuration options
    k_folds = K_FOLDS
    batch_size = BATCH_SIZE
    num_epochs = NUM_EPOCHS
    criterion = nn.BCELoss()
    # For fold results
    results = {}
    # Set fixed random number seed
    torch.manual_seed(42)
    np.random.seed(42)
     # Define the K-fold Cross Validator
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):
        # Print
        print(f'FOLD {fold}')
        print('--------------------------------')
        # Sample elements randomly from a given list of ids, no replacement.
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)
        # Define data loaders for training and testing data in this fold
        trainloader = torch.utils.data.DataLoader(
                        dataset, 
                        batch_size=batch_size, sampler=train_subsampler)
        testloader = torch.utils.data.DataLoader(
                        dataset,
                        batch_size=batch_size, sampler=test_subsampler)
        # Init the neural network
        model = Net()
        #model.apply(reset_weights)

        # Initialize optimizer
        criterion = nn.BCELoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=LR)
        scheduler = ReduceLROnPlateau(optimizer, "min", FACTOR, PATIENCE, verbose=True)
        print("Epoch: ", end = "")
        # Run the training loop for defined number of epochs
        for epoch in range(0, num_epochs):
            model.train()
            current_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                inputs, targets = data
                outputs = model(inputs)
                loss = criterion(outputs.reshape(-1,),targets)
                optimizer.zero_grad()
                loss.backward()
                
                optimizer.step()
                scheduler.step(loss)
                current_loss += loss.item()
            if epoch % 10 == 0:
                print(current_loss)
        # Process is complete.
        print('\nTraining process has finished. Saving trained model.')

        # Print about testing
        print('Starting testing')

        # Saving the model
        save_path = f'./model-fold-{fold}.pth'
        model_scripted = torch.jit.script(model) # Export to TorchScript
        model_scripted.save(save_path) # Save

        # Evaluationfor this fold
        correct, total = 0, 0
        with torch.no_grad():
            # Iterate over the test data and generate predictions
            a = []
            p = []
            r = []
            f1_total = []
            for i, data in enumerate(testloader, 0):
                # Get inputs
                inputs, targets = data
                model.eval()
                test_outputs = model(inputs)
                predicted_labels = (test_outputs > 0.5).float()
                targets = targets.view(-1, 1)
                # Calculate accuracy
                correct_predictions = (predicted_labels == targets).sum().item()
                tp = 0.
                tn = 0.
                fp = 0.
                fn = 0.
                for i in range(len(predicted_labels)):
                    if predicted_labels[i] == 1.:
                        #pred 1 targ 1
                        if targets[i] == 1.:
                            tp +=1.
                        #pred 1 targ 0
                        else:
                            fp += 1.
                    else:
                        #pred 0 targ 1
                        if targets[i] == 1.:
                            fn+= 1.
                        #pred 0 targ 0
                        else:
                            tn += 1.
                total_samples = len(targets)
                accuracy = round(correct_predictions / total_samples, DECIMALS)
                precision = round(tp / (tp+fp), DECIMALS)
                recall = round(tp / (tp+fn), DECIMALS)
                f1 = round(2 * (precision*recall) / (precision+ recall),DECIMALS)
                a.append(accuracy)
                p.append(precision)
                r.append(recall)
                f1_total.append(f1)
            a = round(np.average(a) * 100, DECIMALS)
            p = round(np.average(p) * 100, DECIMALS)
            r = round(np.average(r) * 100, DECIMALS)
            f1_total = round(np.average(f1_total), DECIMALS)
            print(f'Accuracy on test set: {a}%')
            print(f'Precision on test set: {p}%')
            print(f'Recall on test set: {r}%')
            print(f'F1 on test set: {f1_total}')
            results[fold] =  {'a': a, 'p' : p, 'r' : r, 'f1' : f1_total}
    # Print fold results
    print(f'K-FOLD CROSS VALIDATION RESULTS FOR {k_folds} FOLDS')
    print('--------------------------------')
    sum_a = 0.0
    sum_p = 0.0
    sum_r = 0.0
    sum_f1 = 0.0
    for key, value in results.items():
        sum_a += value['a']
        sum_p += value['p']
        sum_r += value['r']
        sum_f1 += value['f1']
    print(f'Average accuracy: {round(sum_a/len(results.items()), DECIMALS)} %')
    print(f'Average precision: {round(sum_p/len(results.items()), DECIMALS)} %')
    print(f'Average recall: {round(sum_r/len(results.items()), DECIMALS)} %')
    print(f'Average f1: {round(sum_f1/len(results.items()), DECIMALS)}')
    print(results)