import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
import time
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from torch.utils.data import TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from task1 import GetDataset

def ResetWeights(model):
    for layer in model.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.l1 = 100
        #self.l2 = 600
        #self.l3 = 600
        self.layers = nn.Sequential(
            nn.Linear(30, self.l1),
            nn.Sigmoid(),
            nn.Dropout(p=0.50),
            nn.BatchNorm1d(self.l1),
            #nn.Linear(self.l1, self.l2),
            #nn.Sigmoid(),
            #nn.Dropout(p=0.50),
            #nn.BatchNorm1d(self.l2),
            nn.Linear(self.l1, 1),
            nn.Dropout(p=0.50),
            nn.Sigmoid(),
        )
    def forward(self, x):
        return self.layers(x)

#Variables
K_FOLDS = 5
BATCH_SIZE = 100
NUM_EPOCHS = 100
LR = 0.5
DECIMALS = 3

if __name__ == '__main__':
    X, y = GetDataset(False)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X = Variable(torch.from_numpy(np.array(X).astype('float32')))
    y = Variable(torch.from_numpy(np.array(y).astype('float32')))
    dataset = TensorDataset(X,y)
    # For fold results
    results = {}
    # Set fixed random number seed
    torch.manual_seed(42)
    np.random.seed(42)
    times = []
    # Define the K-fold Cross Validator
    kfold = KFold(n_splits=K_FOLDS, shuffle=True, random_state=42)

    for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):
        t1 = time.time()
        # Print
        print(f'FOLD {fold}')
        print(f"Training size:{len(train_ids)}, Test size:{len(test_ids)}")
        print('--------------------------------')
        # Sample elements randomly from a given list of ids, no replacement.
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)
        # Define data loaders for training and testing data in this fold
        trainloader = torch.utils.data.DataLoader(
                        dataset, 
                        batch_size=BATCH_SIZE, sampler=train_subsampler)
        #Test doesn't use batch size
        testloader = torch.utils.data.DataLoader(
                        dataset,
                        batch_size = len(test_ids), sampler=test_subsampler)
        # Init the neural network
        model = Net()
        ResetWeights(model)
        # Configuration options
        criterion = nn.BCELoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=LR)
        # Run the training loop for defined number of epochs
        for epoch in range(NUM_EPOCHS):
            model.train()
            #Total loss of each epoch
            current_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                #Gets X, y
                inputs, targets = data
                #Calculates y_predicted
                outputs = model(inputs)
                #Calculates batch loss
                loss = criterion(outputs.reshape(-1,),targets)
                optimizer.zero_grad()
                #Back-propagates loss
                loss.backward()
                #Applies SGD
                optimizer.step()
                
                #Calls scheduler for possibly reducing LR
                
                current_loss += loss.item()
            if epoch % 10 == 0:
                #Every ten epochs prints the current loss
                print(f"Fold {fold}| epoch {epoch}| loss {current_loss}")

        # Process is complete.
        print('\nTraining process has finished. Saving trained model.')
        times.append(time.time()-t1)
        # Print about testing
        print('Starting testing')

        # Saving the model
        save_path = f'./model-fold-{fold}.pth'
        model_scripted = torch.jit.script(model) # Export to TorchScript
        model_scripted.save(save_path) # Save

        # Evaluation for this fold
        correct, total = 0, 0
        with torch.no_grad():
            # Iterate over the test data and generate predictions
            tp = 0.
            tn = 0.
            fp = 0.
            fn = 0.
            for i, data in enumerate(testloader, 0):
                # Get inputs
                inputs, targets = data
                model.eval()
                test_outputs = model(inputs)
                predicted_labels = (test_outputs > 0.5).float()
                targets = targets.view(-1, 1)
                # Calculate accuracy
                #correct_predictions = (predicted_labels == targets).sum().item()

                #Manually calculates accuracy,precision, recall, f1
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
                            
            total_samples = tp+tn+fp+fn
            accuracy = round((tp+tn)/(tp+tn+fp+fn), DECIMALS)
            precision = 0.
            if tp+fp != 0:
                precision = round(tp / (tp+fp), DECIMALS)
            recall = 0.
            if tp+fn:
                recall = round(tp / (tp+fn), DECIMALS)
            f1 = 0
            if precision != 0 or recall != 0:
                f1 = round(2 * (precision*recall) / (precision+ recall),DECIMALS)
            a = round(accuracy * 100, DECIMALS)
            p = round(precision * 100, DECIMALS)
            r = round(recall * 100, DECIMALS)
            f1 = round(f1 * 100, DECIMALS)
            print(f'Accuracy on test set: {a}%')
            print(f'Precision on test set: {p}%')
            print(f'Recall on test set: {r}%')
            print(f'F1 on test set: {f1}%')
            print(f"{a}% {p}% {r}% {f1}%")
            results[fold] =  {'a': a, 'p' : p, 'r' : r, 'f1' : f1}
    
    
    #Average time 10.51s
    print(f"\n\nAverage training time {np.average(times)}\n")


    # Print folds results
    print(f'K-FOLD CROSS VALIDATION RESULTS FOR {K_FOLDS} FOLDS')
    print('--------------------------------')
    #Averages results between folds
    sum_a = 0.0
    sum_p = 0.0
    sum_r = 0.0
    sum_f1 = 0.0
    for key, value in results.items():
        sum_a += value['a']
        sum_p += value['p']
        sum_r += value['r']
        sum_f1 += value['f1']
    a = round(sum_a/len(results.items()), DECIMALS)
    p = round(sum_p/len(results.items()), DECIMALS)
    r = round(sum_r/len(results.items()), DECIMALS)
    f1 = round(sum_f1/len(results.items()), DECIMALS)
    print(f'Average accuracy: {a} %')
    print(f'Average precision: {p} %')
    print(f'Average recall: {r} %')
    print(f'Average f1: {f1}%')
    print(f"{a}% {p}% {r}% {f1}%")
    
    print(model)