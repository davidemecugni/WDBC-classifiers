import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
from torch.autograd import Variable
import numpy as np
import random
import csv
from task1_2 import GetDataset
DECIMALS = 5
models = []
for i in range(5):
    m = torch.jit.load(f"model-fold-{i}.pth")
    models.append(m)
X, y = GetDataset(False)
scaler = StandardScaler()
X = scaler.fit_transform(X)
#X = MultiplePCA(X, [10])[10]
X = Variable(torch.from_numpy(np.array(X).astype('float32')))
y = Variable(torch.from_numpy(np.array(y).astype('float32')))
y_pred = []
for model in models : 
    model.eval()
    pred = model(X)
    pred = (pred > 0.5).float().view(-1,1)
    y_pred.append(pred)

def GetGroupOpinion(i):
    opinions = []
    for pred in y_pred:
        opinions.append(pred[i].item())
    zero = 0
    one = 0
    for opinion in opinions:
        if opinion == 1.:
            one += 1
        else:
            zero += 1
    if one > zero:
        return 1.
    return 0.

final_prediction = [GetGroupOpinion(i) for i in range(len(y))] 
final_prediction = Variable(torch.from_numpy(np.array(final_prediction).astype('float32')))
correct_predictions = (final_prediction == y).sum().item()
tp = 0.
tn = 0.
fp = 0.
fn = 0.
for i in range(len(y)):
    for i in range(len(final_prediction)):
        if final_prediction[i] == 1.:
            #pred 1 targ 1
            if y[i] == 1.:
                tp +=1.
            #pred 1 targ 0
            else:
                fp += 1.
        else:
            #pred 0 targ 1
            if y[i] == 1.:
                fn+= 1.
            #pred 0 targ 0
            else:
                tn += 1.
total_samples = len(y)
accuracy = round(correct_predictions / total_samples, DECIMALS)
precision = round(tp / (tp+fp), DECIMALS)
recall = round(tp / (tp+fn), DECIMALS)
f1 = round(2 * (precision*recall) / (precision+ recall),DECIMALS)
print(f'Accuracy on test set: {accuracy*100}%')
print(f'Precision on test set: {precision*100}%')
print(f'Recall on test set: {recall*100}%')
print(f'F1 on test set: {f1}')
