"""
Uses an Ensemble network of 5 FCNN to reach a majority vote on the binary classification 
"""
import time
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
from torch.autograd import Variable
import numpy as np
from task1 import GetDataset
DECIMALS = 5

def GetGroupOpinion(i):
    """
    Gets the majority vote out of the predictions made by the models
    """
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




if __name__ == '__main__':
    #Contains the 5 models retrieved from the file
    models = []
    #Retrieves 5 MLP generated in task 3
    for i in range(5):
        m = torch.jit.load(f"model-fold-{i}.pth")
        models.append(m)

    #Gets the dataset, scales it and saves it as a torch variable
    X, y = GetDataset(False)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X = Variable(torch.from_numpy(np.array(X).astype('float32')))
    y = Variable(torch.from_numpy(np.array(y).astype('float32')))
    #Contains the prediction made by the single network
    t1 = time.time()
    y_pred = []
    for model in models : 
        model.eval()
        pred = model(X)
        pred = (pred > 0.5).float().view(-1,1)
        y_pred.append(pred)

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
    #(tp+tn)/(tp+tn+fp+fn) could also be used
    accuracy = round(correct_predictions / total_samples, DECIMALS)
    precision = round(tp / (tp+fp), DECIMALS)
    recall = round(tp / (tp+fn), DECIMALS)
    f1 = round(2 * (precision*recall) / (precision+ recall),DECIMALS)
    print(f"Time for classification: {time.time()-t1}")
    print("Ensemble network metrics")
    print(f'Accuracy on test set: {accuracy*100}%')
    print(f'Precision on test set: {precision*100}%')
    print(f'Recall on test set: {recall*100}%')
    print(f'F1 on test set: {f1}')
