import csv
import random
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.metrics import precision_score, precision_recall_fscore_support, accuracy_score


def GetDataset(split_train_test = False, seed=314):
    """Returns dataset"""
    random.seed(seed)
    with open("wdbc.data") as f:
        data = [row for row in csv.reader(f)]
    len_data = len(data)
    fields = len(data[0])
    random_placement = random.sample(range(len_data), 169)

    classify = {"B":0, "M":1}
    if split_train_test:
        X_training, X_test = [],[] 
        y_training, y_test = [],[] 
        for i in range(len_data):
            to_float = [float(x) for x in data[i][2:]]
            if i in random_placement:
                X_test.append(to_float)
                y_test.append(classify[data[i][1]])
            else:
                X_training.append(to_float)
                y_training.append(classify[data[i][1]])
        return X_training,y_training,X_test,y_test
    else:
        X,y = [],[]
        for i in range(len_data):
            to_float = [float(x) for x in data[i][2:]]
            X.append(to_float)
            y.append(classify[data[i][1]])
        return X,y
def MultiplePCA(X, dimentions, X_test = None):
    """Returns a dict nr_of_dim:reduced_dataset"""
    d = {}
    for n_of_elements in dimentions:
        pca = PCA(n_components=n_of_elements)
        X_red = pca.fit_transform(X)
        if X_test:
            X_test_red = pca.fit_transform(X_test)
            d[n_of_elements] = X_red, X_test_red
        else:
            d[n_of_elements] = X_red
    return d

def MultipleKNN(X,y, X_test, y_test, neighbours):
    """
    Calculates accuracy, precision, recall, f1 for different k
    """
    results = {}
    for n in neighbours:
        partial = {}
        knn = KNeighborsClassifier(n)
        knn.fit(X, y)
        y_pred = knn.predict(X_test)
        partial["accuracy"] = accuracy_score(y_test, y_pred)
        partial["precision"], partial["recall"], partial["f1"], i_ =  precision_recall_fscore_support(y_test,y_pred,average="binary", pos_label=1)
        results[n] = partial
    return results
#X_training,y_training,X_test,y_test = GetDataset() 
X, y, X_test, y_test = GetDataset(split_train_test=True)
reduced = MultiplePCA(X,[1,2,5,10,15,20,29], X_test = X_test)
l = {}
k = [1,3,5,7,9]
for dimention, dataset in reduced.items():
    X_tr, X_t = dataset
    res = MultipleKNN(X_tr,y, X_t,y_test, k)
    l[dimention] = res
l[30] = MultipleKNN(X, y, X_test, y_test,k)

for k,v in l.items():
    print(f"Dimentions: {k}----")
    print(v)