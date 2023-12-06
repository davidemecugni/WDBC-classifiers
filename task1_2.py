import csv
import random
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.metrics import precision_score, precision_recall_fscore_support, accuracy_score
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
import time
def GetDataset(split_train_test = False, seed=314):
    """Returns dataset"""
    random.seed(seed)
    with open("wdbc.data") as f:
        data = [row for row in csv.reader(f)]
    len_data = len(data)
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

def MultipleKNN(X_training,y_training, X_test, y_test, neighbours):
    """
    Calculates accuracy, precision, recall, f1 for different k
    """
    results = {}
    for n in neighbours:
        partial = {}
        knn = KNeighborsClassifier(n)
        knn.fit(X_training, y_training)
        y_pred = knn.predict(X_test)
        partial["accuracy"] = accuracy_score(y_test, y_pred)
        partial["precision"], partial["recall"], partial["f1"], i_ =  precision_recall_fscore_support(y_test,y_pred,average="binary", pos_label=1)
        results[n] = partial
    return results

def ResultValues(dim, metric):
    return [v[metric] for v in results[dim].values()]

def GraphSingleData(metric):
    for d in dim:
        plt.plot(k,ResultValues(d, metric),label = f"{d} dimensions")
    plt.xlabel("Number of k-neighbours")
    plt.ylabel(metric.capitalize())
    plt.legend()
    plt.show()

def TabData():
    metrics = ["accuracy", "precision", "recall", "f1"]
    nx = len(dim)+1
    ny = len(metrics)+1
    for d in dim:
        pl.figure(figsize=(8, 2))
        header = k.copy()
        header.insert(0," ")
        tab_data = [header]
        for m in metrics:
            x = [round(i,3) for i in ResultValues(d, m)]
            x.insert(0, m)
            tab_data.append(x)
        pl.table(cellText=np.array(tab_data), loc=(0,0), cellLoc='center')

        ax = pl.gca()
        ax.set_xticks([])
        ax.set_yticks([])
        pl.title(f"Statistics for {d} dimentions")
        plt.show()
def GraphData():
    GraphSingleData("accuracy")
    GraphSingleData("precision")
    GraphSingleData("recall")
    GraphSingleData("f1")
if __name__ == "__main__":
    t1 = time.time()
    #Get data
    X_training, y_training, X_test, y_test = GetDataset(split_train_test=True)
    dim = [5,10,15,20]
    k = [1,2,3,4,5,6,7,8,9]
    #Reduced dataset
    reduced = MultiplePCA(X_training, dim, X_test = X_test)
    #dic dim:scores
    results = {}
    #Trains KNN and saves results
    for dimention, dataset in reduced.items():
        X_tr, X_t = dataset
        res = MultipleKNN(X_tr,y_training, X_t,y_test, k)
        results[dimention] = res

    #Add full dimention KNN analysis
    results[30] = MultipleKNN(X_training, y_training, X_test, y_test,k)
    print(f"Total time: {time.time()-t1}")
    dim.append(30)
    GraphData()
    TabData()

