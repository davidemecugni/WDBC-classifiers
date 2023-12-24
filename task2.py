from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
import time
from sklearn.preprocessing import StandardScaler

from task1 import GetDataset, MultiplePCA

def MultipleKNN(X_training,y_training, X_test, y_test, neighbours):
    """
    Calculates accuracy, precision, recall, f1 for different k
    """
    #Dict of k_neighbours:{metric:value}
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

def ResultValues(dim, metric, folds_results):
    """
    Gets the metric value(es. acc/f1...) for a single reduced dimention dataset
    """
    l = []
    for i in range(5):
        l.append([v[metric] for v in folds_results[i][dim].values()])
    l = np.mean(l, axis=0 )
    return l

def GraphSingleData(metric,folds_results):
    """
    Graphs the data for a single metric.
    The resulting table has (dimentions) x (value of the metric)
    """

    for d in dim:
        plt.plot(k,ResultValues(d, metric, folds_results),label = f"{d} dimensions")
    plt.xlabel("Number of k-neighbours")
    plt.ylabel(metric.capitalize())
    plt.legend()
    plt.show()

def TabData(folds_results):
    """
    Graphs a table for a single reduced dimenction dataset
    """
    metrics = ["accuracy", "precision", "recall", "f1"]
    nx = len(dim)/2+1
    ny = len(metrics)+1
    for d in dim:
        pl.figure(figsize=(8, 2))
        header = k[::2].copy()
        header.insert(0," ")
        tab_data = [header]
        for m in metrics:
            r = ResultValues(d, m,folds_results)
            r = r[::2]
            x = [round(i,4) for i in r]
            x.insert(0, m)
            tab_data.append(x)
        pl.table(cellText=np.array(tab_data), loc=(0,0), cellLoc='center')

        ax = pl.gca()
        ax.set_xticks([])
        ax.set_yticks([])
        pl.title(f"Statistics for {d} dimentions")
        plt.show()

def GraphData(folds_results):
    """
    Graphs all the metrics
    """
    GraphSingleData("accuracy",folds_results)
    GraphSingleData("precision",folds_results)
    GraphSingleData("recall",folds_results)
    GraphSingleData("f1",folds_results)

if __name__ == "__main__":
    #Gets data
    folds_results = []
    for fold in range(5):
        t1 = time.time()
        X_training, y_training, X_test, y_test = GetDataset(split_train_test=True, seed= fold)
        #Scales the data
        scaler = StandardScaler()   
        X_training = scaler.fit_transform(X_training)
        X_test = scaler.transform(X_test)
        #Defines dimentions and number of neighbours
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
        dim.append(30)
        folds_results.append(results)
    #Prints total time needed for reduction and KNN 
    print(f"Total time: {time.time()-t1}")

    #Creates all the graphs for the report
    GraphData(folds_results)
    #Creates all the tables for the report
    TabData(folds_results)
