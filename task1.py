import csv
import random
from sklearn.decomposition import PCA
def GetDataset(split_train_test = False, seed=314):
    """Returns dataset obtained from the wdbc.data file"""
    random.seed(seed)
    with open("wdbc.data") as f:
        data = [row for row in csv.reader(f)]
    len_data = len(data)
    #Binary label
    classify = {"B":0, "M":1}
    #If a training test split is needed
    if split_train_test:
        #Used to randomly select training/test data
        random_placement = random.sample(range(len_data), 169)
        X_training, X_test = [],[] 
        y_training, y_test = [],[] 
        for i in range(len_data):
            to_float = [float(x) for x in data[i][2:]]
            #Pseudo-random puti n test
            if i in random_placement:
                X_test.append(to_float)
                y_test.append(classify[data[i][1]])
            #Put in training
            else:
                X_training.append(to_float)
                y_training.append(classify[data[i][1]])
        return X_training,y_training,X_test,y_test
    else:
        #Returns the entire dataset
        X,y = [],[]
        for i in range(len_data):
            to_float = [float(x) for x in data[i][2:]]
            X.append(to_float)
            y.append(classify[data[i][1]])
        return X,y
    
def MultiplePCA(X, dimentions, X_test = None):
    """Returns a dict nr_of_dim:reduced_dataset"""
    d = {}
    #Does reduction for each size in dimentions
    for n_of_elements in dimentions:
        pca = PCA(n_components=n_of_elements)
        X_red = pca.fit_transform(X)
        #If reduced X_test is needed
        if X_test.all() != None:
            X_test_red = pca.fit_transform(X_test)
            d[n_of_elements] = X_red, X_test_red
        else:
            d[n_of_elements] = X_red
    return d