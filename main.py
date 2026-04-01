from neural_network import *
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def one_hot(y, n_classes=3):
    Y = np.zeros((len(y), n_classes))
    for i, label in enumerate(y):
        Y[i][label] = 1
    return Y

if __name__ == "__main__":

    # load del dataset
    dataset = load_iris()
    print(dataset)
    x = dataset.data
    y = dataset.target

    # split train/validation

    x_train, x_val, y_train, y_val = train_test_split(x,y,test_size=0.2, random_state=42)


    # usiamo random_state=42 per fissare il seed in modo che il risultato sia riproducibile
    # ogni volta che eseguo il programma ottengo lo stesso split, mi permette di confrontare i risultati tra run diverse

    # Normalizzazione
    
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_val = scaler.transform(x_val)

    
    y_train = one_hot(y_train)
    y_val = one_hot(y_val)
