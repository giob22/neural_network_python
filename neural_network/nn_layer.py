import numpy as np


def relu(x):
    """
    @brief Funzione di attivazione ReLU (Rectified Linear Unit).

    @param x (numpy.ndarray) Valore o array di valori pre-attivazione.
    @return (numpy.ndarray) Array con valori negativi azzerati: max(0, x).
    """
    return np.maximum(0, x)


def drelu(x):
    """
    @brief Derivata della funzione ReLU.

    @param x (numpy.ndarray) Valore pre-attivazione (prima di applicare ReLU).
    @return (numpy.ndarray) Array di 1.0 dove x > 0, 0.0 altrove.
    """
    return np.where(x > 0, 1.0, 0.0)


def leaky_relu(x):
    """
    @brief Funzione di attivazione Leaky ReLU.

    Variante di ReLU che permette un piccolo gradiente negativo (slope=0.01)
    per valori negativi, evitando il problema dei neuroni "morti".

    @param x (numpy.ndarray) Valore o array di valori pre-attivazione.
    @return (numpy.ndarray) x se x >= 0, 0.01*x altrimenti.
    """
    return np.maximum(0.01 * x, x)


def d_leaky_relu(x):
    """
    @brief Derivata della funzione Leaky ReLU.

    @param x (numpy.ndarray) Valore pre-attivazione.
    @return (numpy.ndarray) Array di 1.0 dove x > 0, 0.01 altrove.
    """
    return np.where(x > 0, 1.0, 0.01)


def linear(x):
    """
    @brief Funzione di attivazione lineare (identità).

    @param x (numpy.ndarray) Valore o array di valori pre-attivazione.
    @return (numpy.ndarray) x invariato.
    """
    return x


def d_linear(x):
    """
    @brief Derivata della funzione lineare.

    @param x (numpy.ndarray) Valore pre-attivazione (non usato).
    @return (numpy.ndarray)  array della stessa dimensione dell'ingresso riempito di 1.
    """
    return np.ones_like(x)


def sigmoid(x):
    """
    @brief Funzione di attivazione Sigmoid.

    Mappa i valori nell'intervallo (0, 1). Il clipping a [-500, 500]
    previene overflow numerici in np.exp.

    @param x (numpy.ndarray) Valore o array di valori pre-attivazione.
    @return (numpy.ndarray) Valori nell'intervallo (0, 1).
    @note L'input viene clippato a [-500, 500] per stabilità numerica.
    """
    x_clipped = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x_clipped))


def d_sigmoid(x):
    """
    @brief Derivata della funzione Sigmoid.

    Calcola s * (1 - s) dove s = sigmoid(x).

    @param x (numpy.ndarray) Valore pre-attivazione (non la sigmoid già applicata).
    @return (numpy.ndarray) Derivata della sigmoid in x.
    """
    s = sigmoid(x)
    return s * (1 - s)


def softmax(x):
    """
    @brief Funzione di attivazione Softmax.

    Converte un vettore di punteggi in una distribuzione di probabilità.
    Sottrae il massimo da x per stabilità numerica (evita overflow in exp).

    @param x (numpy.ndarray) Vettore colonna di punteggi pre-attivazione.
    @return (numpy.ndarray) Vettore di probabilità con somma 1.
    @note Opera sull'intero vettore, non elemento per elemento.
    @note Se tutti i valori di exp(x - max(x)) underflowano a 0.0 (input
          estremamente negativi), la somma sarebbe 0 e la divisione produrrebbe
          NaN. In quel caso la funzione restituisce una distribuzione uniforme
          1/n per ciascuna classe (nessuna classe preferita), evitando la propagazione di NaN nel calcolo
          della fitness e nei grafici matplotlib.
    """
    e_x = np.exp(x - np.max(x))
    total = e_x.sum()
    if total == 0.0:
        return np.full_like(e_x, 1.0 / len(e_x))
    return e_x / total


def d_softmax(x):
    """
    @brief Approssimazione diagonale della derivata della Softmax.

    La derivata esatta della Softmax è un Jacobiano (matrice). Questa
    implementazione usa l'approssimazione diagonale s * (1 - s), identica
    alla derivata della Sigmoid. È sufficiente per la backpropagation su
    problemi semplici come Iris, dove le classi sono ben separabili.

    @param x (numpy.ndarray) Vettore pre-attivazione.
    @return (numpy.ndarray) Approssimazione diagonale del Jacobiano.
    @note Ignora le interazioni tra neuroni di output diversi.
    """
    s = softmax(x)
    return s * (1 - s)


## Dizionario che mappa ogni funzione di attivazione alla sua derivata.
hidden_functions = {relu: drelu, linear: d_linear, sigmoid: d_sigmoid, leaky_relu: d_leaky_relu, softmax: d_softmax}


class layer:
    def __init__(self, row, col, activation_function, rng=None):
        """
        @brief Costruisce un layer della rete neurale con pesi e bias inizializzati casualmente.

        @param row (int) Numero di neuroni del layer (righe della matrice pesi).
        @param col (int) Numero di neuroni del layer precedente (colonne della matrice pesi).
        @param activation_function (callable) Funzione di attivazione del layer;
               deve essere una chiave presente in hidden_functions.
        @param rng (numpy.random.Generator | None) Generatore NumPy isolato per la
               riproducibilità. Se None usa np.random globale (comportamento legacy).
               Passare un Generator creato con numpy.random.default_rng(seed) garantisce
               che l'inizializzazione dei pesi sia deterministica e isolata dagli altri layer.
        @note I pesi sono inizializzati con distribuzione normale: He (std=sqrt(2/col)) per
              relu/leaky_relu, Xavier (std=sqrt(1/col)) per le altre funzioni. Il bias
              è inizializzato con distribuzione uniforme in [-0.5, 0.5].
        """
        _rng = rng if rng is not None else np.random

        # ! spiega perché si sta utilizzando un modo diverso per assegnare i valori ai pesi, ovvero il problema del dying ReLU e la saturazione della sigmoid
        if activation_function in (relu, leaky_relu):
            std = np.sqrt(2/col) #He: compensa ReLU che taglia metà attivazioni
        else:
            std = np.sqrt(1/col) # Xavier: per funzioni simmetriche attorno a 0

        self.weights = _rng.normal(0,std,size=(row,col))

        # self.weights = _rng.uniform(-0.5, 0.5, size=(row, col))
        self.func = activation_function
        self.dfunc = hidden_functions[self.func]
        self.bias = _rng.uniform(-0.5, 0.5, size=(row, 1))

    @property
    def W(self):
        """
        @brief Restituisce la matrice dei pesi del layer.

        @return (numpy.ndarray) Matrice pesi di forma (row, col).
        """
        return self.weights

    @W.setter
    def W(self, x):
        """
        @brief Imposta la matrice dei pesi del layer.

        @param x (numpy.ndarray) Nuova matrice pesi; deve avere la stessa forma di weights.
        """
        self.weights = x

    @property
    def b(self):
        """
        @brief Restituisce il vettore bias del layer.

        @return (numpy.ndarray) Vettore bias di forma (row, 1).
        """
        return self.bias

    @b.setter
    def b(self, x):
        """
        @brief Imposta il vettore bias del layer.

        @param x (numpy.ndarray) Nuovo vettore bias; deve avere forma (row, 1).
        """
        self.bias = x

    @property
    def activation_function(self):
        """
        @brief Restituisce la funzione di attivazione del layer.

        @return (callable) La funzione di attivazione assegnata al costruttore.
        """
        return self.func

    def __str__(self):
        """
        @brief Rappresentazione testuale del layer.

        @return (str) Stringa con la matrice dei pesi formattata da numpy.
        """
        return f"{self.weights}"
