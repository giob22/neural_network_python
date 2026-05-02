import numpy as np
from .nn_layer import layer

class neural_network:
    def __init__(self, hidden_config, size_input, size_output, learning_rate, output_function, rng=None):
        """
        @brief Costruisce una rete neurale multi-layer con architettura personalizzabile.

        @param hidden_config (list[tuple[int, callable]]) Lista di tuple (n_neuroni, funzione_attivazione)
               che descrive ogni hidden layer. Esempio: [(8, relu), (4, sigmoid)].
        @param size_input (int) Numero di feature in input.
        @param size_output (int) Numero di neuroni nel layer di output.
        @param learning_rate (float) Passo di aggiornamento dei pesi nella backpropagation.
        @param output_function (callable) Funzione di attivazione del layer di output.
        @param rng (numpy.random.Generator | None) Generatore NumPy isolato, propagato a ogni
               layer per l'inizializzazione deterministica di pesi e bias.
               Creare con numpy.random.default_rng(seed). Se None usa np.random globale.
        @note Il layer di output viene sempre aggiunto automaticamente; non va incluso in hidden_config.
        """
        self.size = len(hidden_config) + 1 #1 per l'output layer sempre presente
        self.input_size = size_input
        self.output_size = size_output
        self.lr = learning_rate


        self.hiddens_layers = []

        # creo i layer intermedi (hidden) + input layer
        prev = size_input
        for rows, f in hidden_config:
            self.hiddens_layers.append(layer(rows, prev, f, rng=rng))
            prev = rows

        self.output_layer = layer(size_output, prev, output_function, rng=rng)


    def feedforward(self, x):
        """
        @brief Esegue il passo forward della rete neurale.

        Propaga l'input attraverso tutti gli hidden layer e il layer di output,
        salvando i valori pre-attivazione e gli input di ogni layer per la
        backpropagation.

        @param x (array-like) Vettore di input di lunghezza size_input.
        @return (dict) Dizionario con le chiavi:
                - 'guess' (numpy.ndarray): output del layer finale dopo la funzione di attivazione.
                - 'input_vals' (list[numpy.ndarray]): input di ogni layer prima della moltiplicazione.
                - 'intermediate_vals' (list[numpy.ndarray]): valori pre-attivazione di ogni layer.
        @note Lancia ValueError se la dimensione di x non corrisponde a size_input.
        """
        layer_inputs = []
        pre_activations = []

        x = np.array(x).reshape(-1,1)

        if x.shape[0] != self.input_size:
            raise ValueError(f"Input atteso di dimensione {self.input_size}, ricevuto {x.shape[0]}")

        t = x

        for elem in self.hiddens_layers:
            layer_inputs.append(t)
            t = elem.W @ t + elem.b
            pre_activations.append(t)

            t = elem.activation_function(t)

        layer_inputs.append(t)



        t = self.output_layer.W @ t + self.output_layer.b

        pre_activations.append(t)
        res = self.output_layer.func(t)

        return {'guess': res, 'input_vals': layer_inputs, 'intermediate_vals': pre_activations}

    def feedback(self, inputs, target):
        """
        @brief Esegue un passo di backpropagation e aggiorna pesi e bias.

        Chiama internamente feedforward per ottenere guess e valori intermedi,
        calcola i delta con la regola della catena partendo dall'output layer,
        poi aggiorna ogni layer con la discesa del gradiente (SGD).

        I delta vengono accumulati in ordine inverso rispetto ai layer:
        @code
          deltas = [delta_out, delta_hidden_N, ..., delta_hidden_0]
        @endcode
        L'aggiornamento usa indicizzazione negativa diretta su deltas per
        evitare la creazione di una lista rovesciata:
        - hidden layer i  →  deltas[-(i+1)]
        - output layer    →  deltas[0]

        @param inputs (array-like) Vettore di input di lunghezza size_input.
        @param target (array-like) Vettore target atteso di lunghezza size_output.
        @note Modifica direttamente i pesi e i bias di tutti i layer (side effect).
        @note Utilizza cross-entropy come loss, combinata con softmax sull'output layer.
              Il delta di output è calcolato con dCE_softmax (gradiente combinato CE+softmax),
              quindi dfunc del layer di output non viene chiamata.
        """
        target = np.array(target).reshape(-1,1)
        res = self.feedforward(inputs)
        guess = res['guess']
        pre_activations = res['intermediate_vals']
        layer_inputs = res['input_vals']
        deltas = []


        # delta dell'output
        # delta_out = self.dMSE(target, guess) * self.output_layer.dfunc(pre_activations[-1])
        delta_out = self.dCE_softmax(target, guess)
        deltas.append(delta_out)


        # calcolo i delta degli hidden layer, l'ultimo deve utilizzare la matrice dell'output layer

        # I layer precedenti utilizzano i pesi del layer alla loro destra

        current_delta = delta_out
        next_W = self.output_layer.W


        for i in range(len(self.hiddens_layers) - 1, -1, -1):
            current_delta = (next_W.T @ current_delta) * self.hiddens_layers[i].dfunc(pre_activations[i])

            deltas.append(current_delta)

            next_W = self.hiddens_layers[i].W

        # AGGIORNAMENTO DEI PARAMETRI (PESI e BIAS)
        # deltas = [delta_out, delta_hidden_N, ..., delta_hidden_0]
        # Per hidden layer i: deltas[-(i+1)]   (es. i=0 → deltas[-1] = delta_hidden_0)
        # Per output layer:   deltas[0]

        # gradient clipping: previene l'esplosione dei gradienti con reti profonde
        deltas = [np.clip(d, -5.0, 5.0) for d in deltas]

        for i in range(len(self.hiddens_layers)):
            self.hiddens_layers[i].W -= self.lr * (deltas[-(i + 1)] @ layer_inputs[i].T)
            self.hiddens_layers[i].b -= self.lr * deltas[-(i + 1)]

        self.output_layer.W -= self.lr * (deltas[0] @ layer_inputs[-1].T)
        self.output_layer.b -= self.lr * deltas[0]


    # def MSE(self, target, guess):
    #     """
    #     @brief Calcola la Mean Squared Error tra target e predizione.

    #     La formula usata è MSE = sum((target - guess)^2) / 2. Il divisore 2
    #     semplifica la derivata (la costante si cancella).

    #     @param target (numpy.ndarray) Vettore target atteso, forma (size_output, 1).
    #     @param guess (numpy.ndarray) Output predetto dalla rete, forma (size_output, 1).
    #     @return (float) Valore scalare della loss.
    #     """
    #     return np.sum((target - guess)**2) / 2

    # def dMSE(self, target, guess):
    #     """
    #     @brief Calcola la derivata della MSE rispetto all'output della rete.

    #     @param target (numpy.ndarray) Vettore target atteso, forma (size_output, 1).
    #     @param guess (numpy.ndarray) Output predetto dalla rete, forma (size_output, 1).
    #     @return (numpy.ndarray) Gradiente della loss rispetto a guess: (guess - target).
    #     """
    #     return guess - target

    def cross_entropy(self, target, guess):
        """
        @brief Calcola la cross-entropy loss per classificazione multi-classe.

        Formula: CE = -sum(target * log(guess)), dove guess è l'output softmax.
        Il segno negativo è assorbito dal fatto che target è one-hot e si
        considera il log-likelihood della classe corretta.

        @param target (numpy.ndarray) Vettore one-hot del target, forma (size_output, 1).
        @param guess  (numpy.ndarray) Output softmax della rete, forma (size_output, 1).
                      Valori attesi in (0, 1] con somma 1.
        @return (float) Valore scalare della loss (negativo: più vicino a 0 = meglio).
        @note guess viene clippato in [1e-12, 1.0] per evitare log(0) → -inf.
        """
        guess_clipped = np.clip(guess, 1e-12, 1.0)
        return -np.sum(target * np.log(guess_clipped))
    
    def dCE_softmax(self, target, guess):
        """
        @brief Gradiente combinato cross-entropy + softmax rispetto ai logit di input.

        Quando l'ultimo layer usa softmax come attivazione e la loss è cross-entropy,
        il gradiente si semplifica algebricamente a (guess - target), evitando
        il calcolo esplicito della jacobiana della softmax.

        @param target (numpy.ndarray) Vettore one-hot del target, forma (size_output, 1).
        @param guess  (numpy.ndarray) Output softmax della rete, forma (size_output, 1).
        @return (numpy.ndarray) Delta dell'output layer, forma (size_output, 1).
                Il risultato è ∂Loss/∂z, dove z sono i **logit** — i valori pre-attivazione
                dell'output layer (z = W·x + b, prima di softmax). Nel codice corrispondono
                a `pre_activations[-1]` restituito da feedforward.
        @note Valido solo se l'output layer usa softmax. Non chiamare dfunc del
              layer di output dopo questo metodo: il gradiente è già completo.
        """
        return guess - target



    def __str__(self):
        """
        @brief Rappresentazione testuale dell'architettura della rete.

        @return (str) Stringa formattata con numero di layer e neuroni per layer.
        @note Il conteggio neuroni degli hidden layer usa W.size (pesi totali),
              mentre l'output layer usa W.shape[0] (numero neuroni).
        """
        testo = "-"* 35 + "\n" +f"Caratteristiche della rete neurale:\n#layer: {self.size:<10}\n"
        i = 0
        for x in self.hiddens_layers:
            testo += f"{i}. neuroni: {x.W.shape[0]:<10}\n"
            i+=1
        testo += f"output. neuroni: {self.output_layer.W.shape[0]:<10}\n" + "-"*35+ "\n"
        return testo
