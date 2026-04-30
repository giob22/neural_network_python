import random
import logging
from .nn_layer import *
from .nn_engine import neural_network

from concurrent.futures import ProcessPoolExecutor

logger = logging.getLogger(__name__) 

MIN_NEURONI = 4
MAX_NEURONI = 64
MIN_LAYER = 1
MAX_LAYER = 10

class GeneticAlgorithm:

    def __init__(self, population_size, generations, mutation_rate, tournament_size, epochs, learning_rate, n_feature, n_output, K, lambda_, X_Train, Y_Train, X_val, Y_val, min_neuroni=MIN_NEURONI, max_neuroni=MAX_NEURONI, min_layer=MIN_LAYER, max_layer=MAX_LAYER, seed=None, max_workers=None):
        """
        @brief Inizializza l'algoritmo genetico per la ricerca automatica dell'architettura.

        @param population_size (int) Numero di individui per generazione.
        @param generations (int) Numero di generazioni evolutive da eseguire.
        @param mutation_rate (float) Probabilità [0, 1] che ogni gene venga mutato.
        @param tournament_size (int) Numero di individui che partecipano a ogni torneo di selezione.
        @param epochs (int) Epoche di backpropagation per addestrare ogni rete durante la fitness.
        @param learning_rate (float) Learning rate della backpropagation interna.
        @param n_feature (int) Numero di feature in input (corrisponde a size_input della rete).
        @param n_output (int) Numero di classi in output (corrisponde a size_output della rete).
        @param K (int) Numero di valutazioni indipendenti per individuo; la fitness finale è la media.
        @param lambda_ (float) Coefficiente di penalità sulla complessità nella funzione di fitness.
        @param X_Train (numpy.ndarray) Feature del training set, forma (n_campioni, n_feature).
        @param Y_Train (numpy.ndarray) Label one-hot del training set, forma (n_campioni, n_output).
        @param X_val (numpy.ndarray) Feature del validation set.
        @param Y_val (numpy.ndarray) Label one-hot del validation set.
        @param min_neuroni (int) Numero minimo di neuroni per hidden layer generabile. Default: 4.
        @param max_neuroni (int) Numero massimo di neuroni per hidden layer generabile. Default: 64.
        @param min_layer (int) Numero minimo di hidden layer per cromosoma generabile. Default: 1.
        @param max_layer (int) Numero massimo di hidden layer per cromosoma generabile. Default: 10.
        @param seed (int | None) Seed per la riproducibilità. Controlla tutta la casualità strutturale
               del GA (generazione popolazione, selezione, crossover, mutazione) tramite self.rng.
               I worker di _fitness ricevono seed derivati (seed + gen*pop_size + idx).
               Se None, il comportamento è non deterministico. Default: None.
        @note I dati devono essere già normalizzati e splittati prima di essere passati.
        @note I bounds min_neuroni/max_neuroni/min_layer/max_layer sovrascrivono le costanti di modulo
              MIN_NEURONI, MAX_NEURONI, MIN_LAYER, MAX_LAYER tramite global.
        """
        
        self.min_layer = min_layer 
        self.min_neuroni = min_neuroni 
        self.max_layer = max_layer
        self.max_neuroni = max_neuroni 


        self.max_workers = max_workers
        self.seed = seed
        self.rng = random.Random(seed)  # RNG isolato: non interferisce con il random globale

        self.n_feature = n_feature
        self.n_output = n_output
        self.population_size = population_size
        # numero di individui
        self.generations = generations
        # quante generazioni far evolvere
        self.mutation_rate = mutation_rate
        # mutation rate
        self.tournament_size = tournament_size
        # quanti individui partecipano ad ogni torneo di selezione
        self.epochs = epochs
        self.learning_rate = learning_rate

        self.lambda_ = lambda_


        self.X_train = X_Train
        self.Y_train = Y_Train
        self.X_val = X_val
        self.Y_val = Y_val
        # Dati già slittati e normalizzati

        self.hidden_functions = [relu, leaky_relu, sigmoid]
        self.output_functions = [softmax, linear, sigmoid]

        # Serve per evitare che la fitness sia falsata di troppo dalla scelta casuale dei pesi
        self.K = K


    def _genera_individuo(self):
        """
        @brief Genera un cromosoma casuale che rappresenta un'architettura di rete neurale.

        Sceglie casualmente un numero di hidden layer tra MIN_LAYER e MAX_LAYER; per ognuno
        sorteggia neuroni (MIN_NEURONI–MAX_NEURONI) e funzione di attivazione da hidden_functions.

        @return (list[tuple[int, callable]]) Lista di tuple (n_neuroni, funzione),
                una per ogni hidden layer.
        """
        cromosoma = []
        n_hidden_layer = self.rng.randint(self.min_layer, self.max_layer)
        for _ in range(0, n_hidden_layer):
            cromosoma.append((self.rng.randint(self.min_neuroni, self.max_neuroni), self.rng.choice(self.hidden_functions)))

        return cromosoma


    def _complessita(self, individuo):
        """
        @brief Calcola il numero totale di parametri (pesi + bias) di un'architettura.

        Conta tutti i parametri addestrabili della rete corrispondente al cromosoma:
        per ogni hidden layer (n_neuroni × n_prev + n_neuroni), più il layer di output.
        Il valore restituito è il conteggio grezzo; la trasformazione logaritmica
        viene applicata in _fitness per stabilizzare la scala della penalità.

        @param individuo (list[tuple[int, callable]]) Cromosoma che descrive l'architettura.
        @return (int) Numero totale di parametri della rete (pesi + bias di tutti i layer).
        @note Il valore viene passato a np.log10() in _fitness; la funzione restituisce
              sempre un intero >= 1 (almeno il layer di output esiste).
        """
        params = 0
        prev = self.n_feature # n of inputs
        for n_neuroni, _ in individuo:
            params += n_neuroni * prev + n_neuroni # pesi + bias
            prev = n_neuroni
        params += self.n_output * prev + self.n_output # output layer
        return params

    def _fitness(self, individuo, seed=None):
        """
        @brief Valuta la qualità di un individuo tramite addestramento e misura dell'accuracy.

        Ripete K volte: costruisce la rete con quell'architettura (pesi casuali),
        la addestra sul training set per self.epochs epoche, valuta l'accuracy sul
        validation set. La fitness finale penalizza la complessità con una penalità
        logaritmica per rendere lambda_ stabile indipendentemente dalla scala dei parametri.

        Formula della fitness:
        @code
          fitness = accuracy - lambda_ * log10(n_params)
        @endcode

        La scelta di log10 invece di una penalità lineare (accuracy - lambda_ * n_params)
        è motivata dalla stabilità della scala: n_params varia tipicamente di 2-3 ordini
        di grandezza nello spazio di ricerca (es. 50–50000), mentre log10(n_params) varia
        solo di ~1.7–4.7. Questo rende lambda_ calibrabile con valori nell'ordine di
        [0.05, 0.30] anziché dover usare valori dell'ordine di 1e-5 per la versione lineare.

        @param individuo (list[tuple[int, callable]]) Cromosoma da valutare.
        @param seed (int | None) Seed per i due RNG locali del worker:
                   - random.Random(seed): scelta dei campioni di training per SGD.
                   - numpy.random.default_rng(seed): inizializzazione pesi/bias di ogni rete.
                   In run() viene passato un seed derivato da self.seed per garantire
                   riproducibilità completa anche con ProcessPoolExecutor. Se None, non deterministico.
        @return (tuple[float, float]) Coppia (fitness, accuracy) dove:
                - fitness = accuracy - lambda_ * log10(n_params)
                - accuracy = media delle K valutazioni sul validation set, in [0, 1].
        @note K valutazioni indipendenti riducono la varianza dovuta all'inizializzazione casuale.
        @note La fitness può essere negativa se lambda_ è molto alto e la rete è complessa.
        """
        rng = random.Random(seed)          # RNG locale (Python) per la scelta dei campioni
        np_rng = np.random.default_rng(seed)  # RNG locale (NumPy) per l'inizializzazione dei pesi
        sum_accuracy = 0
        for _ in range(0, self.K):
            n = neural_network(individuo, self.n_feature, self.n_output, self.learning_rate, softmax, rng=np_rng)
            # ADDESTRAMENTO → training set
            for _ in range(0, self.epochs):
                x = rng.randint(0, len(self.Y_train) - 1)
                n.feedback(self.X_train[x], self.Y_train[x])
            # VALUTAZIONE → testing set
            correct = 0
            for i in range(0, len(self.Y_val)):
                guess = n.feedforward(self.X_val[i])['guess']
                if np.argmax(guess) == np.argmax(self.Y_val[i]):
                    correct += 1
            sum_accuracy += correct / len(self.X_val)

        accuracy = sum_accuracy / self.K

        penalita = self._complessita(individuo)
        fitness = accuracy - self.lambda_ * (np.log10(penalita))

        return (fitness, accuracy)

    def _selezione(self, popolazione, fitness_scores):
        """
        @brief Seleziona un individuo tramite torneo.

        Estrae casualmente tournament_size individui dalla popolazione e
        restituisce quello con la fitness più alta.

        @param popolazione (list[list[tuple[int, callable]]]) Lista dei cromosomi della generazione corrente.
        @param fitness_scores (list[float]) Lista di fitness corrispondente a ogni individuo,
               con lo stesso ordine di popolazione.
        @return (list[tuple[int, callable]]) Il cromosoma vincitore del torneo.
        """
        idx_t = self.rng.sample(range(0, len(popolazione)), k=self.tournament_size)
        idx_sorted = sorted(idx_t, key=lambda x: fitness_scores[x], reverse=True)
        return popolazione[idx_sorted[0]]


    def _crossover(self, genitore1, genitore2):
        """
        @brief Genera un figlio combinando due genitori con punto di taglio variabile.

        Sceglie un punto di taglio indipendente per ciascun genitore (t1 in [0, len(g1)],
        t2 in [0, len(g2)]) e costruisce il figlio con i primi t1 layer di genitore1
        e i layer da t2 in poi di genitore2.

        @param genitore1 (list[tuple[int, callable]]) Primo cromosoma genitore.
        @param genitore2 (list[tuple[int, callable]]) Secondo cromosoma genitore.
        @return (list[tuple[int, callable]]) Cromosoma figlio con lunghezza in [1, max_layer].
        @note Se il crossover produce un figlio vuoto (t1=0 e t2=len(g2)), viene inserito
              un gene casuale scelto dal pool dei layer dei due genitori.
        @note Il figlio viene troncato a self.max_layer se la concatenazione supera il limite:
              questo evita la crescita indefinita della profondità nelle generazioni successive.
        """
        t1 = self.rng.randint(0, len(genitore1))
        t2 = self.rng.randint(0, len(genitore2))

        # caso in cui t1 = 0 e t2 = len(genitore2)
        figlio = genitore1[:t1] + genitore2[t2:]
        if len(figlio) == 0:
            pool = genitore1 + genitore2
            figlio.append(self.rng.choice(pool))
        # il figlio non può crescere indefinitivamente
        if len(figlio) > self.max_layer:
            figlio = figlio[:self.max_layer]
        return figlio

    def _mutazione(self, individuo):
        """
        @brief Applica mutazioni casuali a un cromosoma.

        Per ogni gene, con probabilità mutation_rate, applica una delle tre mutazioni
        con pesi [5, 5, 3] (tipo 0 e 1 più frequenti, tipo 2 meno frequente):
        - tipo 0: cambia il numero di neuroni del layer (min_neuroni–max_neuroni);
        - tipo 1: cambia la funzione di attivazione scelta da hidden_functions;
        - tipo 2: aggiunge un nuovo layer casuale in posizione random, oppure rimuove
                  un layer esistente (50/50), garantendo profondità minima 1.

        @param individuo (list[tuple[int, callable]]) Cromosoma da mutare.
        @return (list[tuple[int, callable]]) Cromosoma mutato (lista nuova; l'originale non viene modificato).
        @note Il loop itera all'indietro (da len-1 a 0): il range è valutato una sola volta
              all'inizio, quindi dopo una rimozione gli indici rimanenti (più bassi) restano validi.
        @note La rimozione di un layer è consentita solo se l'individuo ha più di un layer,
              garantendo che il cromosoma non diventi mai vuoto.
        @note L'inserimento di un layer non è vincolato a max_layer: per una lunghezza
              garantita usare _crossover, che applica il troncamento.
        """
        individuo = [layer_ for layer_ in individuo]
        for i in range(len(individuo) - 1, -1, -1):
            if self.rng.random() < self.mutation_rate:
                new_layer = ()
                mut = self.rng.choices([0, 1, 2], [5,5,3])[0]
                if mut == 0: # mutano il numero di neuroni
                    new_layer = (self.rng.randint(self.min_neuroni, self.max_neuroni), individuo[i][1])
                    individuo[i] = new_layer
                elif mut == 1: # muta la funzione di attivazione
                    new_layer = (individuo[i][0], self.rng.choice(self.hidden_functions))
                    individuo[i] = new_layer
                else:

                    if self.rng.randint(0, 1) == 0 and len(individuo) > 1:
                        pos = self.rng.randint(0, len(individuo) - 1)
                        del individuo[pos]
                    elif len(individuo) < self.max_layer:
                        pos = self.rng.randint(0, len(individuo))
                        individuo.insert(pos, (self.rng.randint(self.min_neuroni, self.max_neuroni), self.rng.choice(self.hidden_functions)))
        return individuo


    def run(self):
        """
        @brief Esegue il ciclo evolutivo principale e restituisce i risultati.

        Ad ogni generazione: valuta la fitness di tutti gli individui, aggiorna
        il miglior individuo assoluto, salva le statistiche, genera la nuova
        popolazione tramite elitismo + crossover + mutazione.

        @return (tuple) Tupla con sei elementi:
                - best_individuo (list[tuple[int, callable]]): architettura ottimale trovata (best globale).
                - best_fitness (float): fitness del miglior individuo in assoluto, in [−∞, 1].
                - best_accuracy (float): accuracy sul validation set del miglior individuo, in [0, 1].
                - storia_best_fitness (list[float]): fitness dell'individuo con best fitness per generazione.
                - storia_best_accuracy (list[float]): accuracy dell'individuo con best fitness per generazione.
                  Corrisponde allo stesso soggetto di storia_best_fitness (stesso idx_best), non al massimo
                  dell'accuracy nella popolazione; questo garantisce coerenza con best_individuo nel plot.
                - storia_mean_accuracy (list[float]): accuracy media della popolazione per generazione.
        @note Il miglior individuo assoluto (best globale tra tutte le generazioni) viene
              reinserito nella nuova popolazione ad ogni generazione (elitismo).
        @note Se self.seed è impostato, i worker di ProcessPoolExecutor ricevono seed
              deterministici derivati come `seed + gen * population_size + idx_individuo`,
              garantendo riproducibilità completa anche con valutazione parallela.
        @note idx_best è calcolato con np.nanargmax(fitness_scores): se un worker restituisce
              NaN (es. underflow numerico), la generazione non viene invalidata. La media
              usa np.nanmean per la stessa ragione.
        """
        # Cosa deve restituire:
        # - miglior architettura trovata
        # - storia della fitness migliore generazione
        # - storia della fitness media per generazione
        popolazione = [self._genera_individuo() for _ in range(0, self.population_size)]

        storia_best_fitness = [] # migliore per generazione della fitness
        storia_best_accuracy = [] # migliore per generazione della accuracy
        storia_mean_accuracy = [] # media per generazione dell'accuracy

        best_individuo = None
        best_fitness = -float('inf')
        best_accuracy = 0.0

        for i in range(0, self.generations):

            # valutiamo la fitness di ogni individuo della popolazione
            # risultati = [self._fitness(individuo,self.seed) for individuo in popolazione]
            # I seed per i worker sono derivati dal seed principale + indice generazione + indice individuo,
            # così ogni valutazione è deterministica e distinta anche tra generazioni diverse.
            worker_seeds = [
                self.seed + i * self.population_size + j if self.seed is not None else None
                for j in range(len(popolazione))
            ]
            with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                risultati = list(executor.map(self._fitness, popolazione, worker_seeds))
            fitness_scores  = [r[0] for r in risultati]
            accuracy_scores = [r[1] for r in risultati]


            # Aggiornamento del migliore in assoluto
            idx_best = np.nanargmax(fitness_scores)
            if fitness_scores[idx_best] > best_fitness:
                best_fitness = fitness_scores[idx_best]
                best_individuo = popolazione[idx_best]
                best_accuracy = accuracy_scores[idx_best]


            # salviamo le statistiche
            storia_best_fitness.append(fitness_scores[idx_best])
            storia_best_accuracy.append(accuracy_scores[idx_best])

            storia_mean_accuracy.append(np.nanmean(accuracy_scores))

            # stampa dell'avanzamento
            logger.info(f"[gen: {i:>3}] best fitness: {round(storia_best_fitness[-1]*100, 2):>5} | best accuracy: {round(storia_best_accuracy[-1] * 100, 2):>5}% | mean accuracy: {round(storia_mean_accuracy[-1] * 100, 2):>5}%")



            # Nuova popolazione
            new_population = []

            # elitismo
            if best_individuo is not None:
                new_population.append(best_individuo)

            while len(new_population) < self.population_size:
                # scelta dei genitori
                genitore1 = self._selezione(popolazione, fitness_scores)
                genitore2 = self._selezione(popolazione, fitness_scores)

                # generiamo il figlio
                figlio = self._crossover(genitore1, genitore2)

                # mutazioni
                figlio = self._mutazione(figlio)

                new_population.append(figlio)
            popolazione = new_population

        return (best_individuo, best_fitness, best_accuracy, storia_best_fitness, storia_best_accuracy, storia_mean_accuracy)
