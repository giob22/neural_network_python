import random
from .nn_layer import *
from .nn_engine import neural_network

class GeneticAlgorithm:

    def __init__(self, population_size, generations, mutation_rate, tournament_size, epochs, learning_rate, X_Train, Y_Train, X_val, Y_val):
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


        self.X_train = X_Train
        self.Y_train = Y_Train
        self.X_val = X_val
        self.Y_val = Y_val
        # Dati già slittati e normalizzati

        self.hidden_functions = [relu, leaky_relu, sigmoid]
        self.output_functions = [softmax, linear, sigmoid]

        # Serve per evitare che la fitness sia falsata di troppo dalla scelta casuale dei pesi
        self.K = 3

    def _genera_individuo(self): #genera un cromosoma casuale
        cromosoma = []
        n_hidden_layer = random.randint(1,4)
        for _ in range(0, n_hidden_layer):
            cromosoma.append((random.randint(4,32),random.choice(self.hidden_functions)))

        return cromosoma


    def _fitness(self, individuo):
        sum_accuracy = 0
        for _ in range(0,self.K):
            n = neural_network(individuo, 4,3, self.learning_rate,softmax)
            # ADDESTRAMENTO → training set
            for _ in range(0, self.epochs):
                x = random.randint(0,len(self.Y_train) - 1)
                n.feedback(self.X_train[x], self.Y_train[x])
            # VALUTAZIONE → testing set
            correct = 0
            for i in range(0,len(self.Y_val)):
                guess = n.feedforward(self.X_val[i])
                if np.argmax(guess) == np.argmax(self.Y_val[i]):
                    correct += 1
            sum_accuracy += correct/len(self.X_val)
        
        return sum_accuracy/self.K
        
    def _selezione(self, popolazione, fitness_scores):
        idx_t = random.sample(range(0, len(popolazione)), k=self.tournament_size)
        idx_sorted = sorted(idx_t, key= lambda x: fitness_scores[x], reverse=True)
        return popolazione[idx_sorted[0]]
        
        
        


    def _crossover(self, genitore1, genitore2):
        t1 = random.randint(0,len(genitore1)) # potrebbe essere pari a len(genitore)
        t2 = random.randint(0,len(genitore2)) # significa che prende tutti i geni da tale genitore + gli eventuali dell'altro genitore 
        figlio = genitore1[:t1] + genitore2[t2:]
        if len(figlio) == 0:
            pool = genitore1 + genitore2
            figlio.append(random.choice(pool))
        return figlio
    
    def _mutazione(self, individuo):
        individuo = [layer_ for layer_ in individuo]
        for i in range(0,len(individuo)):
            if random.random() < self.mutation_rate:
                new_layer = ()
                mut = random.choices([0,1,2],[14,14,2])[0]
                if mut == 0: # mutano il numero di neuroni
                    new_layer = (random.randint(4,32), individuo[i][1])
                    individuo[i] = new_layer
                elif mut == 1: # muta la funzione di attivazione
                    new_layer = (individuo[i][0], random.choice(self.hidden_functions))
                    individuo[i] = new_layer
                else:
                    if random.randint(0,1) == 0 and len(individuo) > 1:
                        pos = random.randint(0,len(individuo) - 1)
                        del individuo[pos]
                        break # necessario perché dopo l'elimiazione gli indici non sono più validi
                    else:
                        pos = random.randint(0,len(individuo))
                        individuo.insert(pos, (random.randint(4,32),random.choice(self.hidden_functions)))
        return individuo
        

    def run(self): # ciclo principale
        # Cosa deve restituire:
        # - miglior architettura trovata
        # - storia della fitness migliore generazione
        # - storia della fitness media per generazione
        popolazione = [self._genera_individuo() for _ in range(0,self.population_size)]

        storia_best_fitness = [] # migliore per generazione
        storia_mean_fitness = [] # media per generazione

        best_individuo = None
        best_fitness = 0

        for i in range(0, self.generations):

            # valutiamo la fitness di ogni individuo della popolazione
            fitness_scores = [self._fitness(individuo) for individuo in popolazione]

            # Aggiornamento del migliore in assoluto
            idx_best = np.argmax(fitness_scores)
            if fitness_scores[idx_best] > best_fitness:
                best_fitness = fitness_scores[idx_best]
                best_individuo = popolazione[idx_best]
                
            
            # salviamo le statistiche
            storia_best_fitness.append(max(fitness_scores))
            storia_mean_fitness.append(np.mean(fitness_scores))

            # stampa dell'avanzamento
            print(f"[gen: {i:<10}] best fitness: {storia_best_fitness[-1]:<10} | mean fitness: {storia_mean_fitness[-1]}")

            # Nuova popolazione
            new_population = []
            
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

        return (best_individuo, best_fitness, storia_best_fitness, storia_mean_fitness)

