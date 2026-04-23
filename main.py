from neural_network import genetic_algorithm, neural_network
from neural_network.nn_layer import softmax, leaky_relu, relu, linear
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from sklearn.datasets import load_iris, load_digits, load_wine, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder



import time
import logging

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def one_hot(y, n_classes=3):
    """
    @brief Converte un vettore di etichette intere in matrice one-hot.

    @param y (array-like) Vettore di etichette intere in [0, n_classes).
    @param n_classes (int) Numero di classi. Default: 3.
    @return (numpy.ndarray) Matrice di forma (len(y), n_classes) con un solo 1 per riga.
    """
    Y = np.zeros((len(y), n_classes))
    for i, label in enumerate(y):
        Y[i][label] = 1
    return Y

def number_params(input_size, individuo, output_size):
    """
    @brief Calcola il numero totale di parametri (pesi + bias) di un'architettura.

    @param input_size (int) Numero di feature in input.
    @param individuo (list[tuple[int, any]]) Cromosoma: lista di (n_neuroni, funzione) per ogni hidden layer.
    @param output_size (int) Numero di neuroni nel layer di output.
    @return (int) Totale pesi + bias di tutta la rete.
    """
    params = 0
    prev = input_size
    for neuroni, _ in individuo:
        params += neuroni * prev + neuroni
        prev = neuroni
    params += output_size * prev + output_size
    return params

def mean_accuracy_on_K_runs(individuo, n_feature, n_output, learning_rate, epochs, Y_train, X_train, Y_val, X_val, K, seed=None):
    """
    @brief Valuta l'accuracy media di un'architettura su K addestramenti indipendenti.

    Per ogni run costruisce una rete con pesi casuali, la addestra su X_train per
    il numero di epoche indicato e misura l'accuracy su X_val.

    @param individuo (list[tuple[int, callable]]) Cromosoma dell'architettura da valutare.
    @param n_feature (int) Numero di feature in input.
    @param n_output (int) Numero di classi in output.
    @param learning_rate (float) Learning rate della backpropagation.
    @param epochs (int) Numero di epoche di addestramento per ogni run.
    @param Y_train (numpy.ndarray) Label one-hot del training set.
    @param X_train (numpy.ndarray) Feature del training set.
    @param Y_val (numpy.ndarray) Label one-hot del validation set.
    @param X_val (numpy.ndarray) Feature del validation set.
    @param K (int) Numero di run indipendenti; la media riduce la varianza da inizializzazione casuale.
    @param seed (int | None) Seed per numpy e random, garantisce riproducibilità. Default: None.
    @return (float) Accuracy media su K run, in [0, 1].
    """

    rng_np = np.random.default_rng(seed)

    rng = random.Random(seed)

    sum_accuracy = 0
    for _ in range(0, K):
        n = neural_network(individuo, n_feature, n_output, learning_rate, softmax, rng=rng_np)
        # ADDESTRAMENTO → training set
        for _ in range(0, epochs):
            x = rng.randint(0, len(Y_train) - 1)
            n.feedback(X_train[x], Y_train[x])
        # VALUTAZIONE → testing set
        correct = 0
        for i in range(0, len(Y_val)):
            guess = n.feedforward(X_val[i])['guess']
            if np.argmax(guess) == np.argmax(Y_val[i]):
                correct += 1
        sum_accuracy += correct / len(X_val)
    accuracy = sum_accuracy / K

    return accuracy

def accuracy_on_test_set(rete_best: neural_network, epochs, x_train, y_train, x_test, y_test, seed=42):
    rng = random.Random(seed)
    for _ in range(epochs):
        idx = rng.randint(0, len(x_train) - 1)
        rete_best.feedback(x_train[idx], y_train[idx])

    # valutiamo sul test set
    correct_test = sum(1 for i in range(len(x_test)) if np.argmax(rete_best.feedforward(x_test[i])['guess']) == np.argmax(y_test[i]))

    accuracy_test = correct_test / len(x_test)

    return accuracy_test
    



SEED = 0

K = 10 # numero di addestramenti per individuo

POPULATION_SIZE = 80
GENERATIONS = 30
MUTATION_RATE = 0.2
TOURNAMENT_SIZE = 10
EPOCHS =  250 #150
LEARNING_RATE = 0.01
LAMBDA_ = 0.1

EPOCHS_BASELINE = EPOCHS


if __name__ == "__main__":

    # load del dataset
    dataset = load_iris()
    
    x = dataset.data
    y = dataset.target

    # numero di feature
    input_size = x.shape[1]

    # numero di classi
    n_classi = len(np.unique(y))


    output_function = softmax

    
    nome_riga = [r for r in dataset.DESCR.split('\n') if r.strip() and not r.startswith('..')]
    DATASET_NAME = nome_riga[0].strip()   # → "Iris plants dataset"
    


    
    
    

    # split train/validation/test
    # abbiamo aggiunto test perché in questo caso abbiamo due algoritmi di ottimizzazione
    # backpropagation che ottimizza una rete sul training set
    # genetic_algorithm che ottimizza una popolazione di reti sul validation set
    # quindi i risultati che otterremmo sono OTTIMISTICI
    # Quindi aggiungiamo l'ultimo set, il test set, che utilizzeremo per calcolare la accuracy 
    # della rete migliore trovata

    # x_train, x_val, y_train, y_val = train_test_split(x,y,test_size=0.2, random_state=SEED)

    
    # primo split: separa il test set (20% del totale) → non sarà mai usato nel GA
    x_trainval, x_test, y_trainval, y_test = train_test_split(x,y,test_size=0.2, random_state=SEED)
    # secondo split: separa val dal rimanente 80% → usato dal GA per valutare la fitness
    x_train, x_val, y_train, y_val = train_test_split(x_trainval, y_trainval, test_size=0.25, random_state=SEED)
    # 0.25 x 0.80 = 0.2 del totale → split finale 60/20/20


    # usiamo random_state=SEED per fissare il seed in modo che il risultato sia riproducibile
    # ogni volta che eseguo il programma ottengo lo stesso split, mi permette di confrontare i risultati tra run diverse

    # Normalizzazione
    # X ← (X - media(X)) / std(X)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train) # fit + trasform
    x_val = scaler.transform(x_val) #. solo trasform
    x_test = scaler.transform(x_test) #. solo trasform  

    
    y_train = one_hot(y_train, n_classes=n_classi)
    y_val = one_hot(y_val, n_classes=n_classi)
    y_test = one_hot(y_test, n_classes=n_classi)



    # BASELINE con backpropagation classica
    # rete neurale addestrata per lo stesso quantitativo di epoche 
    # mi permette di confrontare il risultato del GA
    cromosoma_baseline = [(8,leaky_relu),(8,leaky_relu)]
    # rete_baseline = neural_network(hidden_config=cromosoma_baseline,size_input=input_size , size_output=n_classi, learning_rate=0.01, output_function=output_function)

    # for i in range(0, EPOCHS_BASELINE):
    #     idx = random.randint(0, len(x_train) - 1)
    #     rete_baseline.feedback(x_train[idx], y_train[idx])

    # # valutiamo la baseline sul validation set
    # correct = 0
    # for i in range(0, len(x_val)):
    #     guess = rete_baseline.feedforward(x_val[i])['guess']
    #     if np.argmax(guess) == np.argmax(y_val[i]):
    #         correct += 1
    # accuracy_baseline = correct/len(x_val)

    accuracy_baseline = mean_accuracy_on_K_runs(cromosoma_baseline,
                  input_size,
                  n_classi,
                  LEARNING_RATE,
                  EPOCHS_BASELINE,
                  y_train,
                  x_train,
                  y_test,
                  x_test,
                  K,
                  seed=SEED)
    logger.info(f"accuracy della rete baseline: {round((accuracy_baseline) * 100, 2)}%\n#params: {number_params(input_size, cromosoma_baseline, n_classi)}")

    # ESECUZIONE DEL Genetic Algorithm

    ga = genetic_algorithm.GeneticAlgorithm(population_size=POPULATION_SIZE,
                          generations=GENERATIONS,
                          mutation_rate=MUTATION_RATE,
                          tournament_size=TOURNAMENT_SIZE,
                          epochs=EPOCHS,
                          learning_rate=LEARNING_RATE,
                          lambda_=LAMBDA_,
                          n_feature=input_size,
                          n_output= n_classi,
                          K=K,
                          X_Train=x_train, Y_Train=y_train,
                          X_val=x_val, Y_val=y_val,
                          seed=SEED)
    
    start_time = time.perf_counter()

    (best_individuo, best_fitness, best_accuracy, storia_best_fitness, storia_best_accuracy, storia_mean_accuracy) = ga.run()
    
    stop_time = time.perf_counter()
    logger.info(f"tempo di esecuzione: {stop_time - start_time}")


    # -------------
    # Calcolo accuracy della rete migliore trovata
    # rng_test = np.random.default_rng(SEED)
    
    # best_net = neural_network(best_individuo, input_size, n_classi, LEARNING_RATE, softmax, rng_test)
    
    # test_accuracy = accuracy_on_test_set(best_net, EPOCHS, x_train, y_train, x_test, y_test, SEED)
    test_accuracy = mean_accuracy_on_K_runs(best_individuo, input_size, n_classi, LEARNING_RATE, EPOCHS, y_train, x_train, y_test, x_test, K=K, seed=SEED)

    logger.info(f"Accuracy sul validation set (ottimizzazione GA): {round(best_accuracy*100, 2)}%")
    logger.info(f"Accuracy sul test set (stima reale): {round(test_accuracy*100, 2)}%")

    # Se accuracy_test < accuracy_val di qualche punto percentuale, conferma che il leakage era reale e che ora è eliminato.

    ########################################GRAFICA########################################



    
    
    best_individuo = [(neuroni, funzione.__name__) for neuroni, funzione in best_individuo]
    best_fitness = round(best_fitness * 100,2)
    best_accuracy = round(best_accuracy * 100,2)

    logger.info(f"Miglior architettura trovata:\n{best_individuo}")
    logger.info(f"Fitness:{best_fitness}")
    logger.info(f"Accuracy:{best_accuracy}%")
    logger.info(f"Numero di layer={len(best_individuo)}")
    logger.info(f"#parametri={number_params(input_size,best_individuo, n_classi)}")

    # trovo nella storia in che generazione si posizione il best_individuo
    idx_best = np.argmax(storia_best_fitness)
    
    

    
    # formatting
    best_fitness_pct  = [v * 100 for v in storia_best_fitness]
    best_accuracy_pct = [v * 100 for v in storia_best_accuracy]
    mean_accuracy_pct = [v * 100 for v in storia_mean_accuracy]
    baseline_pct      = accuracy_baseline * 100
    test_accuracy_pct = round(test_accuracy * 100, 2)

    linespace_gen = list(range(1, GENERATIONS + 1))

    max_hidden = max(len(best_individuo), len(cromosoma_baseline))
    font_info  = max(8, 12 - max(0, max_hidden+1))
    
    fig_height = max(8, 6.5 + max_hidden * 0.5)

    fig, axes = plt.subplot_mosaic([
         ['accuracy', 'accuracy', 'info'],
         ['accuracy', 'accuracy', 'info'],
         ['fitness', 'fitness', 'info'],
         ['fitness', 'fitness', 'info']],
         figsize=(10, fig_height), sharex=True)

    ax_fit  = axes['fitness']
    ax_acc  = axes['accuracy']
    ax_info = axes['info']

    # fig, (ax_acc, ax_fit) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    fig.suptitle(
        "Algoritmo Genetico — ricerca dell'architettura ottimale\n"
        f"({DATASET_NAME}, classificazione {n_classi} classi)",
        fontsize=13, fontweight='bold'
    )

    # --- Grafico superiore: Accuracy ---
    ax_acc.plot(linespace_gen, best_accuracy_pct,
                color="#344966",
                linewidth=2,
                label="Miglior accuracy della generazione")
    ax_acc.plot(linespace_gen, mean_accuracy_pct,
                color="#D57D00",
                linewidth=1.5,
                linestyle='--',
                label="Accuracy media della popolazione")
    ax_acc.axhline(test_accuracy_pct,
                   color="#8B008B",
                   linewidth=2.0,
                   label=f"Accuracy test - stima reale ({test_accuracy_pct:.1f})")


    ax_acc.axhline(baseline_pct, color="#C30000B2", linestyle='dashed', linewidth=1.8,
                   label=f"Baseline backprop - test set ({baseline_pct:.1f}%)")

    # annotazione valore finale best accuracy
    # val (ottimistica - GA ha selezionato su questo)
    ax_acc.annotate(
        f"val: {best_accuracy:.1f}%",
        xy=(idx_best + 1, best_accuracy),
        xytext=(-60, -20), textcoords='offset points',
        fontsize=10, color='#228B22', fontweight='extra bold',
        arrowprops=dict(arrowstyle='->', color='#228B22', lw=1.5),
        zorder=5
    )
    ax_acc.scatter(idx_best + 1, best_accuracy, c='#228B22', marker='o', zorder=5)

    ax_acc.annotate(
        f"test: {test_accuracy_pct:.1f}",
        xy=(GENERATIONS, test_accuracy_pct),
        xytext=(-60,-20),
        textcoords='offset points',
        fontsize=10,
        color='#8B008B',
        fontweight='extra bold',
        arrowprops=dict(arrowstyle='->', color='#8B008B', lw=1.5),
        zorder=5
    )



    all_values = np.concatenate([ best_fitness_pct, best_accuracy_pct, mean_accuracy_pct, np.array([baseline_pct]), np.array([test_accuracy_pct])])
    
    min_ax_value = round(np.min(all_values), 2) - 5
    max_ax_value = round(np.max(all_values), 2) + 5


    # data leakage: rappresentazione del costo del leakage, ovvero quanto il GA ha "comprato" dall'ottimizzazione su val.
    ax_acc.annotate('', 
                    xy=(idx_best + 1, min(best_accuracy, test_accuracy_pct)),
                    xytext=(idx_best + 1, max(test_accuracy_pct, best_accuracy)),
                    arrowprops=dict(arrowstyle="<-", color="#1D1C1D", lw=1.8),
                    )
    ax_acc.fill_between(
        linespace_gen,
        test_accuracy_pct,
        best_accuracy_pct,
        where=[b > test_accuracy_pct for b in best_accuracy_pct],
        alpha=0.12, color='#8B008B',
        label="Regione bias ottimistico (val > test)"
    )





    gap = round(best_accuracy - test_accuracy_pct, 1)
    gap_handle = mlines.Line2D([], [], color='#1D1C1D', linewidth=1.8,
                               label=f"bias ottimistico ~{gap:+.1f}%")

    ax_acc.set_ylabel("Accuracy (%)", fontsize=11)
    ax_acc.set_ylim(min_ax_value, max_ax_value)
    handles, labels = ax_acc.get_legend_handles_labels()
    ax_acc.legend(handles=handles + [gap_handle], loc='lower right', fontsize=9, framealpha=0.9)
    ax_acc.grid(True, alpha=0.3)
    ax_acc.set_title(
        "Accuratezza — val (usato dal GA) vs test (stima reale non viziata)",
        fontsize=10
    )

    # --- Grafico inferiore: Fitness (accuracy penalizzata dalla complessità) ---
    ax_fit.plot(linespace_gen, best_fitness_pct,
                color="#344966", linewidth=2,
                label="Miglior fitness della generazione")
    
    # ax_fit.axhline([baseline_pct], color="crimson", linestyle='dashed', linewidth=1.8,
                #    label=f"Baseline backprop ({baseline_pct:.1f}%)")
    # non dovrebbe avere senso comparare la fitness della popolazione con l'accuracy della baseline net

    ax_fit.annotate(
        f"{best_fitness:.1f}",
        xy=(idx_best + 1, best_fitness_pct[idx_best]),
        xytext=(0, 15), textcoords='offset points',
        fontsize=10, color='#228B22',fontweight='extra bold',
        arrowprops=dict(arrowstyle='->', color='#228B22', lw=1.5)
    )

    ax_fit.scatter(idx_best + 1, best_fitness, c='#228B22', marker='o', zorder=5)

    ax_fit.set_xlabel("Generazione", fontsize=11)
    ax_fit.set_ylabel("Fitness", fontsize=11)
    ax_fit.set_xlim(1, GENERATIONS)
    ax_fit.set_ylim(min_ax_value, max_ax_value)
    ax_fit.legend(loc='lower right', fontsize=9, framealpha=0.9)
    ax_fit.grid(True, alpha=0.3)
    ax_fit.set_title(
        f"Fitness = accuracy − λ·complessità  (λ={LAMBDA_}) — premia reti accurate e semplici",
        fontsize=10
    )

    # pannello destro: migliore architettura + baseline
    ax_info.axis('off')

    testo_best = f"Migliore architettura trovata:\n\nInput: {input_size} neuroni\n\n"
    for i, (neuroni, funzione) in enumerate(best_individuo):
        testo_best += f"Hidden layer {i + 1}: {neuroni:<3} neuroni → {funzione}\n"
    testo_best += f"\nOutput: {n_classi} neuroni → {output_function.__name__}"
    testo_best += f"\n\nAccuracy val (GA): {best_accuracy}%"
    testo_best += f"\n\nAccuracy test (stima reale): {test_accuracy_pct}%"
    testo_best += f"\nFitness: {best_fitness}"
    testo_best += f"\n#parametri: {number_params(input_size, best_individuo, n_classi)}"

    testo_base = f"Architettura Baseline:\n\nInput: {input_size} neuroni\n\n"
    for i, (neuroni, funzione) in enumerate(cromosoma_baseline):
        testo_base += f"Hidden layer {i + 1}: {neuroni:<3} neuroni → {funzione.__name__}\n"
    testo_base += f"\nOutput: {n_classi} neuroni → {output_function.__name__}"
    testo_base += f"\n\nAccuracy test: {round(accuracy_baseline * 100, 2)}%"
    testo_base += f"\n#parametri: {number_params(input_size, cromosoma_baseline, n_classi)}"

    # posizioni y adattive: più layer → testo più alto → spazio più equo
    lines_best = testo_best.count('\n') + 1
    lines_base = testo_base.count('\n') + 1
    total      = lines_best + lines_base
    y_best = 1.0 - lines_best / (2 * total) - 0.02
    y_base =       lines_base / (2 * total) + 0.02

    ax_info.text(0.5, y_best, testo_best, transform=ax_info.transAxes,
                 fontsize=font_info, verticalalignment='center', horizontalalignment='center',
                 bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    ax_info.text(0.5, y_base, testo_base, transform=ax_info.transAxes,
                 fontsize=font_info, verticalalignment='center', horizontalalignment='center',
                 bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))




    plt.tight_layout()
    plt.savefig('img/risultati.png', dpi=150)
    plt.show()
    


