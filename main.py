from neural_network import *
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, load_digits, load_wine, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder


import time

def one_hot(y, n_classes=3):
    Y = np.zeros((len(y), n_classes))
    for i, label in enumerate(y):
        Y[i][label] = 1
    return Y

def number_params(input_size, individuo, output_size):
    params = 0
    prev = input_size
    for neuroni, _ in individuo:
        params += neuroni * prev + neuroni
        prev = neuroni
    params += output_size * prev + output_size
    return params

def accuracy_base(individuo, n_feature, n_output, learning_rate, epochs, Y_train, X_train, Y_val, X_val, K):

    sum_accuracy = 0
    for _ in range(0, K):
        n = neural_network(individuo, n_feature, n_output, learning_rate, softmax)
        # ADDESTRAMENTO → training set
        for _ in range(0, epochs):
            x = random.randint(0, len(Y_train) - 1)
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



K = 5 # numero di addestramenti per individuo

POPULATION_SIZE = 50
GENERATIONS = 30
MUTATION_RATE = 0.2
TOURNAMENT_SIZE = 10
EPOCHS =  250 #150
LEARNING_RATE = 0.01
LAMBDA_ = 0.15

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
    


    
    
    

    # split train/validation

    x_train, x_val, y_train, y_val = train_test_split(x,y,test_size=0.2, random_state=42)


    # usiamo random_state=42 per fissare il seed in modo che il risultato sia riproducibile
    # ogni volta che eseguo il programma ottengo lo stesso split, mi permette di confrontare i risultati tra run diverse

    # Normalizzazione
    # X ← (X - media(X)) / std(X)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_val = scaler.transform(x_val)

    
    y_train = one_hot(y_train, n_classes=n_classi)
    y_val = one_hot(y_val, n_classes=n_classi)


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

    accuracy_baseline = accuracy_base(cromosoma_baseline,
                  input_size,
                  n_classi,
                  LEARNING_RATE,
                  EPOCHS_BASELINE,
                  y_train,
                  x_train,
                  y_val,
                  x_val,
                  K)
    print(f"accuracy della rete baseline: {round((accuracy_baseline) * 100, 2)}%\n#params: {number_params(input_size, cromosoma_baseline, n_classi)}")

    # ESECUZIONE DEL Genetic Algorithm

    ga = GeneticAlgorithm(population_size=POPULATION_SIZE,
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
                          seed=42)
    
    start_time = time.perf_counter()

    (best_individuo, best_fitness, best_accuracy, storia_best_fitness, storia_best_accuracy, storia_mean_accuracy) = ga.run()
    
    stop_time = time.perf_counter()
    print(f"tempo di esecuzione: {stop_time - start_time}")
    
    
    best_individuo = [(neuroni, funzione.__name__) for neuroni, funzione in best_individuo]
    best_fitness = round(best_fitness * 100,2)
    best_accuracy = round(best_accuracy * 100,2)

    print(f"Miglior architettura trovata:\n{best_individuo}")
    print(f"Fitness:{best_fitness}")
    print(f"Accuracy:{best_accuracy}%")
    print(f"Numero di layer={len(best_individuo)}")
    print(f"#parametri={number_params(input_size,best_individuo, n_classi)}")

    # trovo nella storia in che generazione si posizione il best_individuo
    idx_best = np.argmax(storia_best_fitness)
    
    

    
    linespace_gen = list(range(1, GENERATIONS + 1))

    best_fitness_pct  = [v * 100 for v in storia_best_fitness]
    best_accuracy_pct = [v * 100 for v in storia_best_accuracy]
    mean_accuracy_pct = [v * 100 for v in storia_mean_accuracy]
    baseline_pct      = accuracy_baseline * 100

    max_hidden = max(len(best_individuo), len(cromosoma_baseline))
    font_info  = max(8, 12 - max(0, max_hidden+1))
    
    fig_height = max(8, 6.5 + max_hidden * 0.5)

    fig, axes = plt.subplot_mosaic([
         ['accuracy', 'info'],
         ['accuracy', 'info'],
         ['fitness', 'fitness']],
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
                color="#344966", linewidth=2,
                label="Miglior accuracy della generazione")
    ax_acc.plot(linespace_gen, mean_accuracy_pct,
                color="#D57D00", linewidth=1.5, linestyle='--',
                label="Accuracy media della popolazione")
    ax_acc.axhline(baseline_pct, color="#C30000B2", linestyle='dashed', linewidth=1.8,
                   label=f"Baseline backprop ({baseline_pct:.1f}%)")

    # annotazione valore finale best accuracy
    ax_acc.annotate(
        f"{best_accuracy:.1f}%",
        xy=(idx_best + 1, best_accuracy),
        xytext=(0, -20), textcoords='offset points',
        fontsize=10, color='#228B22', fontweight='extra bold',
        arrowprops=dict(arrowstyle='->', color='#228B22', lw=1.5)
    )
    ax_acc.scatter(idx_best + 1, best_accuracy, c='#228B22', marker='o')

    ax_acc.set_ylabel("Accuracy (%)", fontsize=11)
    ax_acc.set_ylim(0, 110)
    ax_acc.legend(loc='lower right', fontsize=9, framealpha=0.9)
    ax_acc.grid(True, alpha=0.3)
    ax_acc.set_title(
        "Accuratezza — quanto bene classifica la rete migliore e la popolazione media",
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

    ax_fit.scatter(idx_best + 1, best_fitness, c='#228B22', marker='o')

    ax_fit.set_xlabel("Generazione", fontsize=11)
    ax_fit.set_ylabel("Fitness", fontsize=11)
    ax_fit.set_xlim(1, GENERATIONS)
    ax_fit.set_ylim(0, 110)
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
    testo_best += f"\nOutput: 3 neuroni → {output_function.__name__}"
    testo_best += f"\n\nAccuracy: {best_accuracy}%"
    testo_best += f"\nFitness: {best_fitness}"
    testo_best += f"\n#parametri: {number_params(input_size, best_individuo, n_classi)}"

    testo_base = f"Architettura Baseline:\n\nInput: {input_size} neuroni\n\n"
    for i, (neuroni, funzione) in enumerate(cromosoma_baseline):
        testo_base += f"Hidden layer {i + 1}: {neuroni:<3} neuroni → {funzione.__name__}\n"
    testo_base += f"\nOutput: 3 neuroni → {output_function.__name__}"
    testo_base += f"\n\nAccuracy: {round(accuracy_baseline * 100, 2)}%"
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
    


