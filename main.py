from neural_network import *
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def one_hot(y, n_classes=3):
    Y = np.zeros((len(y), n_classes))
    for i, label in enumerate(y):
        Y[i][label] = 1
    return Y


K = 3 # numero di addestramenti per individuo

POPULAZION_SIZE = 20
GENERATIONS = 20
MUTATION_RATE = 0.2
TOURNAMENT_SIZE = 3
EPOCHS =  120 #150
LEARNING_RATE = 0.01
LAMBDA_ = 0.00005

EPOCHS_BASELINE = K * EPOCHS


if __name__ == "__main__":

    # load del dataset
    dataset = load_iris()
    
    x = dataset.data
    y = dataset.target

    # split train/validation

    x_train, x_val, y_train, y_val = train_test_split(x,y,test_size=0.2, random_state=42)


    # usiamo random_state=42 per fissare il seed in modo che il risultato sia riproducibile
    # ogni volta che eseguo il programma ottengo lo stesso split, mi permette di confrontare i risultati tra run diverse

    # Normalizzazione
    # X ← (X - media(X)) / std(X)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_val = scaler.transform(x_val)

    
    y_train = one_hot(y_train)
    y_val = one_hot(y_val)


    # BASELINE con backpropagation classica
    # rete neurale addestrata per lo stesso quantitativo di epoche 
    # mi permette di confrontare il risultato del GA
    cromosoma_baseline = [(8,leaky_relu),(8,leaky_relu),(8,leaky_relu),(8,leaky_relu)]
    rete_baseline = neural_network(hidden_config=cromosoma_baseline,size_input=4 , size_output=3, learning_rate=0.01, output_function=softmax)

    for i in range(0, EPOCHS_BASELINE):
        idx = random.randint(0, len(x_train) - 1)
        rete_baseline.feedback(x_train[idx], y_train[idx])

    # valutiamo la baseline sul validation set
    correct = 0
    for i in range(0, len(x_val)):
        guess = rete_baseline.feedforward(x_val[i])
        if np.argmax(guess) == np.argmax(y_val[i]):
            correct += 1
    accuracy_baseline = correct/len(x_val)
    print(f"accuracy della rete baseline: {round((accuracy_baseline) * 100, 2)}%")

    # ESECUZIONE DEL Genetic Algorithm

    ga = GeneticAlgorithm(population_size=POPULAZION_SIZE,
                          generations=GENERATIONS,
                          mutation_rate=MUTATION_RATE,
                          tournament_size=TOURNAMENT_SIZE,
                          epochs=EPOCHS,
                          learning_rate=LEARNING_RATE,
                          lambda_=LAMBDA_,
                          X_Train=x_train, Y_Train=y_train,
                          X_val=x_val, Y_val=y_val)
    
    (best_individuo, best_fitness, best_accuracy, storia_best_fitness, storia_best_accuracy, storia_mean_accuracy) = ga.run()
    best_individuo = [(neuroni, funzione.__name__) for neuroni, funzione in best_individuo]
    print(f"Miglior architettura trovata:\n{best_individuo}")
    print(f"Fitness:{round(best_fitness * 100,2)}")
    print(f"Accuracy:{round(best_accuracy * 100,2)}%")
    print(f"Numero di layer={len(best_individuo)}")

    
    linespace_gen = list(range(1, GENERATIONS + 1))

    best_fitness_pct  = [v * 100 for v in storia_best_fitness]
    best_accuracy_pct = [v * 100 for v in storia_best_accuracy]
    mean_accuracy_pct = [v * 100 for v in storia_mean_accuracy]
    baseline_pct      = accuracy_baseline * 100

    fig, axes = plt.subplot_mosaic([
         ['fitness', 'architettura'],
         ['accuracy', 'accuracy']],
         figsize=(10,8))


    ax_fit  = axes['fitness']
    ax_acc  = axes['accuracy']
    ax_arch = axes['architettura']

    # fig, (ax_acc, ax_fit) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    fig.suptitle(
        "Algoritmo Genetico — ricerca dell'architettura ottimale\n"
        "(Dataset Iris, classificazione 3 classi)",
        fontsize=13, fontweight='bold'
    )

    # --- Grafico superiore: Accuracy ---
    ax_acc.plot(linespace_gen, best_accuracy_pct,
                color="steelblue", linewidth=2,
                label="Miglior accuracy della generazione")
    ax_acc.plot(linespace_gen, mean_accuracy_pct,
                color="seagreen", linewidth=1.5, linestyle='--',
                label="Accuracy media della popolazione")
    ax_acc.axhline(baseline_pct, color="crimson", linestyle='dashed', linewidth=1.8,
                   label=f"Baseline backprop ({baseline_pct:.1f}%)")

    # annotazione valore finale best accuracy
    ax_acc.annotate(
        f"{best_accuracy_pct[-1]:.1f}%",
        xy=(GENERATIONS, best_accuracy_pct[-1]),
        xytext=(0, 12), textcoords='offset points',
        fontsize=9, color='steelblue',
        arrowprops=dict(arrowstyle='->', color='steelblue', lw=1.2)
    )

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
                color="darkorange", linewidth=2,
                label="Miglior fitness della generazione")
    ax_fit.axhline(baseline_pct, color="crimson", linestyle='dashed', linewidth=1.8,
                   label=f"Baseline backprop ({baseline_pct:.1f}%)")

    ax_fit.annotate(
        f"{best_fitness_pct[-1]:.1f}",
        xy=(GENERATIONS, best_fitness_pct[-1]),
        xytext=(0, 15), textcoords='offset points',
        fontsize=9, color='darkorange',
        arrowprops=dict(arrowstyle='->', color='darkorange', lw=1.2)
    )

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

    # grafico a destra: architettura
    ax_arch.axis('off')
    testo = "Input: 4 neuroni\n\n"
    for i, (neuroni, funzione) in enumerate(best_individuo):
            testo += f"Hidden layer {i + 1}: {neuroni} neuroni → {funzione}\n"
    testo += f"\n Output: 3 neuroni → softmax"
    testo += f"\n\nAccuracy: {round(best_accuracy * 100,2)}%"
    testo += f"\nFitness: {round(best_fitness * 100, 2)}"
    ax_arch.text(0.5,0.5, testo, transform=ax_arch.transAxes, fontsize=12, verticalalignment='center', horizontalalignment='center', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

    ax_arch.set_title("Migliore architettura trovata")



    plt.tight_layout()
    plt.savefig('img/risultati.png', dpi=150)
    plt.show()
    


