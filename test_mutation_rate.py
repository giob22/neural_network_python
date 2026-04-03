import csv
import matplotlib.pyplot as plt
from neural_network import *
import numpy as np
import random
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ── preprocessing (identico a main.py) ──────────────────────────────
def one_hot(y, n_classes=3):
    Y = np.zeros((len(y), n_classes))
    for i, label in enumerate(y):
        Y[i][label] = 1
    return Y

dataset = load_iris()
x = dataset.data
y = dataset.target
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_val   = scaler.transform(x_val)
y_train = one_hot(y_train)
y_val   = one_hot(y_val)

# ── baseline ─────────────────────────────────────────────────────────
EPOCHS_BASELINE = 450
rete_baseline = neural_network([(8, relu), (8, relu)], 4, 3, 0.01, softmax)
for _ in range(EPOCHS_BASELINE):
    idx = random.randint(0, len(x_train) - 1)
    rete_baseline.feedback(x_train[idx], y_train[idx])
correct = sum(
    1 for i in range(len(x_val))
    if np.argmax(rete_baseline.feedforward(x_val[i])) == np.argmax(y_val[i])
)
accuracy_baseline = correct / len(x_val)
print(f"Baseline accuracy: {round(accuracy_baseline * 100, 2)}%")

# ── parametri fissi ───────────────────────────────────────────────────
POPULATION_SIZE  = 20
GENERATIONS      = 30
LEARNING_RATE    = 0.01
LAMBDA_          = 0.00005
EPOCHS           = 150
TOURNAMENT_SIZE  = 3

# ── valori da testare ─────────────────────────────────────────────────
mutation_rates = [0.05, 0.2, 0.4]

# ── strutture per raccogliere i risultati ─────────────────────────────
csv_rows = []
all_storie = []  # per i grafici comparativi

for mr in mutation_rates:
    print(f"\n{'='*50}")
    print(f"TEST mutation_rate = {mr}")
    print(f"{'='*50}")

    ga = GeneticAlgorithm(
        population_size = POPULATION_SIZE,
        generations     = GENERATIONS,
        mutation_rate   = mr,
        tournament_size = TOURNAMENT_SIZE,
        epochs          = EPOCHS,
        learning_rate   = LEARNING_RATE,
        lambda_         = LAMBDA_,
        X_Train=x_train, Y_Train=y_train,
        X_val=x_val,     Y_val=y_val
    )

    (best_ind, best_fit, best_acc,
     storia_bf, storia_ba, storia_ma) = ga.run()

    n_layer  = len(best_ind)
    n_params = ga._complessita(best_ind)
    arch     = [(n, f.__name__) for n, f in best_ind]

    # salva scalari per CSV
    csv_rows.append({
        'mutation_rate':    mr,
        'best_accuracy':    round(best_acc * 100, 2),
        'best_fitness':     round(best_fit * 100, 2),
        'n_layer':          n_layer,
        'n_params':         n_params,
        'accuracy_baseline': round(accuracy_baseline * 100, 2),
        'architettura':     str(arch)
    })

    # salva serie temporali per i grafici
    all_storie.append({
        'label':      f"mr={mr}",
        'storia_bf':  storia_bf,
        'storia_ba':  storia_ba,
        'storia_ma':  storia_ma
    })

    print(f"Best accuracy: {round(best_acc * 100, 2)}%")
    print(f"Architettura: {arch}")

# ── salva CSV ─────────────────────────────────────────────────────────
with open('tests/test_mutation_rate.csv', 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
    writer.writeheader()
    writer.writerows(csv_rows)
print("\nCSV salvato: test_mutation_rate.csv")

# ── grafici comparativi ───────────────────────────────────────────────
gen = range(1, GENERATIONS + 1)

fig, axes = plt.subplot_mosaic(
    [['best_acc', 'mean_acc'],
     ['best_fit', 'mean_acc']],
    figsize=(14, 8)
)

for storia in all_storie:
    axes['best_acc'].plot(gen, storia['storia_ba'], label=storia['label'])
    axes['best_fit'].plot(gen, storia['storia_bf'], label=storia['label'])
    axes['mean_acc'].plot(gen, storia['storia_ma'], label=storia['label'])

# baseline su best_acc e mean_acc
for ax_key in ['best_acc', 'mean_acc']:
    axes[ax_key].axhline(y=accuracy_baseline, color='red',
                         linestyle='dashed', label='baseline')

axes['best_acc'].set_title("Best accuracy per generazione")
axes['best_acc'].set_ylabel("Accuracy")
axes['best_acc'].set_xlabel("Generazione")
axes['best_acc'].legend()

axes['best_fit'].set_title("Best fitness per generazione")
axes['best_fit'].set_ylabel("Fitness")
axes['best_fit'].set_xlabel("Generazione")
axes['best_fit'].legend()

axes['mean_acc'].set_title("Mean accuracy per generazione")
axes['mean_acc'].set_ylabel("Accuracy")
axes['mean_acc'].set_xlabel("Generazione")
axes['mean_acc'].legend()

plt.suptitle(f"Test mutation_rate — baseline: {round(accuracy_baseline*100,2)}%",
             fontsize=14)
plt.tight_layout()
plt.savefig('tests/test_mutation_rate.png', dpi=150)
plt.show()
print("Grafico salvato: test_mutation_rate.png")