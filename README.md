# GA on Neural Network

Implementazione didattica di una rete neurale **Multi-Layer Perceptron (MLP)** scritta da zero in Python, combinata con un **algoritmo genetico** per la ricerca automatica dell'architettura ottimale (Neural Architecture Search).

L'obiettivo è mostrare come funziona il processo di apprendimento "sotto il cofano", usando esclusivamente logica matriciale con `numpy`, senza appoggiarsi a framework come TensorFlow o PyTorch. Il GA evolve la topologia della rete (layer, neuroni, funzioni di attivazione); la backpropagation apprende i pesi al suo interno.

---

## Indice

- [Caratteristiche principali](#caratteristiche-principali)
- [Struttura del progetto](#struttura-del-progetto)
- [Architettura del sistema](#architettura-del-sistema)
- [Algoritmo genetico](#algoritmo-genetico)
- [Requisiti e installazione](#requisiti-e-installazione)
- [Utilizzo](#utilizzo)
- [Test di sensibilità ai parametri](#test-di-sensibilità-ai-parametri)
- [Documentazione](#documentazione)

---

## Caratteristiche principali

- Rete neurale MLP completamente personalizzabile (layer, neuroni, funzioni di attivazione)
- Backpropagation con SGD e **gradient clipping** (previene l'esplosione dei gradienti)
- Inizializzazione pesi **He** (ReLU/Leaky ReLU) e **Xavier** (altre funzioni)
- Loss: **cross-entropy + softmax** con gradiente combinato
- Funzioni di attivazione: ReLU, Leaky ReLU, Sigmoid, Linear, Softmax
- **Algoritmo genetico** per architecture search su 4 dataset sklearn
- Valutazione parallela degli individui con `ProcessPoolExecutor`
- Script di test per 9 iperparametri con output CSV + grafici comparativi

---

## Struttura del progetto

```text
GA_on_neural_network/
├── neural_network/                  # Modulo principale
│   ├── __init__.py
│   ├── nn_engine.py                 # Classe neural_network (feedforward + backprop)
│   ├── nn_layer.py                  # Classe layer e funzioni di attivazione
│   └── genetic_algorithm.py        # Algoritmo genetico per architecture search
├── tests_script/                    # Script di test per ogni iperparametro
│   ├── test_1_learning_rate.py
│   ├── test_2_epochs.py
│   ├── test_3_lambda.py
│   ├── test_4_k.py
│   ├── test_5_population.py
│   ├── test_6_generations.py
│   ├── test_7_mutation_rate.py
│   ├── test_8_dataset.py
│   ├── test_9A_leakage.py
│   └── test_9B_leakage.py
├── tests_img/                       # Output dei test (CSV + PNG)
│   └── summary.csv                  # CSV consolidato con tutti i valori ottimali
├── docs/
│   ├── SRS.md                       # Specifiche dei requisiti software
│   ├── appunti.md                   # Note tecniche
│   └── analisi_test.md              # Analisi dettagliata dei risultati
├── img/                             # Grafici generati da main.py
├── main.py                          # Entry point e interfaccia run()
├── guida_test.md                    # Guida all'esecuzione dei test nell'ordine corretto
├── considerazioni.md                # Motivazioni delle scelte per ogni iperparametro
└── requirements.txt
```

---

## Architettura del sistema

### `neural_network/nn_engine.py` — Il motore della rete

Contiene la classe `neural_network`.

**Costruttore:**

| Parametro | Tipo | Descrizione |
|---|---|---|
| `hidden_config` | `list[tuple]` | Lista di `(n_neuroni, funzione_attivazione)` per ogni layer nascosto |
| `size_input` | `int` | Numero di feature in input |
| `size_output` | `int` | Numero di neuroni nel layer di output |
| `learning_rate` | `float` | Tasso di apprendimento per la backpropagation |
| `output_function` | `callable` | Funzione di attivazione del layer di output |
| `rng` | `np.random.Generator \| None` | Generatore NumPy per riproducibilità deterministica |

**Metodi principali:**

- `feedforward(x)` — Forward pass; restituisce `dict` con `guess`, `input_vals`, `intermediate_vals`
- `feedback(inputs, target)` — Backpropagation con gradient clipping; aggiorna pesi e bias
- `cross_entropy(target, guess)` — Cross-entropy loss
- `dCE_softmax(target, guess)` — Gradiente combinato CE + softmax (`guess - target`)

### `neural_network/nn_layer.py` — Layer e funzioni di attivazione

| Funzione | Formula | Inizializzazione pesi consigliata |
|---|---|---|
| `relu` | `max(0, x)` | He: `sqrt(2/col)` |
| `leaky_relu` | `max(0.01x, x)` | He: `sqrt(2/col)` |
| `sigmoid` | `1 / (1 + e^-x)` | Xavier: `sqrt(1/col)` |
| `linear` | `x` | Xavier: `sqrt(1/col)` |
| `softmax` | `e^x / Σ(e^x)` | Xavier: `sqrt(1/col)` |

I pesi vengono inizializzati automaticamente con He o Xavier a seconda della funzione di attivazione del layer.

### `neural_network/genetic_algorithm.py` — Algoritmo genetico

Vedi la sezione dedicata qui sotto.

---

## Algoritmo genetico

### Rappresentazione del cromosoma

Ogni individuo è una lista di tuple che definiscono i layer nascosti:

```python
[(24, relu)]               # 1 layer nascosto: 24 neuroni ReLU
[(16, relu), (8, sigmoid)] # 2 layer nascosti
```

**Vincoli di ricerca (default):**

| Parametro | Range |
|---|---|
| Numero di layer nascosti | 1 – 10 |
| Neuroni per layer | 4 – 64 |
| Funzioni di attivazione | ReLU, Leaky ReLU, Sigmoid, Linear, Softmax |

### Funzione di fitness

```
fitness = accuracy - λ × log10(n_params)
```

L'uso di `log10` attenua la penalità per le reti di grandi dimensioni, permettendo al GA di trovare architetture compatte senza svantaggiare eccessivamente quelle più grandi.

### Operatori genetici

| Operatore | Descrizione |
|---|---|
| **Selezione** | Tournament selection |
| **Crossover** | Punto di taglio variabile e indipendente per ciascun genitore |
| **Mutazione** | Modifica neuroni (tipo 0), funzione di attivazione (tipo 1), aggiunge/rimuove layer (tipo 2) |

### Iperparametri ottimali (trovati sperimentalmente su `load_digits`)

| Parametro | Valore ottimale | Default `main.py` |
|---|---|---|
| `epochs` | 500 | 500 |
| `K` | 3 | 20 |
| `learning_rate` | 0.05 | 0.01 |
| `lambda_` | 0.05 | 0.01 |
| `mutation_rate` | 0.2 | 0.2 |
| `population_size` | 20 | 30 |
| `generations` | 50 | 20 |
| `tournament_size` | 5 | 10 |

I valori ottimali e le motivazioni di ogni scelta sono documentati in [`considerazioni.md`](considerazioni.md).

---

## Requisiti e installazione

**Python 3.x** richiesto.

```bash
# Clona il repository
git clone <url-repo>
cd GA_on_neural_network

# Crea e attiva un ambiente virtuale
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/macOS

# Installa le dipendenze
pip install -r requirements.txt
```

**Dipendenze:**

| Libreria | Versione | Utilizzo |
|---|---|---|
| `numpy` | 2.4.4 | Operazioni matriciali |
| `matplotlib` | 3.10.8 | Visualizzazione dei risultati |
| `scikit-learn` | 1.8.0 | Dataset, preprocessing, split |
| `scipy` | 1.17.1 | Calcolo scientifico |

---

## Utilizzo

### Entry point principale

```bash
python main.py
```

Esegue il GA su `load_digits` con i parametri default e salva i grafici in `img/`.

### Interfaccia `run()`

La funzione `run()` in `main.py` è l'interfaccia principale per eseguire esperimenti:

```python
from main import run
from sklearn.datasets import load_digits

result = run(
    dataset=load_digits(),
    population=20,
    generations=50,
    mutation_rate=0.2,
    tournament_size=5,
    epochs=500,
    learning_rate=0.05,
    lambda_=0.05,
    K=3,
    seed=42,
    plot=False,
    max_workers=2   # limita i processi paralleli (consigliato su macchine con poca RAM)
)
```

**Valori restituiti:**

```python
result['best_individuo']       # architettura trovata, es. [(24, 'relu')]
result['best_fitness']         # fitness del miglior individuo (percentuale)
result['best_accuracy']        # accuracy sul val set — ottimistica (percentuale)
result['test_accuracy']        # accuracy sul test set — stima reale (percentuale)
result['accuracy_baseline']    # accuracy rete baseline fissa (percentuale)
result['n_params']             # #parametri architettura GA
result['n_params_baseline']    # #parametri architettura baseline
result['storia_best_fitness']  # lista float, una per generazione
result['storia_best_accuracy'] # lista float, una per generazione
result['storia_mean_accuracy'] # lista float, una per generazione
```

### Uso diretto della rete neurale

```python
from neural_network import neural_network
from neural_network.nn_layer import relu, softmax
import numpy as np

nn = neural_network(
    hidden_config=[(24, relu)],
    size_input=64,
    size_output=10,
    learning_rate=0.05,
    output_function=softmax
)

# Un passo di addestramento
nn.feedback(X_train[0], y_train_onehot[0])

# Predizione
risultato = nn.feedforward(X_test[0])
classe_predetta = np.argmax(risultato['guess'])
```

---

## Test di sensibilità ai parametri

Tutti gli script di test si trovano in `tests_script/`. Ogni script varia un solo parametro tenendo fissi gli altri, e salva i risultati in `tests_img/` come CSV + grafici PNG.

Per l'ordine corretto di esecuzione e i criteri di scelta dei valori ottimali, vedere [`guida_test.md`](guida_test.md).

| Script | Parametro testato |
|---|---|
| `test_1_learning_rate.py` | `learning_rate` |
| `test_2_epochs.py` | `epochs` |
| `test_3_lambda.py` | `lambda_` |
| `test_4_k.py` | `K` |
| `test_5_population.py` | `population` |
| `test_6_generations.py` | `generations` |
| `test_7_mutation_rate.py` | `mutation_rate` |
| `test_8_dataset.py` | generalizzabilità su 4 dataset |
| `test_9A_leakage.py` | bias ottimistico val vs test |
| `test_9B_leakage.py` | leakage da preprocessing |

