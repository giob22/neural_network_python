# Neural Network Python

Implementazione didattica di una rete neurale **Multi-Layer Perceptron (MLP)** scritta completamente da zero in Python. L'obiettivo è mostrare in modo pratico come funziona il processo di apprendimento "sotto il cofano", usando esclusivamente logica matriciale con `numpy`, senza appoggiarsi a framework come TensorFlow o PyTorch.

Include un **algoritmo genetico** per la ricerca automatica dell'architettura ottimale della rete.

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

- Rete neurale MLP completamente personalizzabile (numero di layer, neuroni, funzioni di attivazione)
- Apprendimento tramite **backpropagation** con discesa del gradiente
- Funzioni di attivazione: ReLU, Leaky ReLU, Sigmoid, Linear, Softmax (con derivate)
- **Algoritmo genetico** per architecture search automatizzato
- Dataset di riferimento: **Iris** (via scikit-learn)
- Visualizzazioni dei risultati con matplotlib
- Test di sensibilità per i principali iperparametri

---

## Struttura del progetto

```text
neural_network_python/
├── neural_network/               # Modulo principale
│   ├── __init__.py               # Esporta le classi e le funzioni principali
│   ├── nn_engine.py              # Motore della rete (feedforward + backpropagation)
│   ├── nn_layer.py               # Classe layer e funzioni di attivazione
│   └── genetic_algorithm.py     # Algoritmo genetico per architecture search
├── examples/
│   └── xor_example.py            # Esempio: apprendimento della porta logica XOR
├── docs/
│   ├── SRS.md                    # Specifiche dei requisiti software
│   ├── appunti.md                # Note tecniche (one-hot, softmax, normalizzazione)
│   └── analisi_test.md           # Analisi dettagliata dei risultati dei test
├── img/
│   └── risultati.png             # Grafici generati da main.py
├── tests/                        # Output dei test di sensibilità (CSV + PNG)
├── main.py                       # Entry point: Iris + algoritmo genetico
├── main_test.py                  # Test suite con visualizzazione
├── test_epochs.py                # Sensibilità al numero di epoche
├── test_learning_rate.py         # Sensibilità al learning rate
├── test_lambda.py                # Sensibilità al coefficiente di penalità λ
├── test_mutation_rate.py         # Sensibilità al tasso di mutazione
└── requirements.txt
```

---

## Architettura del sistema

### `neural_network/nn_engine.py` — Il motore della rete

Contiene la classe `neural_network`. Gestisce la composizione multi-layer, il forward pass e l'aggiornamento dei pesi tramite backpropagation.

**Parametri del costruttore:**

| Parametro | Tipo | Descrizione |
|---|---|---|
| `hidden_config` | `list[tuple]` | Lista di `(n_neuroni, funzione_attivazione)` per ogni layer nascosto |
| `size_input` | `int` | Dimensione dell'input |
| `size_output` | `int` | Dimensione dell'output |
| `learning_rate` | `float` | Tasso di apprendimento per la backpropagation |
| `output_function` | `callable` | Funzione di attivazione del layer di output |

**Metodi principali:**

- `feedforward(x)` — Forward pass; restituisce la predizione della rete
- `feedback(input, target)` — Backpropagation; aggiorna i pesi in base all'errore
- `MSE(target, guess)` — Calcola il Mean Squared Error
- `dMSE(target, guess)` — Calcola il gradiente del MSE

### `neural_network/nn_layer.py` — I singoli layer

Contiene la classe `layer` (pesi, bias, funzione di attivazione) e le funzioni di attivazione con le relative derivate:

| Funzione | Formula | Uso consigliato |
|---|---|---|
| `relu` | `max(0, x)` | Layer nascosti |
| `leaky_relu` | `max(0.01x, x)` | Layer nascosti (evita neuroni morti) |
| `sigmoid` | `1 / (1 + e^-x)` | Classificazione binaria |
| `linear` | `x` | Regressione |
| `softmax` | `e^x / Σ(e^x)` | Output multi-classe |

I pesi sono inizializzati con distribuzione uniforme in `[-0.5, 0.5]`.

### `neural_network/genetic_algorithm.py` — Algoritmo genetico

Ottimizzazione evolutiva dell'architettura della rete. Vedi la sezione dedicata qui sotto.

---

## Algoritmo genetico

L'algoritmo genetico (`GeneticAlgorithm`) esplora automaticamente lo spazio delle possibili architetture senza progettazione manuale.

### Rappresentazione del cromosoma

Ogni individuo è una lista di tuple che definiscono i layer nascosti:

```python
[(16, relu), (8, sigmoid)]   # 2 layer nascosti: 16 nodi ReLU + 8 nodi Sigmoid
```

**Vincoli di ricerca:**
- Numero di layer nascosti: 1–4
- Neuroni per layer: 4–32
- Funzioni di attivazione: ReLU, Leaky ReLU, Sigmoid
- Layer di output: fisso (Softmax, 3 neuroni per Iris)

### Funzione di fitness

```
fitness = accuracy - λ × complessità
```

dove `complessità` è il numero totale di parametri della rete. Il termine `λ` penalizza le architetture troppo grandi, favorendo soluzioni compatte.

### Operatori genetici

| Operatore | Descrizione |
|---|---|
| **Selezione** | Tournament selection (dimensione torneo: 5) |
| **Crossover** | Scambio di porzioni di architettura tra due genitori |
| **Mutazione** | Modifica neuroni, funzione di attivazione, aggiunge/rimuove un layer |

### Iperparametri principali (default in `main.py`)

| Parametro | Valore | Descrizione |
|---|---|---|
| `POPULATION_SIZE` | 20 | Individui per generazione |
| `GENERATIONS` | 30 | Numero di generazioni |
| `MUTATION_RATE` | 0.2 | Probabilità di mutazione per layer |
| `TOURNAMENT_SIZE` | 5 | Dimensione del torneo di selezione |
| `K` | 5 | Ripetizioni di training per ogni individuo |
| `EPOCHS` | 300 | Epoche di addestramento per valutazione |
| `LEARNING_RATE` | 0.001 | Learning rate della backpropagation |
| `LAMBDA` | 0.0005 | Coefficiente di penalità sulla complessità |

---

## Requisiti e installazione

**Python 3.x** richiesto.

```bash
# Clona il repository
git clone https://github.com/giob22/neural_network_python.git
cd neural_network_python

# Crea e attiva un ambiente virtuale
python -m venv .venv
source .venv/bin/activate       # Linux/macOS
# .venv\Scripts\activate        # Windows

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
| `joblib` | 1.5.3 | Elaborazione parallela |

---

## Utilizzo

### Esempio rapido — XOR

```bash
python examples/xor_example.py
```

Addestra una piccola rete a imparare la funzione logica XOR e stampa le predizioni per tutte e 4 le combinazioni di input.

### Entry point principale — Iris + Algoritmo genetico

```bash
python main.py
```

Il programma:
1. Carica e normalizza il dataset Iris
2. Addestra una rete baseline come confronto
3. Esegue l'algoritmo genetico (20 individui × 30 generazioni)
4. Mostra metriche di convergenza generazione per generazione
5. Salva i grafici in `img/risultati.png`:
   - Evoluzione dell'accuracy (best, media, baseline)
   - Evoluzione del fitness (con penalità complessità)
   - Migliore architettura trovata

### Utilizzo del modulo direttamente

```python
from neural_network import neural_network
from neural_network.nn_layer import relu, sigmoid, softmax

# Rete con 2 layer nascosti: 8 nodi (sigmoid) + 16 nodi (relu)
# 4 input, 3 output (classificazione Iris), learning rate 0.001
nn = neural_network(
    hidden_config=[(8, sigmoid), (16, relu)],
    size_input=4,
    size_output=3,
    learning_rate=0.001,
    output_function=softmax
)

# Un passo di addestramento
nn.feedback(input=[5.1, 3.5, 1.4, 0.2], target=[1, 0, 0])

# Predizione
predizione = nn.feedforward([5.1, 3.5, 1.4, 0.2])
print(predizione)  # es. [0.97, 0.02, 0.01]
```

### Uso dell'algoritmo genetico

```python
from neural_network import GeneticAlgorithm

ga = GeneticAlgorithm(
    X_train=X_train,
    y_train=y_train,
    X_val=X_val,
    y_val=y_val,
    population_size=20,
    generations=30,
    mutation_rate=0.2,
    lam=0.0005
)

best_architecture, best_fitness = ga.run()
print("Architettura ottimale:", best_architecture)
```

---

## Test di sensibilità ai parametri

Ogni script di test analizza l'impatto di un iperparametro sulla performance dell'algoritmo genetico. I risultati vengono salvati in `tests/` come file CSV e grafici PNG.

```bash
python test_epochs.py          # Effetto del numero di epoche
python test_learning_rate.py   # Effetto del learning rate
python test_lambda.py          # Effetto del coefficiente λ
python test_mutation_rate.py   # Effetto del tasso di mutazione
```

Per l'interpretazione dettagliata dei risultati, vedere [`docs/analisi_test.md`](docs/analisi_test.md).

---

## Documentazione

| File | Contenuto |
|---|---|
| [`docs/SRS.md`](docs/SRS.md) | Specifiche dei requisiti software: architettura, operatori genetici, workflow |
| [`docs/appunti.md`](docs/appunti.md) | Note tecniche: one-hot encoding, derivata softmax, normalizzazione |
| [`docs/analisi_test.md`](docs/analisi_test.md) | Analisi e interpretazione dei test di sensibilità |
