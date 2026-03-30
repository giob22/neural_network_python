# Neural Network Python

Questo progetto è un'implementazione didattica di una rete neurale (Multi-Layer Perceptron) scritta completamente da zero. L'obiettivo è mostrare in modo pratico come funziona il processo di apprendimento "sotto il cofano", usando solo logica matriciale nuda e cruda con `numpy`, senza appoggiarsi a framework pronti.

## Struttura del codice

Il progetto è modulare e si trova all'interno della cartella `src/`:

### 1. `src/nn_engine.py` - Il motore della rete
Contiene la classe `neural_network`, che fa tutto il lavoro sporco per la rete flessibile. Gestisce la struttura multi-layer personalizzabile, calcola l'inoltro in avanti (`feedforward`) e gestisce l'aggiornamento dei pesi tramite l'algoritmo di backpropagation (`feedback`).

### 2. `src/nn_layer.py` - I singoli strati
Contiene la classe `layer`, che rappresenta i mattoni della rete. Ogni layer gestisce i propri pesi (`W`) e bias (`b`). Questo modulo include anche un dizionario con funzioni di attivazione e relative derivate per fare un po' di tuning:
- `relu` e `leaky_relu`
- `sigmoid`
- `linear`

### 3. `src/test_nn_engine.py` - Il file di test
Uno script di esempio che inizializza la rete e la addestra a risolvere la porta logica XOR, mostrando quanto rapidamente riesce a convergere sui dati corretti.

## Requisiti ed esecuzione

Moduli necessari:
- `numpy`

Per avviare l'addestramento di test del problema XOR:
```bash
python src/test_nn_engine.py
```

## Utilizzo base

Come creare una rete con un livello nascosto da 100 nodi:

```python
from src.nn_engine import neural_network
from src.nn_layer import relu, sigmoid

# Crea la rete: 1 livello nascosto, 100 nodi, 2 input, 1 output, lr=0.2
# Funzioni di attivazione: ReLU (nascosto), Sigmoide (output)
nn = neural_network(
    n_hidden_layer=1, 
    size_hidden=100, 
    size_input=2, 
    size_output=1, 
    learning_rate=0.2, 
    hidden_function=relu, 
    output_function=sigmoid
)

# Addestra (es. XOR validation)
nn.feedback(input=[1, 0], target=[1])

# Ottieni predizioni dalla rete
predizione = nn.feedforward([1, 0])
```
