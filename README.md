# Neural Network Python

Implementazione didattica di una rete neurale (Multi-Layer Perceptron) scritta completamente da zero. L'obiettivo è mostrare in modo pratico come funziona il processo di apprendimento "sotto il cofano", usando solo logica matriciale con `numpy`, senza appoggiarsi a framework pronti.

Include anche un **algoritmo genetico** per la ricerca automatica dell'architettura ottimale.

## Struttura del progetto

```text
neural_network_python/
├── neural_network/           # Modulo principale
│   ├── __init__.py
│   ├── nn_engine.py          # Motore della rete (feedforward + backpropagation)
│   ├── nn_layer.py           # Strati e funzioni di attivazione
│   └── genetic_algorithm.py  # Algoritmo genetico per architecture search
├── examples/
│   └── xor_example.py        # Esempio: addestramento sulla porta logica XOR
├── docs/
│   ├── SRS.md                # Specifiche dei requisiti software
│   └── appunti.md            # Note tecniche
├── main.py                   # Entry point (dataset Iris + algoritmo genetico)
└── requirements.txt
```

## Moduli

### `neural_network/nn_engine.py` - Il motore della rete

Contiene la classe `neural_network`. Gestisce l'architettura multi-layer personalizzabile, il feedforward e l'aggiornamento dei pesi tramite backpropagation.

### `neural_network/nn_layer.py` - I singoli strati

Contiene la classe `layer` e le funzioni di attivazione con le relative derivate:

- `relu`, `leaky_relu`
- `sigmoid`
- `linear`
- `softmax`

### `neural_network/genetic_algorithm.py` - Algoritmo genetico

Cerca automaticamente la migliore architettura di rete tramite evoluzione. Ogni individuo è una lista di layer `[(n_neuroni, funzione), ...]`, valutato tramite accuracy sul dataset Iris.

## Requisiti e installazione

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Utilizzo base

```python
from neural_network import neural_network
from neural_network.nn_layer import relu, sigmoid

# Rete con 2 layer nascosti: 8 nodi (sigmoid) + 100 nodi (relu)
# 2 input, 1 output, learning rate 0.2
nn = neural_network(
    hidden_config=[(8, sigmoid), (100, relu)],
    size_input=2,
    size_output=1,
    learning_rate=0.2,
    output_function=sigmoid
)

# Addestramento
nn.feedback(input=[1, 0], target=[1])

# Predizione
predizione = nn.feedforward([1, 0])
```

## Esempi

```bash
# Esempio XOR
python examples/xor_example.py

# Entry point principale (Iris + algoritmo genetico)
python main.py
```
