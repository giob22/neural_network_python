# Feature backlog — GA on Neural Network

Modifiche da apportare per rendere il progetto coerente e completo.

---

## 1. Dataset come parametro di `run()`

**Stato:** non implementato  
**File:** `main.py`

`load_iris()` è hardcoded dentro `run()`. Per testare il GA su dataset diversi (digits, wine, breast_cancer) bisogna duplicare il codice.

**Modifica:**
```python
def run(dataset=None, ...):
    if dataset is None:
        dataset = load_iris()
    ...
```

Il preprocessing (split, normalizzazione, one_hot) funziona già in modo generico — basta passare il dataset come parametro.

---

## 2. Rimuovere il TODO non risolto in `nn_layer.py`

**Stato:** non implementato  
**File:** `neural_network/nn_layer.py`, riga 158

```python
# TODO spiega perché si sta utilizzando un modo diverso per assegnare i valori ai pesi...
```

Va rimosso o convertito in commento Doxygen che spiega He vs Xavier.

---

## 3. Early stopping nel GA

**Stato:** non implementato  
**File:** `neural_network/genetic_algorithm.py`

Il GA esegue esattamente `generations` iterazioni anche se la fitness non migliora da N generazioni. Aggiungere un parametro `patience` (es. default 5): se `best_fitness` non migliora per `patience` generazioni consecutive, fermarsi.

**Parametro da aggiungere a `GeneticAlgorithm.__init__`:**
```python
patience=5  # None = disabilita early stopping
```

**Utilità per lo studio:** permette di riportare il numero di generazioni effettive alla convergenza, non solo il limite massimo.

---

## 4. `max_workers` come parametro di `GeneticAlgorithm`

**Stato:** non implementato  
**File:** `neural_network/genetic_algorithm.py`

`ProcessPoolExecutor()` senza `max_workers` usa tutti i core disponibili. Su macchine con molti core o con K grande questo può saturare la memoria.

```python
with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
```

**Parametro da aggiungere:**
```python
max_workers=None  # None = usa tutti i core (comportamento attuale)
```

---

## 5. Aggiungere `n_params` nel dict di ritorno di `run()`

**Stato:** non implementato  
**File:** `main.py`

Il numero di parametri della miglior architettura è calcolato ma non incluso nel dict di ritorno. I test che vogliono confrontare complessità vs accuracy devono ricalcolarlo.

```python
return {
    ...
    'n_params': number_params(input_size, best_individuo_str, n_classi),
    ...
}
```

---

## 6. Aggiornare i file `test_*.py` per usare `run()`

**Stato:** non implementato  
**File:** `test_lambda.py`, `test_mutation_rate.py`, `test_learning_rate.py`, `test_epochs.py`, `test_lambda_log.py`

Tutti i file di test replicano il preprocessing (split, normalizzazione, one_hot) e chiamano `GeneticAlgorithm` direttamente. Ora che `run()` è callable con `plot=False`, i test possono importare `run` da `main` ed eliminare il codice duplicato.

**Pattern atteso dopo la modifica:**
```python
from main import run

result = run(lambda_=0.0, generations=30, K=3, plot=False)
best_acc      = result['best_accuracy']
test_acc      = result['test_accuracy']
storia_ba     = result['storia_best_accuracy']  # già in percentuale
```
