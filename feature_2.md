# Review — GA on Neural Network

Problemi trovati dopo il review del codice. Ordinati per priorità.

---

## 1. Pesi mutazione invertiti rispetto al docstring

**Severity:** WARNING  
**File:** `neural_network/genetic_algorithm.py`, righe 228 e 243

Il docstring di `_mutazione` dichiara pesi `[20, 20, 1]` per i tre tipi di mutazione, ma il codice usa `[2, 2, 20]` — esattamente l'inverso.

```python
# docstring dice:
con pesi [20, 20, 1]:
- tipo 0: cambia neuroni      ← frequente (20)
- tipo 1: cambia attivazione  ← frequente (20)
- tipo 2: add/remove layer    ← raro (1)

# codice fa:
mut = self.rng.choices([0, 1, 2], [2, 2, 20])[0]
# tipo 2 (add/remove layer) ha peso 20 → 83% dei casi
```

Con `[2, 2, 20]` ogni gene mutato ha l'83% di probabilità di aggiungere o rimuovere un intero layer, rendendo il GA iperdistruttivo sulla topologia e impedendo il fine-tuning. Il docstring (`[20, 20, 1]`) descrive il comportamento ragionevole: tuning frequente, ristrutturazione rara.

**Fix:**
```python
mut = self.rng.choices([0, 1, 2], [20, 20, 1])[0]
```

---

## 2. Baseline non deterministica nei test script

**Severity:** WARNING  
**File:** `test_epochs.py:36`, `test_learning_rate.py:36`, `test_lambda.py:36`, `test_mutation_rate.py:36`

Il GA usa `seed=42`, ma il loop di addestramento della baseline usa `random.randint` globale mai seeded. Ogni esecuzione produce una baseline diversa, rendendo il confronto GA vs baseline inaffidabile tra run diverse.

```python
# attuale — non deterministico
idx = random.randint(0, len(x_train) - 1)
```

**Fix:** aggiungere `random.seed(42)` prima del loop baseline, oppure usare un RNG isolato:
```python
_rng = random.Random(42)
for _ in range(EPOCHS_BASELINE):
    idx = _rng.randint(0, len(x_train) - 1)
    rete_baseline.feedback(x_train[idx], y_train[idx])
```

---

## 3. Mutable default argument in `run()`

**Severity:** WARNING  
**File:** `main.py:119`

```python
def run(..., dataset=load_iris(), ...):
```

`load_iris()` è valutata **una sola volta all'import del modulo**, non ad ogni chiamata. Se un caller modifica `dataset.data` in-place, le chiamate successive a `run()` senza argomento esplicito ricevono dati già alterati.

**Fix:**
```python
def run(..., dataset=None, ...):
    if dataset is None:
        dataset = load_iris()
```

---

## 4. `plt.savefig` crasha se la directory non esiste

**Severity:** WARNING  
**File:** `main.py:407`

```python
plt.savefig(f'{img_path}.png', dpi=150)
```

Se `img_path="nuova_dir/results"` e `nuova_dir/` non esiste, solleva `FileNotFoundError` senza messaggio utile. Stesso problema nei test script con la cartella `tests/` (attualmente esiste, ma non è garantita).

**Fix:**
```python
import os
os.makedirs(os.path.dirname(img_path) or '.', exist_ok=True)
plt.savefig(f'{img_path}.png', dpi=150)
```

---

## 5. `one_hot` duplicata in 4 file di test

**Severity:** SUGGESTION  
**File:** `test_epochs.py:15`, `test_learning_rate.py:15`, `test_lambda.py:15`, `test_mutation_rate.py:15`

Copia identica della funzione già definita in `main.py`. Qualsiasi modifica va replicata manualmente in tutti e quattro i file.

**Fix:** importare direttamente da `main`:
```python
from main import one_hot
```

Nota: questo diventa automatico se i test vengono migrati a usare `run()` come proposto in `feature_1.md` §6.

---

## 6. Magic numbers nel blocco plot

**Severity:** SUGGESTION  
**File:** `main.py`, blocco `if plot:`

Valori numerici senza nome sparsi nel codice grafico:

| Riga | Valore | Significato |
|------|--------|-------------|
| 299, 309 | `-60, -20` | offset annotazioni val/test (punti) |
| 357 | `15` | offset annotazione fitness (punti) |
| 319, 320 | `-5, +5` | padding asse Y (%) |
| 392, 393 | `0.02` | padding pannello info (fraction) |

Sostituire con costanti nominate all'inizio del blocco plot per rendere il tuning visivo immediato.
