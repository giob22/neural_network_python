# Guida all'implementazione dei test sperimentali

Ogni test varia **un solo parametro** tenendo fissi tutti gli altri.
Tutti i test usano `run(plot=False)` da `main.py` e salvano risultati in CSV + grafici comparativi.

---

## Parametri fissi di riferimento

Usare questi valori come default in tutti i test, salvo il parametro che si sta variando.

```python
FIXED = dict(
    population    = 20,
    generations   = 20,
    mutation_rate = 0.2,
    tournament_size = 5,
    epochs        = 150,
    learning_rate = 0.01,
    lambda_       = 0.05,
    K             = 5,
    seed          = 42,
    plot          = False,
)
```

`code`

> **Perché questi valori?**  
> Sono volutamente ridotti (population=20, generations=20, K=5) per mantenere il tempo di esecuzione ragionevole durante i test. Una volta identificati i valori ottimali si può rieseguire con parametri più grandi per la run finale.

---

## Cosa restituisce `run()`

```python
result = run(**FIXED)

result['best_accuracy']        # accuracy sul val set (ottimistica) — percentuale
result['test_accuracy']        # accuracy sul test set (stima reale) — percentuale
result['accuracy_baseline']    # accuracy baseline backprop — percentuale
result['best_fitness']         # fitness miglior individuo — percentuale
result['n_params']             # #parametri architettura migliore
result['best_individuo']       # lista (n_neuroni, nome_funzione) per ogni hidden layer
result['storia_best_fitness']  # lista float, una per generazione
result['storia_best_accuracy'] # lista float, una per generazione
result['storia_mean_accuracy'] # lista float, una per generazione
```

---

## Struttura base di ogni test script

Ogni file di test segue questo schema:

```python
import csv
import matplotlib.pyplot as plt
from main import run
from sklearn.datasets import load_iris   # o altro dataset

FIXED = dict(
    population=20, generations=20, mutation_rate=0.2,
    tournament_size=5, epochs=150, learning_rate=0.01,
    lambda_=0.05, K=5, seed=42, plot=False,
)

VALUES = [...]          # valori del parametro da variare
PARAM  = 'nome_param'  # nome chiave da passare a run()
3333333333
csv_rows  = []
all_results = []

for v in VALUES:
    result = run(**{**FIXED, PARAM: v})

    csv_rows.append({
        PARAM:               v,
        'test_accuracy':     result['test_accuracy'],
        'best_accuracy':     result['best_accuracy'],
        'accuracy_baseline': result['accuracy_baseline'],
        'n_params':          result['n_params'],
        'best_individuo':    str(result['best_individuo']),
    })

    all_results.append({'label': f"{PARAM}={v}", **result})

# ── salva CSV ─────────────────────────────────────────────────────────
with open(f'tests/test_{PARAM}.csv', 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
    writer.writeheader()
    writer.writerows(csv_rows)

# ── grafico comparativo ───────────────────────────────────────────────
gen = range(1, FIXED['generations'] + 1)
fig, axes = plt.subplot_mosaic(
    [['best_acc', 'mean_acc'],
     ['best_fit', 'n_params']],
    figsize=(14, 9)
)

for r in all_results:
    axes['best_acc'].plot(gen, r['storia_best_accuracy'], label=r['label'])
    axes['best_fit'].plot(gen, r['storia_best_fitness'],  label=r['label'])
    axes['mean_acc'].plot(gen, r['storia_mean_accuracy'], label=r['label'])

# baseline come linea orizzontale su best_acc e mean_acc
baseline = all_results[0]['accuracy_baseline']
for key in ['best_acc', 'mean_acc']:
    axes[key].axhline(baseline, color='red', linestyle='dashed', label='baseline')

# n_params come bar chart (un valore per configurazione, non serie temporale)
labels  = [r['label'] for r in all_results]
nparams = [r['n_params'] for r in all_results]
axes['n_params'].bar(labels, nparams, color='steelblue')
axes['n_params'].set_title('#parametri architettura migliore')
axes['n_params'].set_ylabel('#params')
axes['n_params'].tick_params(axis='x', rotation=30)

for key, title, ylabel in [
    ('best_acc', 'Best accuracy per generazione',  'Accuracy (%)'),
    ('best_fit', 'Best fitness per generazione',   'Fitness (%)'),
    ('mean_acc', 'Mean accuracy per generazione',  'Accuracy (%)'),
]:
    axes[key].set_title(title)
    axes[key].set_ylabel(ylabel)
    axes[key].set_xlabel('Generazione')
    axes[key].legend(fontsize=8)
    axes[key].grid(True, alpha=0.3)

plt.suptitle(f'Test {PARAM} — baseline: {baseline:.1f}%', fontsize=13)
plt.tight_layout()
plt.savefig(f'tests/test_{PARAM}.png', dpi=150)
plt.show()
```

---

## Test 1 — `learning_rate`

**Ipotesi:** LR troppo bassa → convergenza lenta, fitness mai satura nelle 20 generazioni.
LR troppo alta → oscillazioni, accuracy instabile tra generazioni.
LR ottimale → crescita rapida e stabile della best_accuracy.

**Valori da testare:**
```python
VALUES = [0.001, 0.005, 0.01, 0.05, 0.1]
PARAM  = 'learning_rate'
```

**Cosa osservare:**
- `storia_best_accuracy`: con LR alta la curva oscilla; con LR bassa cresce lentamente
- `test_accuracy`: confronta con la baseline — la LR ottimale deve batterla stabilmente
- `n_params`: LR alta tende a favorire reti semplici (fitness rumorosa penalizza i grandi)

**Grafici aggiuntivi consigliati:**  
Bar chart di `test_accuracy` per ogni LR con linea baseline — mostra direttamente quale configurazione batte la baseline e di quanto.

```python
fig2, ax = plt.subplots(figsize=(8, 5))
test_accs = [r['test_accuracy'] for r in all_results]
ax.bar(labels, test_accs, color='#344966')
ax.axhline(baseline, color='red', linestyle='dashed', label=f'baseline ({baseline:.1f}%)')
ax.set_title('Test accuracy per learning rate')
ax.set_ylabel('Test accuracy (%)')
ax.legend()
plt.tight_layout()
plt.savefig('tests/test_learning_rate_bar.png', dpi=150)
```

---

## Test 2 — `epochs`

**Ipotesi:** poche epoche → segnale fitness rumoroso → GA seleziona architetture per caso.
Molte epoche → segnale accurato ma ogni generazione è più lenta.
Punto ottimale: il minimo di epoche che produce un segnale fitness affidabile.

**Valori da testare:**
```python
VALUES = [50, 100, 200, 400]
PARAM  = 'epochs'
```

> **Nota:** variare `epochs` cambia anche `epochs_baseline` che di default è uguale a `epochs`.
> Per confronto equo passare anche `epochs_baseline` con lo stesso valore:
> ```python
> result = run(**{**FIXED, 'epochs': v, 'epochs_baseline': v})
> ```

**Cosa osservare:**
- `storia_best_accuracy`: con epochs basse la curva è rumorosa (zigzag); con epochs alte è più liscia
- `n_params`: con fitness rumorosa il GA non riesce a distinguere reti complesse da semplici → n_params erratico
- Tempo di esecuzione (misurabile con `time.perf_counter()` attorno a `run()`)

**Metriche aggiuntive da salvare nel CSV:**
```python
import time
t0 = time.perf_counter()
result = run(**{**FIXED, 'epochs': v, 'epochs_baseline': v})
elapsed = time.perf_counter() - t0

csv_rows.append({
    'epochs':        v,
    'test_accuracy': result['test_accuracy'],
    'n_params':      result['n_params'],
    'elapsed_s':     round(elapsed, 1),
    ...
})
```

---

## Test 3 — `lambda_`

**Ipotesi:** λ=0 → GA ignora la complessità → seleziona reti grandi, overfitting sul val set,
gap val/test elevato. λ alto → penalizza troppo → reti minuscole con accuracy insufficiente.
λ ottimale → bilancia accuracy e semplicità.

**Valori da testare:**
```python
VALUES = [0.0, 0.01, 0.05, 0.1, 0.2]
PARAM  = 'lambda_'
```

**Cosa osservare:**
- `n_params`: deve diminuire all'aumentare di λ — se non succede λ è troppo piccolo
- Gap `best_accuracy - test_accuracy`: con λ=0 il gap è massimo (overfitting sul val); cresce con reti più grandi
- `test_accuracy`: non deve scendere troppo con λ alto — se scende, la penalità è eccessiva

**Grafico aggiuntivo — gap ottimistico:**
```python
gaps = [r['best_accuracy'] - r['test_accuracy'] for r in all_results]
fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.plot(VALUES, [r['test_accuracy'] for r in all_results], marker='o', label='test accuracy')
ax1.plot(VALUES, [r['best_accuracy'] for r in all_results], marker='s', linestyle='--', label='val accuracy')
ax1.axhline(baseline, color='red', linestyle='dashed', label='baseline')
ax1.set_xlabel('lambda_')
ax1.set_ylabel('Accuracy (%)')
ax1.set_title('Accuracy val vs test al variare di lambda_')
ax1.legend()

ax2.plot(VALUES, gaps, marker='o', color='purple')
ax2.set_xlabel('lambda_')
ax2.set_ylabel('Gap val - test (%)')
ax2.set_title('Bias ottimistico al variare di lambda_')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('tests/test_lambda_gap.png', dpi=150)
```

---

## Test 4 — `K`

**Ipotesi:** K=1 → fitness stimata su un singolo addestramento → alta varianza → GA instabile,
seleziona per fortuna. K alto → stima stabile → GA converge su architetture genuinamente buone,
ma il costo computazionale scala linearmente con K.

**Valori da testare:**
```python
VALUES = [1, 3, 5, 10, 20]
PARAM  = 'K'
```

**Cosa osservare:**
- `storia_best_fitness`: con K=1 la curva oscilla molto tra generazioni; con K alto è monotona
- `test_accuracy`: K basso produce architetture selezionate per caso → test_accuracy inferiore
- `n_params`: con K=1 il GA non riesce a penalizzare correttamente la complessità

**Metriche aggiuntive — varianza tra seed:**  
Per K è utile ripetere il test con più seed e misurare la deviazione standard del `test_accuracy`:
```python
SEEDS = [42, 123, 7, 99, 0]
for v in VALUES:
    accs = []
    for s in SEEDS:
        r = run(**{**FIXED, 'K': v, 'seed': s})
        accs.append(r['test_accuracy'])
    csv_rows.append({
        'K':           v,
        'mean_test':   round(sum(accs) / len(accs), 2),
        'std_test':    round((sum((a - sum(accs)/len(accs))**2 for a in accs) / len(accs))**0.5, 2),
        'min_test':    min(accs),
        'max_test':    max(accs),
    })
```

---

## Test 5 — `population`

**Ipotesi:** popolazione piccola → esplorazione limitata → convergenza precoce su ottimi locali.
Popolazione grande → esplora meglio ma ogni generazione è più lenta.
Con `ProcessPoolExecutor` l'aumento di popolazione è parzialmente compensato dalla parallelizzazione.

**Valori da testare:**
```python
VALUES = [5, 10, 20, 30, 50]
PARAM  = 'population'
```

**Cosa osservare:**
- `storia_best_fitness`: con popolazione piccola la curva raggiunge il massimo presto ma su un valore inferiore
- `test_accuracy`: dovrebbe migliorare all'aumentare della popolazione fino a un plateau
- `n_params`: popolazioni grandi esplorano architetture più diverse → n_params più variabile

---

## Test 6 — `generations`

**Ipotesi:** il GA converge entro un numero finito di generazioni; aumentare oltre non porta benefici
e aggiunge solo tempo. Il test permette di trovare il "gomito" della curva di convergenza.

**Valori da testare:**
```python
VALUES = [5, 10, 20, 30, 50]
PARAM  = 'generations'
```

> **Attenzione:** i risultati non sono comparabili direttamente come lista — run con `generations=50`
> restituisce una storia di 50 elementi, quella con `generations=5` solo 5.
> Per confrontarle sui grafici, plottare fino al minimo comune o usare l'asse x normalizzato (%).

**Cosa osservare:**
- In quale generazione si stabilizza `storia_best_fitness` — è il numero minimo di generazioni necessario
- `test_accuracy` con generazioni=5 vs 50: se simile, il GA converge presto e le generazioni extra sono sprecate
- Rapporto tra `best_accuracy` e `test_accuracy`: non deve peggiorare molto aumentando le generazioni
  (se peggiora, il GA sta overfittando sul val set nel tempo)

**Grafico convergenza normalizzata:**
```python
fig, ax = plt.subplots(figsize=(10, 5))
for r in all_results:
    n = len(r['storia_best_fitness'])
    x_norm = [i / n * 100 for i in range(n)]  # asse x in %
    ax.plot(x_norm, r['storia_best_fitness'], label=r['label'])
ax.set_xlabel('Avanzamento generazioni (%)')
ax.set_ylabel('Best fitness (%)')
ax.set_title('Convergenza normalizzata — tutte le configurazioni a confronto')
ax.legend()
plt.tight_layout()
plt.savefig('tests/test_generations_norm.png', dpi=150)
```

---

## Test 7 — `mutation_rate`

**Ipotesi:** mutation_rate basso → poca esplorazione → convergenza precoce su ottimi locali.
mutation_rate alto → il GA distrugge buoni individui → si comporta come ricerca casuale.
Valore ottimale: abbastanza alto da uscire dagli ottimi locali, abbastanza basso da preservare buone architetture.

**Valori da testare:**
```python
VALUES = [0.0, 0.05, 0.1, 0.2, 0.4, 0.8]
PARAM  = 'mutation_rate'
```

> **Nota:** `mutation_rate=0.0` è un caso estremo utile come lower bound — il GA diventa
> puro crossover senza mutazione. Dovrebbe convergere velocemente ma su ottimi locali.

**Cosa osservare:**
- `storia_best_fitness`: con rate=0 cresce subito ma si blocca; con rate alto oscilla sempre
- `n_params`: rate alto aumenta la varianza della struttura → n_params molto variabile tra run
- `test_accuracy`: il punto con test_accuracy massima identifica il mutation_rate ottimale

---

## Test 8 — `dataset`

**Ipotesi:** dataset più complessi richiedono architetture più grandi e/o più epoche.
Il GA deve trovare architetture diverse per dataset diversi, con test_accuracy significativamente
superiore alla baseline su tutti.

**Dataset da testare:**

| Dataset | #feature | #classi | #campioni | Difficoltà |
|---------|----------|---------|-----------|------------|
| iris | 4 | 3 | 150 | bassa |
| wine | 13 | 3 | 178 | media |
| breast_cancer | 30 | 2 | 569 | media |
| digits | 64 | 10 | 1797 | alta |

```python
from sklearn.datasets import load_iris, load_wine, load_breast_cancer, load_digits

DATASETS = {
    'iris':           load_iris(),
    'wine':           load_wine(),
    'breast_cancer':  load_breast_cancer(),
    'digits':         load_digits(),
}

csv_rows   = []
all_results = []

for name, ds in DATASETS.items():
    result = run(**{**FIXED, 'dataset': ds})

    csv_rows.append({
        'dataset':           name,
        'n_feature':         ds.data.shape[1],
        'n_classi':          len(set(ds.target)),
        'n_campioni':        len(ds.target),
        'test_accuracy':     result['test_accuracy'],
        'accuracy_baseline': result['accuracy_baseline'],
        'gap_val_test':      round(result['best_accuracy'] - result['test_accuracy'], 2),
        'n_params':          result['n_params'],
        'n_layer':           len(result['best_individuo']),
        'best_individuo':    str(result['best_individuo']),
    })

    all_results.append({'label': name, **result})
```

**Cosa osservare:**
- `test_accuracy - accuracy_baseline`: il GA deve battere la baseline su tutti i dataset;
  se non succede su digits, probabilmente servono più epoche o più generazioni
- `n_params`: dovrebbe scalare con la complessità del dataset (digits >> iris)
- `n_layer`: il GA tende a trovare reti più profonde per dataset più complessi

**Grafico consigliato — confronto per dataset:**
```python
import numpy as np

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

names     = [r['label']           for r in all_results]
test_accs = [r['test_accuracy']   for r in all_results]
baselines = [r['accuracy_baseline'] for r in all_results]
nparams   = [r['n_params']         for r in all_results]

x = np.arange(len(names))
w = 0.35

axes[0].bar(x - w/2, test_accs, w, label='GA test accuracy', color='#344966')
axes[0].bar(x + w/2, baselines, w, label='Baseline',          color='#C30000B2')
axes[0].set_xticks(x)
axes[0].set_xticklabels(names)
axes[0].set_ylabel('Accuracy (%)')
axes[0].set_title('GA vs Baseline per dataset')
axes[0].legend()
axes[0].grid(True, alpha=0.3, axis='y')

axes[1].bar(names, nparams, color='steelblue')
axes[1].set_ylabel('#parametri')
axes[1].set_title('#parametri architettura migliore per dataset')
axes[1].grid(True, alpha=0.3, axis='y')

plt.suptitle('Test dataset — architettura trovata dal GA', fontsize=13)
plt.tight_layout()
plt.savefig('tests/test_dataset.png', dpi=150)
```

---

## Test 9 — Data leakage

Questo test non varia un parametro: **dimostra due forme di data leakage** presenti (o evitate) nel progetto e misura il loro impatto numerico sull'accuracy riportata.

### Leakage A — GA ottimizza sul validation set (bias ottimistico)

**Problema:** il GA usa il validation set come segnale di fitness per selezionare l'architettura migliore. Questo significa che `best_accuracy` (accuracy sul val) è una stima distorta verso l'alto: l'architettura è stata scelta *perché* funzionava bene su quel val set specifico. `test_accuracy` è la stima reale.

**Test:** eseguire lo stesso setup con `N_SEEDS` seed diversi. Per ogni seed, registrare `best_accuracy` (val, ottimistica) e `test_accuracy` (test, reale). Il gap sistematico tra le due dimostra il bias ottimistico.

```python
import csv
import numpy as np
import matplotlib.pyplot as plt
from main import run

FIXED = dict(
    population=20, generations=20, mutation_rate=0.2,
    tournament_size=5, epochs=150, learning_rate=0.01,
    lambda_=0.05, K=5, plot=False,
)

SEEDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

csv_rows = []
val_accs  = []
test_accs = []
gaps      = []

for s in SEEDS:
    result = run(**{**FIXED, 'seed': s})

    val_acc  = result['best_accuracy']
    test_acc = result['test_accuracy']
    gap      = round(val_acc - test_acc, 2)

    val_accs.append(val_acc)
    test_accs.append(test_acc)
    gaps.append(gap)

    csv_rows.append({
        'seed':              s,
        'val_accuracy':      val_acc,
        'test_accuracy':     test_acc,
        'gap_val_test':      gap,
        'accuracy_baseline': result['accuracy_baseline'],
        'n_params':          result['n_params'],
    })

mean_gap = round(sum(gaps) / len(gaps), 2)
std_gap  = round(np.std(gaps), 2)

# aggiungi riga di riepilogo
csv_rows.append({
    'seed':              'MEDIA',
    'val_accuracy':      round(sum(val_accs)  / len(val_accs),  2),
    'test_accuracy':     round(sum(test_accs) / len(test_accs), 2),
    'gap_val_test':      mean_gap,
    'accuracy_baseline': csv_rows[0]['accuracy_baseline'],
    'n_params':          '',
})

with open('tests/test_data_leakage_A.csv', 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
    writer.writeheader()
    writer.writerows(csv_rows)

# ── grafico ──────────────────────────────────────────────────────────
x = np.arange(len(SEEDS))
w = 0.35

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

ax1.bar(x - w/2, val_accs,  w, label='val accuracy (GA, ottimistica)', color='#8B008B', alpha=0.85)
ax1.bar(x + w/2, test_accs, w, label='test accuracy (stima reale)',     color='#344966', alpha=0.85)
ax1.axhline(csv_rows[0]['accuracy_baseline'], color='red', linestyle='dashed', label='baseline')
ax1.set_xticks(x)
ax1.set_xticklabels([f'seed={s}' for s in SEEDS], rotation=30)
ax1.set_ylabel('Accuracy (%)')
ax1.set_title('Val vs Test accuracy per seed\n(gap = bias ottimistico)')
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.3, axis='y')

ax2.bar(x, gaps, color='#C30000B2')
ax2.axhline(mean_gap, color='black', linestyle='dashed',
            label=f'gap medio = {mean_gap}% ± {std_gap}%')
ax2.set_xticks(x)
ax2.set_xticklabels([f'seed={s}' for s in SEEDS], rotation=30)
ax2.set_ylabel('Gap val - test (%)')
ax2.set_title('Bias ottimistico per seed\n(sempre positivo = leakage sistematico)')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3, axis='y')

plt.suptitle(
    f'Leakage A — bias ottimistico GA: gap medio {mean_gap}% ± {std_gap}%',
    fontsize=12
)
plt.tight_layout()
plt.savefig('tests/test_data_leakage_A.png', dpi=150)
plt.show()
```

**Cosa osservare:**
- Il gap `val - test` deve essere **sistematicamente positivo** su tutti i seed — se fosse casuale attorno a zero, non ci sarebbe leakage
- La dimensione media del gap quantifica quanto i risultati sarebbero stati sovrastimati senza test set
- Aumentare `lambda_` tende a ridurre il gap (reti più semplici generalizzano meglio)

---

### Leakage B — scaler fittato su train+val+test (leakage statistico)

**Problema:** se `StandardScaler` viene fittato sull'intero dataset prima dello split, media e deviazione standard del test set "trapelano" nel preprocessing del training set. La rete impara su dati che contengono informazioni implicite del test set → accuracy gonfiata in modo non riproducibile in produzione.

Il codice attuale in `main.py` è **corretto**: `scaler.fit_transform(x_train)` poi `scaler.transform(x_val)` e `scaler.transform(x_test)`. Questo test dimostra numericamente il danno del pattern sbagliato.

```python
import csv
import numpy as np
import matplotlib.pyplot as plt
from main import one_hot, mean_accuracy_on_K_runs, number_params
from neural_network.nn_layer import leaky_relu
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

SEED  = 42
K     = 5
LR    = 0.01
EPOCHS = 150
CROMOSOMA = [(8, leaky_relu), (8, leaky_relu)]

dataset = load_iris()
x = dataset.data
y = dataset.target
n_classi = len(set(y))

x_trainval, x_test, y_trainval, y_test = train_test_split(
    x, y, test_size=0.2, random_state=SEED
)
x_train, x_val, y_train, y_val = train_test_split(
    x_trainval, y_trainval, test_size=0.25, random_state=SEED
)

# ── Scenario CORRETTO: scaler fittato solo su x_train ────────────────
scaler_ok = StandardScaler()
x_train_ok = scaler_ok.fit_transform(x_train)
x_val_ok   = scaler_ok.transform(x_val)
x_test_ok  = scaler_ok.transform(x_test)

y_train_oh = one_hot(y_train, n_classi)
y_test_oh  = one_hot(y_test,  n_classi)

acc_corretto = mean_accuracy_on_K_runs(
    CROMOSOMA, x.shape[1], n_classi, LR, EPOCHS,
    y_train_oh, x_train_ok, y_test_oh, x_test_ok,
    K=K, seed=SEED
)

# ── Scenario SBAGLIATO: scaler fittato su tutto x (train+val+test) ───
# Questo è il pattern sbagliato — fit su tutto il dataset
scaler_leak = StandardScaler()
x_all_scaled = scaler_leak.fit_transform(x)          # ← leakage qui

x_tv_leak, x_test_leak, y_tv_leak, y_test_leak = train_test_split(
    x_all_scaled, y, test_size=0.2, random_state=SEED
)
x_train_leak, _, y_train_leak, _ = train_test_split(
    x_tv_leak, y_tv_leak, test_size=0.25, random_state=SEED
)

y_train_leak_oh = one_hot(y_train_leak, n_classi)
y_test_leak_oh  = one_hot(y_test_leak,  n_classi)

acc_leakage = mean_accuracy_on_K_runs(
    CROMOSOMA, x.shape[1], n_classi, LR, EPOCHS,
    y_train_leak_oh, x_train_leak, y_test_leak_oh, x_test_leak,
    K=K, seed=SEED
)

# ── risultati ─────────────────────────────────────────────────────────
acc_corretto_pct = round(acc_corretto * 100, 2)
acc_leakage_pct  = round(acc_leakage  * 100, 2)
gap              = round(acc_leakage_pct - acc_corretto_pct, 2)

print(f"Scaler corretto (fit su train): {acc_corretto_pct}%")
print(f"Scaler sbagliato (fit su tutto): {acc_leakage_pct}%")
print(f"Gonfiamento artificiale:         +{gap}%")

csv_rows = [
    {'scenario': 'scaler_corretto',  'test_accuracy': acc_corretto_pct, 'gap': 0},
    {'scenario': 'scaler_leakage',   'test_accuracy': acc_leakage_pct,  'gap': gap},
]
with open('tests/test_data_leakage_B.csv', 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
    writer.writeheader()
    writer.writerows(csv_rows)

# ── grafico ──────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 5))
ax.bar(['Corretto\n(fit su train)', 'Sbagliato\n(fit su tutto)'],
       [acc_corretto_pct, acc_leakage_pct],
       color=['#344966', '#C30000'])
ax.set_ylabel('Test accuracy (%)')
ax.set_title(f'Leakage B — scaler\nGonfiamento artificiale: +{gap}%')
ax.set_ylim(max(0, min(acc_corretto_pct, acc_leakage_pct) - 10), 100)
ax.grid(True, alpha=0.3, axis='y')
for i, v in enumerate([acc_corretto_pct, acc_leakage_pct]):
    ax.text(i, v + 0.5, f'{v}%', ha='center', fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig('tests/test_data_leakage_B.png', dpi=150)
plt.show()
```

**Cosa osservare:**
- `acc_leakage > acc_corretto` → il preprocessing sbagliato gonfia artificialmente i risultati
- Su Iris il gap è piccolo (dataset semplice, pochi campioni, distribuzione simile tra split)
- Su `load_digits()` il gap è più visibile — più feature, più informazione che trapela

---

## Ordine di esecuzione e aggiornamento progressivo di FIXED

Eseguire i test in questo ordine. Dopo ogni test, **aggiornare `FIXED` in tutti i file successivi** con il valore ottimale trovato. Ogni test misura un solo parametro con tutto il resto già ottimizzato.

---

### Step 1 — `epochs` → `test_2_epochs.py`

**Criterio di scelta:** il minimo di epoche per cui `storia_best_fitness` è liscia (non a zigzag).
Scegliere il valore più basso che produce una curva monotona o quasi monotona.

**Aggiornare nei file successivi (3 → 9):**
```python
FIXED = dict(..., epochs=<VALORE_TROVATO>, ...)
```
500


---

### Step 2 — `K` → `test_4_k.py`

**Prima di eseguire:** imposta `epochs=<valore trovato allo step 1>` in `FIXED`.

**Criterio di scelta:** il minimo K per cui `storia_best_fitness` è stabile tra generazioni consecutive (variazione < 5% tra generazioni adiacenti). Bilanciare stabilità vs tempo di esecuzione.

**Aggiornare nei file successivi (1, 3, 5, 6, 7, 8, 9A):**
```python
FIXED = dict(..., epochs=<step1>, K=<VALORE_TROVATO>, ...)
```

3

---

### Step 3 — `learning_rate` → `test_1_learning_rate.py`

**Prima di eseguire:** imposta `epochs=<step1>`, `K=<step2>` in `FIXED`.

**Criterio di scelta:** il LR con `test_accuracy` massima e curva `storia_best_accuracy` stabile (non oscillante). Se più LR danno accuracy simile, preferire quella con curva più liscia.

**Aggiornare nei file successivi (3, 5, 6, 7, 8, 9A):**
```python
FIXED = dict(..., epochs=<step1>, K=<step2>, learning_rate=<VALORE_TROVATO>, ...)
```

0.05

---

### Step 4 — `lambda_` → `test_3_lambda.py`

**Prima di eseguire:** imposta `epochs=<step1>`, `K=<step2>`, `learning_rate=<step3>` in `FIXED`.

**Criterio di scelta:** il λ con il miglior compromesso tra `test_accuracy` alta e `n_params` basso. Verificare che il gap `best_accuracy - test_accuracy` diminuisca rispetto a λ=0 senza perdere più del 2-3% di `test_accuracy`.

**Aggiornare nei file successivi (5, 6, 7, 8, 9A):**
```python
FIXED = dict(..., epochs=<step1>, K=<step2>, learning_rate=<step3>, lambda_=<VALORE_TROVATO>, ...)
```

0.05

---

### Step 5 — `mutation_rate` → `test_7_mutation_rate.py`

**Prima di eseguire:** imposta i valori trovati agli step 1–4 in `FIXED`.

**Criterio di scelta:** il rate più basso che raggiunge una `test_accuracy` vicina al massimo osservato, con `storia_mean_accuracy` stabile (non oscillante) tra le generazioni. Un rate elevato (es. 0.8) può dare test_accuracy alta su una singola run per puro rumore: con 80% di geni mutati ad ogni step il GA perde la capacità di costruire su ciò che ha imparato e si riduce a ricerca casuale. Preferire il rate più conservativo che non peggiora la test_accuracy di più del 1–2% rispetto al picco.

**Aggiornare nei file successivi (5, 6, 8, 9A):**
```python
FIXED = dict(..., mutation_rate=<VALORE_TROVATO>, ...)
```

0.2

---

### Step 6 — `population` → `test_5_population.py`

**Prima di eseguire:** imposta i valori trovati agli step 1–5 in `FIXED`.

**Criterio di scelta:** il valore di population con `test_accuracy` massima e gap `best_accuracy - test_accuracy` minimo. La relazione non è monotona: con generazioni fisse, popolazioni grandi esplorano più candidati ma li raffinano meno, quindi la test_accuracy tende a formare un picco anziché un plateau. Oltre il picco le performance calano e il gap val-test torna a crescere (maggiore varianza, non maggiore qualità). Usare il gap come segnale di generalizzazione: a parità di test_accuracy, preferire il population con gap più basso.

**Aggiornare nei file successivi (6, 8, 9A):**
```python
FIXED = dict(..., population=<VALORE_TROVATO>, ...)
```

20

---

### Step 7 — `generations` → `test_6_generations.py`

**Prima di eseguire:** imposta i valori trovati agli step 1–6 in `FIXED`.

**Criterio di scelta:** la generazione in cui `storia_best_fitness` si stabilizza (lle ultime N generazioni). Usare il grafico convergenza normalizzata per trovare il "gomito".

**Aggiornare nei file successivi (8, 9A):**
```python
FIXED = dict(..., generations=<VALORE_TROVATO>, ...)
```

50

---

### Step 8 — `dataset` → `test_8_dataset.py`

**Prima di eseguire:** imposta **tutti** i valori ottimali trovati agli step 1–7 in `FIXED`. Questo è il test di generalizzabilità con la configurazione ottimale completa.

**Nessun aggiornamento da fare** — questo test è la verifica finale.

---

### Step 9 — Leakage → `test_9A_leakage.py`, `test_9B_leakage.py`

**9A:** usa i valori ottimali completi (step 1–7) in `FIXED`. Misura il bias ottimistico sistematico.

**9B:** non dipende da `FIXED` — confronto diretto corretto vs sbagliato, configurazione fissa.

---

### Riepilogo aggiornamenti FIXED

| Dopo step | Parametro trovato | Aggiornare in |
|-----------|-------------------|---------------|
| 1 — epochs | `epochs=X` | test 4, 1, 3, 5, 6, 7, 8, 9A |
| 2 — K | `K=X` | test 1, 3, 5, 6, 7, 8, 9A |
| 3 — learning_rate | `learning_rate=X` | test 3, 5, 6, 7, 8, 9A |
| 4 — lambda_ | `lambda_=X` | test 5, 6, 7, 8, 9A |
| 5 — mutation_rate | `mutation_rate=X` | test 5, 6, 8, 9A |
| 6 — population | `population=X` | test 6, 8, 9A |
| 7 — generations | `generations=X` | test 8, 9A |

---

## RISULTATI

- epochs: 500
- K: 3
- lr: 0.05
- lambda: 0.05
- mutation_rate: 0.2
- population: 20
- generations: 50

## CSV finale consolidato

Dopo aver completato tutti i test, produrre un CSV unico per confronto rapido:

```python
import csv

summary = []
# Aggiungere una riga per ogni configurazione ottimale trovata
summary.append({
    'test':              'learning_rate',
    'param_value':       0.01,       # il migliore trovato
    'test_accuracy':     ...,
    'accuracy_baseline': ...,
    'n_params':          ...,
    'note':              'convergenza stabile, batte baseline di X%'
})
# ... una riga per ogni test

with open('tests/summary.csv', 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=summary[0].keys())
    writer.writeheader()
    writer.writerows(summary)
```
