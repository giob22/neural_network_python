Ottima idea, avere una specifica chiara prima di scrivere codice evita di perdere tempo a rifare le cose. Eccola.

---

## Descrizione del progetto finale

### Panoramica generale

Il sistema è composto da due componenti principali che collaborano: la rete neurale (già esistente, con modifiche minori) e l'algoritmo genetico (da implementare). L'obiettivo è usare il GA per trovare automaticamente la migliore architettura della rete neurale sul dataset Iris, senza che tu debba scegliere manualmente il numero di layer, i neuroni per layer e le funzioni di attivazione.

---

### Componente 1 — Rete neurale modificata (`nn_engine.py`)

**Cosa cambia rispetto ad adesso:** il costruttore non riceve più `size_hidden` e `hidden_function` separati, ma una lista di tuple `hidden_config` dove ogni tupla è `(n_neuroni, funzione_attivazione)`. Tutto il resto — feedforward, backpropagation, MSE — rimane invariato.

**Interfaccia del costruttore dopo la modifica:**
```python
neural_network(
    hidden_config=[(16, relu), (8, sigmoid)],
    size_input=4,
    size_output=3,
    learning_rate=0.01,
    output_function=sigmoid
)
```

**Perché questo è sufficiente per il GA:** il GA non usa mai `feedback()`. Gli basta costruire la rete con una certa configurazione, chiamare `feedforward()` su tutti i campioni del dataset, e misurare l'accuratezza. La backpropagation non entra in gioco.

---

### Componente 2 — Algoritmo Genetico (`genetic_algorithm.py`)

Questo è il file nuovo da creare. È composto da quattro parti logiche.

**2a. Rappresentazione del cromosoma**

Un individuo della popolazione è semplicemente una lista Python che descrive l'architettura di una rete. Ogni elemento della lista è una tupla `(n_neuroni, funzione_attivazione)`. Esempio di un individuo:

```python
[(12, relu), (6, sigmoid)]         # 2 hidden layer
[(32, relu), (16, relu), (8, relu)] # 3 hidden layer
[(8, linear)]                       # 1 solo hidden layer
```

Il numero di elementi nella lista corrisponde al numero di hidden layer. Il GA può anche far variare questo numero tra individui diversi — un individuo può avere 1 layer, un altro 3.

**2b. Popolazione iniziale**

Si generano N individui (tipicamente 20–30) in modo casuale. Per ciascuno si sorteggia casualmente: il numero di hidden layer (ad esempio tra 1 e 3), il numero di neuroni per ogni layer (ad esempio tra 4 e 32), e la funzione di attivazione per ogni layer (scelta tra relu, sigmoid, leaky_relu, linear).

**2c. Funzione di fitness**

È il cuore del sistema. Per ogni individuo (cioè ogni cromosoma) si fa:
1. Si costruisce una `neural_network` con quella configurazione
2. Si addestra sul training set con backpropagation per un numero fisso di epoche (ad esempio 100–200 epoche)
3. Si misura l'accuratezza sul validation set
4. Quella accuratezza è la fitness dell'individuo

Un punto importante: la rete viene re-inizializzata con pesi casuali ogni volta che si valuta un individuo. Questo è corretto perché stai valutando l'architettura, non i pesi specifici.

**2d. Operatori genetici**

Sono tre:

*Selezione* — si scelgono i genitori tra gli individui con fitness più alta. Il metodo più semplice è il torneo: si prendono K individui a caso dalla popolazione e si sceglie il migliore tra loro. Si ripete per selezionare il secondo genitore.

*Crossover* — dati due genitori, si genera un figlio combinando le loro caratteristiche. Il modo più naturale con cromosomi di lunghezza variabile è scegliere un punto di taglio e prendere i primi layer dal genitore 1 e i restanti dal genitore 2. Bisogna gestire il caso in cui i due genitori hanno lunghezze diverse.

*Mutazione* — con una certa probabilità (ad esempio 10–20%), si modifica casualmente un gene del figlio: si cambia il numero di neuroni di un layer, oppure si cambia la funzione di attivazione, oppure si aggiunge o rimuove un layer.

---

### Componente 3 — Dataset e preprocessing (`data.py` o inline)

Si usa il dataset Iris di scikit-learn, che ha 150 campioni, 4 feature in input e 3 classi in output. Le operazioni necessarie sono: normalizzazione delle feature (media 0, deviazione standard 1), codifica one-hot delle etichette (necessaria perché la tua rete ha 3 neuroni in output), e split in training set (80%) e validation set (20%).

---

### Componente 4 — Esecuzione e risultati (`main.py`)

Il file principale che orchestra tutto. Esegue il ciclo evolutivo per N generazioni (tipicamente 20–30), e ad ogni generazione salva la fitness del miglior individuo e la fitness media della popolazione. Alla fine stampa la migliore architettura trovata e produce due grafici: l'evoluzione della fitness del miglior individuo generazione per generazione, e il confronto tra quest'ultima e l'accuratezza di una rete addestrata con backpropagation classica sulla stessa architettura di riferimento (ad esempio 2 layer da 8 neuroni con relu).

---

### Schema riassuntivo del flusso

```
main.py
  │
  ├─ genera popolazione iniziale (N cromosomi casuali)
  │
  └─ per ogni generazione:
       ├─ valuta fitness di ogni individuo
       │     └─ costruisce NN → addestra → misura accuratezza
       ├─ salva il migliore
       ├─ selezione + crossover + mutazione → nuova popolazione
       └─ se ultima generazione → stampa risultati + grafici
```

---

Questa descrizione è tutto ciò di cui hai bisogno per implementare il progetto in modo autonomo. Quando sei pronto a iniziare da una parte specifica, dimmi pure da dove vuoi partire e ti aiuto con il codice.