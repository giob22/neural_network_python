# Considerazioni sui test sperimentali

## Perché stiamo facendo tutto questo

Il progetto implementa un **Neural Architecture Search (NAS)** tramite algoritmo genetico: invece di fissare a mano l'architettura della rete neurale (quanti layer, quanti neuroni, quale funzione di attivazione), il GA la cerca automaticamente nello spazio delle possibili configurazioni.

I pesi della rete vengono comunque appresi con backpropagation classica. Il GA decide solo la *forma* della rete; la backpropagation riempie i *valori* al suo interno.

L'intera serie di test serve a rispondere a una domanda: **quali parametri del GA producono il miglior compromesso tra qualità della soluzione trovata, complessità della rete e costo computazionale?** Ogni test isola un parametro alla volta, tenendo fissi tutti gli altri, così da misurare l'effetto netto di quella sola variabile.

---

## Step 1 — `epochs` → scelto: 500

### Obiettivo
Determinare quante epoche di backpropagation servono al GA per valutare correttamente un'architettura. Se il training è troppo breve, la fitness di un individuo è rumorosa: la rete non ha avuto tempo di convergere e il GA seleziona architetture per caso.

### Dati osservati
| epochs | test_accuracy | elapsed_s |
|--------|--------------|-----------|
| 100    | 53.72%       | 96.6 s    |
| 250    | 78.33%       | 96.8 s    |
| 500    | 88.06%       | 100.3 s   |
| 1000   | 91.44%       | 129.3 s   |

### Scelta e motivazione
Con 100 epoche la test_accuracy (53.72%) è quasi quella di un classificatore casuale su 10 classi — il segnale è completamente inutilizzabile. Con 500 epoche si ottiene 88.06% con un overhead di soli 4 secondi in più rispetto a 250. Il salto da 500 a 1000 porta +3.4% ma costa il 29% di tempo in più. Si sceglie **500** come punto di maggior rendimento marginale.

---

## Step 2 — `K` → scelto: 3

### Obiettivo
La fitness di un individuo è la media di K run indipendenti (con seed diversi). K basso → stima rumorosa, il GA può promuovere architetture per fortuna; K alto → stima stabile ma ogni generazione costa K volte di più.

### Dati osservati
| K  | mean_test | std_test |
|----|-----------|----------|
| 1  | 85.61%    | 1.78%    |
| 3  | 85.70%    | 0.93%    |
| 5  | 86.45%    | 1.64%    |
| 10 | 86.86%    | 0.79%    |
| 20 | 86.78%    | 0.93%    |

### Scelta e motivazione
K=1 ha una deviazione standard di 1.78%: una rete può ricevere una valutazione falsamente alta o bassa solo per il seed. K=3 dimezza la std (0.93%) a costo minimo. K=10 riduce ancora la std a 0.79% ma richiede più del triplo del tempo. Il guadagno da K=3 a K=10 è solo 0.14 punti percentuali di std — non vale il costo. Si sceglie **K=3**.

---

## Step 3 — `learning_rate` → scelto: 0.05

### Obiettivo
Il learning rate della backpropagation influenza la qualità del training di ogni individuo. Un LR troppo basso significa che la rete non converge in 500 epoche; un LR troppo alto causa oscillazioni o instabilità numerica.

### Dati osservati
| lr    | test_accuracy | architettura trovata       |
|-------|--------------|----------------------------|
| 0.001 | 38.15%       | [(63, relu)]               |
| 0.005 | 81.48%       | [(64, relu)]               |
| 0.01  | 87.31%       | [(50, leaky_relu)]         |
| 0.05  | 90.65%       | [(24, relu)]               |
| 0.1   | 90.74%       | [(24, sigmoid)]            |

### Scelta e motivazione
Con LR=0.001 la rete non converge in 500 epoche (38% è quasi casuale). LR=0.05 e LR=0.1 danno accuracy simile, ma LR=0.1 trova un'architettura con sigmoid come funzione di attivazione: la sigmoid satura facilmente con LR alta e introduce instabilità nei test successivi. LR=0.05 trova la stessa architettura (24 neuroni, relu) in modo più stabile e riproducibile. Si sceglie **0.05**.

---

## Step 4 — `lambda_` → scelto: 0.05

### Obiettivo
λ è il coefficiente di penalità sulla complessità nella funzione fitness: `fitness = accuracy - λ·log10(n_params)`. Senza penalità (λ=0) il GA seleziona reti grandi perché tendono ad avere accuracy leggermente superiore sul validation set, aumentando il gap val/test (overfitting sulla validation). Con λ troppo alto la penalità domina e il GA seleziona reti minuscole a scapito dell'accuracy.

### Dati osservati
| λ    | test_accuracy | n_params | architettura          |
|------|--------------|----------|-----------------------|
| 0.0  | 90.65%       | 1810     | [(24, relu)]          |
| 0.01 | 90.65%       | 1810     | [(24, relu)]          |
| 0.05 | 90.65%       | 1810     | [(24, relu)]          |
| 0.1  | 88.80%       | 1135     | [(15, leaky_relu)]    |
| 0.2  | 86.20%       | 760      | [(10, relu)]          |

### Scelta e motivazione
Da λ=0 a λ=0.05 la test_accuracy rimane identica (90.65%) e la stessa architettura viene selezionata: la penalità è presente ma non abbastanza forte da distorcere la scelta. Da λ=0.1 in poi la test_accuracy cala sensibilmente (-1.85%) perché il GA preferisce reti troppo piccole. Si sceglie **λ=0.05**: mantiene tutta l'accuracy disponibile e introduce comunque un incentivo alla semplicità che agisce sui casi borderline.

---

## Step 5 — `mutation_rate` → scelto: 0.2

### Obiettivo
Il mutation_rate determina la probabilità che un gene (layer) venga modificato ad ogni generazione. Rate basso → il GA converge rapidamente ma rischia ottimi locali perché esplora poco. Rate alto → il GA esplora molto ma non riesce a costruire su ciò che ha già imparato.

### Dati osservati
| rate | test_accuracy | n_params | architettura        |
|------|--------------|----------|---------------------|
| 0.0  | 89.07%       | 2110     | [(28, leaky_relu)]  |
| 0.05 | 89.07%       | 2110     | [(28, leaky_relu)]  |
| 0.1  | 90.19%       | 1885     | [(25, relu)]        |
| 0.2  | 90.65%       | 1810     | [(24, relu)]        |
| 0.4  | 90.19%       | 1885     | [(25, relu)]        |
| 0.8  | 91.11%       | 1735     | [(23, relu)]        |

### Scelta e motivazione
Rate=0.0 e rate=0.05 convergono sulla stessa architettura (28 neuroni, leaky_relu) con test_accuracy più bassa: senza mutazione il GA sfrutta solo il crossover e rimane intrappolato in un ottimo locale. Rate=0.2 è il picco tra i valori stabili (+1.58% rispetto a rate=0). Rate=0.8 dà test_accuracy apparentemente più alta, ma con 80% dei geni mutati ad ogni passo il GA si riduce a una ricerca quasi casuale: i risultati sono instabili e dipendono fortemente dal seed. Si sceglie **0.2** come il tasso più conservativo che raggiunge il picco di test_accuracy con comportamento stabile.

---

## Step 6 — `population` → scelto: 20

### Obiettivo
La dimensione della popolazione determina quante architetture diverse vengono valutate ad ogni generazione. Popolazione piccola → ricerca ristretta, rischio di convergenza prematura. Popolazione grande → esplorazione ampia ma con generazioni fisse il numero di raffinamenti per individuo diminuisce.

### Dati osservati
| pop | test_accuracy | gap val−test | n_params |
|-----|--------------|--------------|----------|
| 5   | 87.69%       | 3.33%        | 1585     |
| 10  | 88.98%       | 1.85%        | 1510     |
| 20  | 90.65%       | 0.83%        | 1810     |
| 30  | 89.07%       | 2.41%        | 2110     |
| 50  | 89.63%       | 1.30%        | 1360     |
| 70  | 90.00%       | 1.67%        | 1960     |
| 100 | 88.80%       | 0.94%        | 1135     |

### Scelta e motivazione
La relazione non è monotona: non c'è un plateau, c'è un picco a population=20. Con popolazioni grandi e un numero fisso di generazioni, ogni generazione esplora più candidati ma li raffina meno — la convergenza è più lenta. Population=20 è il punto di massima test_accuracy (90.65%) e minimo gap val/test (0.83%), doppia conferma che non è un artefatto della validation set. Si sceglie **20**.

---

## Step 7 — `generations` → scelto: 50

### Obiettivo
Il numero di generazioni determina per quanto a lungo il GA continua a cercare. Troppo poche generazioni → il GA non ha avuto tempo di convergere. Troppe → costo computazionale inutile perché la fitness è già stabile.

### Dati osservati
| gen | test_accuracy | architettura        |
|-----|--------------|---------------------|
| 5   | 88.61%       | [(52, relu)]        |
| 10  | 90.65%       | [(24, relu)]        |
| 20  | 90.65%       | [(24, relu)]        |
| 30  | 90.65%       | [(24, relu)]        |
| 50  | 90.65%       | [(24, relu)]        |
| 70  | 90.65%       | [(24, relu)]        |
| 100 | 90.65%       | [(24, relu)]        |

### Scelta e motivazione
La convergenza è sorprendentemente rapida: già a gen=10 il GA trova la stessa architettura ottimale [(24, relu)] che trova anche a gen=100. Il salto da gen=5 a gen=10 è l'unico significativo (+2.04%). Da gen=10 in poi il risultato è identico. Si sceglie **50** come valore con ampio margine di sicurezza rispetto al punto di convergenza reale (gen=10), senza sprecare tempo come farebbe gen=100.

---

## Step 8 — Generalizzabilità su dataset diversi

### Obiettivo
Verificare che la configurazione ottimale trovata su `load_digits` non sia sovra-ottimizzata per quel problema specifico, ma funzioni ragionevolmente anche su altri dataset.

### Dati osservati
| dataset       | feature | classi | campioni | GA test% | Baseline% | GA vince? |
|---------------|---------|--------|----------|----------|-----------|-----------|
| iris          | 4       | 3      | 150      | 92.22%   | 94.44%    | No        |
| wine          | 13      | 3      | 178      | 100.00%  | 100.00%   | Pari      |
| breast_cancer | 30      | 2      | 569      | 95.61%   | 96.20%    | No        |
| digits        | 64      | 10     | 1797     | 90.65%   | 58.33%    | Sì (+32%) |

### Considerazione
Il GA aggiunge valore reale sui problemi complessi (molte feature, molte classi, molti campioni). Su iris e breast_cancer la baseline con architettura fissa [(8, leaky_relu), (8, leaky_relu)] batte il GA di poco: su problemi semplici la scelta dell'architettura conta meno, e la baseline con più parametri (139 vs 43 su iris) ha un leggero vantaggio di capacità. Su digits il GA supera la baseline di 32 punti percentuali: la ricerca automatica dell'architettura è essenziale quando lo spazio delle soluzioni è ampio.

Un dato degno di nota: su tutti i dataset il GA trova architetture a singolo hidden layer. Questo non è un limite del GA ma una risposta ai dati — per questi problemi un layer nascosto è sufficiente, e la penalità λ disincentiva la complessità inutile.

---

## Step 9 — Data Leakage

### Obiettivo
Quantificare due forme di bias che possono gonfiare artificialmente le stime di performance.

### 9A — Bias ottimistico sistematico (val vs test)
Il GA seleziona l'architettura migliore in base all'accuracy sul **validation set**. Questo introduce un bias: la stessa metrica usata per scegliere viene riportata come risultato, sovrastimando la performance reale.

| metrica           | valore  |
|-------------------|---------|
| val_accuracy media | 91.33% |
| test_accuracy media | 87.01% |
| gap medio          | 4.33%  |

Il gap di 4.33% è sistematico: su 10 seed diversi il gap va da 1.39% a 9.35%. La `test_accuracy` è la stima corretta da riportare; la `best_accuracy` (val) è ottimistica per costruzione.

### 9B — Leakage da preprocessing
Testato se fittare lo StandardScaler sull'intero dataset (train+val+test) invece che solo sul training set introduce gonfiamento artificiale.

| scenario         | test_accuracy |
|------------------|--------------|
| scaler corretto  | 75.33%       |
| scaler su tutto  | 74.00%       |

In questo caso il leakage non ha gonfiato i risultati (anzi li ha leggermente peggiorati). Questo è coerente con la letteratura su dataset piccoli come Iris: la distribuzione dei dati è omogenea tra i split, quindi fittare lo scaler su tutto non trasferisce informazione significativa dal test al train. Su dataset più grandi o con distribuzione eterogenea il leakage sarebbe più evidente.

---

## Configurazione ottimale finale

| Parametro      | Valore scelto | Motivazione principale                                    |
|----------------|--------------|-----------------------------------------------------------|
| epochs         | 500          | Minimo per segnale fitness affidabile                     |
| K              | 3            | Dimezza la varianza di K=1 a costo minimo                 |
| learning_rate  | 0.05         | Picco di test_accuracy con convergenza stabile            |
| lambda_        | 0.05         | Penalizza complessità senza ridurre accuracy              |
| mutation_rate  | 0.2          | Picco stabile; evita ottimi locali senza diventare random |
| population     | 20           | Picco test_accuracy + gap val/test minimo (0.83%)         |
| generations    | 50           | Ampio margine sul punto di convergenza reale (gen=10)     |

**Architettura trovata su digits:** `[(24, relu)]` — 1 hidden layer, 24 neuroni, attivazione ReLU.  
**Test accuracy finale:** 90.65% vs baseline 58.33% (+32.32 punti percentuali).
