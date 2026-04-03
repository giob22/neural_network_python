# Analisi test

## Analisi del Test 1 — Effetto del Mutation Rate

### Risultati numerici

| mutation_rate | best_accuracy | best_fitness | n_layer | n_params |
|---|---|---|---|---|
| 0.05 | 94.44% | 83.27 | 3 | 2235 |
| 0.20 | 92.22% | 85.73 | 3 | 1298 |
| 0.40 | 91.11% | 84.25 | 3 | 1372 |
| baseline | 70.00% | — | 2 | — |

---

### Osservazione 1 — Tutte e tre le run battono la baseline

La prima cosa da notare è che tutte e tre le configurazioni superano ampiamente la baseline del 70%, con miglioramenti che vanno dal +21% al +24%. Questo è il risultato più importante del test — dimostra che il GA, indipendentemente dal mutation rate scelto, riesce a trovare architetture significativamente migliori di una rete progettata manualmente con architettura fissa.

La baseline è stata addestrata con soli 450 epoche su un'architettura fissa `[(8, relu), (8, relu)]`. Il GA ha esplorato architetture diverse e trovato configurazioni con più neuroni e tre layer, che su Iris danno un vantaggio netto.

---

### Osservazione 2 — Mutation rate basso trova la best accuracy ma genera architetture più complesse

Con `mr=0.05` il GA trova la migliore accuracy in assoluto — 94.44% — ma produce l'architettura più complessa con 2235 parametri, quasi il doppio rispetto alle altre due run. Questo è esattamente il comportamento atteso con una mutazione bassa.

Quando la mutazione è rara, il GA si affida quasi esclusivamente al crossover per generare nuovi individui. Il crossover combina layer esistenti dai genitori, tendendo a preservare architetture grandi se i genitori hanno molti neuroni. Senza mutazioni frequenti che riducano o eliminino layer, la pressione verso la semplicità è debole e la popolazione converge verso reti grandi.

La fitness più bassa di mr=0.05 (83.27) rispetto a mr=0.2 (85.73) conferma questo — nonostante la best accuracy più alta, la penalità per la complessità abbassa significativamente la fitness. In altri termini: il GA con mr=0.05 ha trovato una rete accurata ma "costosa".

---

### Osservazione 3 — Mutation rate medio è il miglior compromesso

Con `mr=0.2` ottieni la fitness più alta in assoluto (85.73), un'architettura più snella (1298 parametri) e un'accuracy ancora molto buona (92.22%). Questo è il punto di equilibrio ottimale per questo problema.

Un mutation rate del 20% significa che mediamente 1 gene su 5 viene modificato ad ogni generazione. Questo è abbastanza frequente da mantenere diversità nella popolazione — impedendo la convergenza prematura — ma abbastanza raro da non distruggere le caratteristiche buone ereditate dai genitori tramite crossover.

---

### Osservazione 4 — Mutation rate alto non migliora ulteriormente

Con `mr=0.4` l'accuracy scende leggermente a 91.11% e i parametri rimangono simili a mr=0.2 (1372 vs 1298). La fitness è intermedia (84.25). Con mutazioni molto frequenti il GA fatica ad accumulare miglioramenti — ogni generazione modifica troppo i figli rispetto ai genitori, rendendo difficile preservare le caratteristiche buone trovate nelle generazioni precedenti.

Questo fenomeno si vede chiaramente nel grafico della **mean accuracy** — la curva verde (mr=0.4) è la più bassa e rumorosa per quasi tutta la run, segno che la popolazione nel suo insieme fatica a migliorare quando la mutazione è troppa aggressiva.

---

### Osservazione 5 — Analisi del grafico Best accuracy per generazione

Tutte e tre le curve mostrano un andamento simile: crescita rapida nelle prime 5-10 generazioni, poi oscillazioni intorno a valori alti. Le oscillazioni sono normali — dipendono dalla casualità del crossover e della mutazione, e dal fatto che la fitness viene valutata con pesi inizializzati casualmente.

La curva blu (mr=0.05) mostra oscillazioni più ampie nella seconda metà — questo suggerisce che con poca mutazione la popolazione perde diversità e il GA fatica a migliorare stabilmente dopo la convergenza iniziale.

---

### Osservazione 6 — Analisi del grafico Mean accuracy per generazione

Questo grafico è molto informativo perché mostra la salute dell'intera popolazione, non solo del miglior individuo.

Tutte e tre le curve partono intorno al 40-50% nella prima generazione — la popolazione iniziale è completamente casuale quindi è atteso. La curva blu (mr=0.05) sale più rapidamente nelle prime generazioni perché con poca mutazione l'elitismo domina — il miglior individuo si replica molto e trascina su la media. Le curve arancione e verde salgono più lentamente ma in modo più stabile.

La cosa più importante è che tutte e tre le curve superano la baseline entro generazione 10-15, il che conferma che il GA sta imparando e migliorando la popolazione nel suo insieme, non solo trovando un individuo fortunato.

---

### Conclusione del Test 1

Il mutation rate ottimale per questo problema è **0.2**. Offre il miglior compromesso tra accuratezza, complessità e stabilità della convergenza. Un valore troppo basso (0.05) porta a reti accurati ma complesse, un valore troppo alto (0.4) introduce troppo rumore nel processo evolutivo.

Tutti i valori testati producono comunque risultati nettamente superiori alla baseline, il che valida l'approccio generale del GA per la ricerca automatica dell'architettura.

## Analisi del Test 2 — Effetto di Lambda (penalità di complessità)

### Risultati numerici

| lambda | best_accuracy | best_fitness | n_layer | n_params |
|---|---|---|---|---|
| 0.0 | 98.89% | 98.89 | 7 | 5447 |
| 0.00005 | 88.89% | 87.83 | 1 | 211 |
| 0.0002 | 84.44% | 80.06 | 1 | 219 |
| baseline | 60.00% | — | 2 | — |

---

### Osservazione 1 — Con lambda=0 fitness e accuracy coincidono

Con `lambda=0.0` la penalità è completamente assente, quindi la formula diventa:

```
fitness = accuracy - 0 * complessità = accuracy
```

Questo spiega perché nel CSV `best_accuracy` e `best_fitness` sono entrambi 98.89 — sono lo stesso valore. È anche l'unico caso in cui questo accade, e conferma che il codice funziona correttamente.

---

### Osservazione 2 — Senza penalità il GA trova reti enormi

Con `lambda=0.0` il GA trova un'architettura con **7 hidden layer** e **5447 parametri** — la più complessa possibile entro i limiti imposti da `_genera_individuo()`. Questo è esattamente il comportamento atteso e dimostra perché la penalità è necessaria.

Senza nessun costo associato alla complessità, il GA ha un solo obiettivo — massimizzare l'accuracy — e la strada più facile per farlo è aumentare la capacità della rete aggiungendo neuroni e layer. Il risultato è una rete che ottiene 98.89% su Iris, ma che è inutilmente grande per un problema così semplice. Su un dataset più complesso questa rete rischierebbe overfitting severo.

Dal punto di vista biologico è come se la selezione naturale premiasse solo la forza bruta senza nessun costo energetico — tutti gli organismi diventerebbero enormi.

---

### Osservazione 3 — La penalità riduce drasticamente la complessità

Passando da `lambda=0` a `lambda=0.00005` succede qualcosa di sorprendente — l'architettura passa da 7 layer e 5447 parametri a **1 solo layer e 211 parametri**. Una riduzione di complessità del 96%.

Questo mostra quanto sia sensibile il GA alla penalità. Anche un valore piccolo come 0.00005 è sufficiente a scoraggiare completamente architetture grandi, perché su 5447 parametri la penalità vale `0.00005 * 5447 = 0.27` — quasi 27 punti percentuali sottratti dall'accuracy. Il GA impara rapidamente che non vale la pena aggiungere layer se il costo supera il beneficio in termini di accuracy.

---

### Osservazione 4 — Con lambda alto il GA trova architetture minimaliste ma meno accurate

Con `lambda=0.0002` l'architettura è ancora più penalizzata — il GA trova di nuovo 1 solo layer con 219 parametri, e l'accuracy scende a 84.44%. La penalità su una rete da 219 parametri vale `0.0002 * 219 = 0.044` — già 4.4 punti sottratti. Il GA preferisce quindi architetture piccole anche a costo di rinunciare ad accuracy.

Questo è il segnale che `lambda=0.0002` è troppo aggressivo per questo problema — spinge il GA verso reti troppo semplici che non riescono a sfruttare appieno la capacità di apprendimento disponibile.

---

### Osservazione 5 — Il valore ottimale di lambda è 0.00005

Confrontando i tre risultati il punto di equilibrio ottimale è `lambda=0.00005` — riduce la complessità da 5447 a 211 parametri mantenendo un'accuracy dell'88.89%, comunque nettamente superiore alla baseline del 60%.

Tuttavia c'è un aspetto interessante da notare: con `lambda=0.00005` il GA trova un'architettura con **1 solo layer**, che è più semplice dell'architettura baseline usata come riferimento (2 layer). Questo suggerisce che per Iris un singolo hidden layer con abbastanza neuroni è sufficiente — il dataset è semplice abbastanza da non richiedere profondità.

---

### Osservazione 6 — Analisi del grafico Best fitness per generazione

Questo grafico mostra una differenza fondamentale tra le tre run che non è visibile nel grafico dell'accuracy.

La curva blu (`lambda=0`) sale rapidamente fino a 0.95-1.0 — con fitness uguale all'accuracy il GA ha un segnale di ottimizzazione chiarissimo e converge velocemente. Le curve arancione e verde oscillano molto di più intorno a valori più bassi perché la fitness non riflette solo l'accuracy ma anche la complessità — due individui con accuracy simile possono avere fitness molto diverse se hanno complessità diverse, rendendo il paesaggio di ottimizzazione più accidentato.

---

### Osservazione 7 — Analisi del grafico Mean accuracy per generazione

La curva blu (`lambda=0`) domina nettamente — la mean accuracy supera il 70% già intorno alla generazione 8 e rimane stabilmente al di sopra della baseline per tutta la run. Questo è coerente con il fatto che senza penalità tutti gli individui tendono ad avere architetture grandi e accurate.

Le curve arancione e verde sono molto più rumorose e oscillano intorno alla baseline — segno che con la penalità la popolazione è più eterogenea, con individui piccoli e semplici che hanno accuracy bassa e individui più grandi che hanno accuracy alta. La media della popolazione riflette questa eterogeneità.

---

### Osservazione 8 — La baseline è più bassa in questo test (60% vs 70%)

Noti che la baseline in questo test è 60% invece del 70% del test precedente. Questo non dipende da nessun cambiamento nel codice — dipende dalla **casualità dell'inizializzazione dei pesi** della rete baseline. La baseline viene riaddestrarta da zero ad ogni esecuzione del file di test con pesi casuali diversi, quindi il suo valore può variare tra run diverse.

Questo è un punto importante da menzionare nella documentazione — la baseline non è un valore deterministico ma dipende dall'inizializzazione casuale, il che la rende un riferimento approssimativo piuttosto che una misura esatta. Per avere una baseline più affidabile si potrebbe mediare su K run come facciamo per la fitness del GA.

---

### Conclusione del Test 2

Lambda è il parametro con l'effetto più drammatico sull'architettura trovata. Il valore `lambda=0.00005` è il più equilibrato — riduce drasticamente la complessità rispetto al caso senza penalità mantenendo un'accuracy elevata. Un valore troppo alto come `0.0002` spinge il GA verso reti troppo semplici che sacrificano troppa accuracy.

Il risultato più interessante di questo test è che anche con una rete di 1 solo layer il GA supera nettamente la baseline — 88.89% vs 60% — il che dimostra che l'ottimizzazione automatica dell'architettura porta benefici reali anche quando la penalità di complessità è attiva.

## Analisi del Test 3 — Effetto del numero di Epoche

### Risultati numerici

| epochs | best_accuracy | best_fitness | n_layer | n_params | architettura |
|---|---|---|---|---|---|
| 50 | 75.56% | 74.30 | 1 | 251 | [(31, relu)] |
| 150 | 90.00% | 88.75 | 1 | 251 | [(31, leaky_relu)] |
| 300 | 92.22% | 91.13 | 1 | 219 | [(27, relu)] |
| baseline | 70.00% | — | 2 | — | — |

---

### Osservazione 1 — Le epoche sono il parametro con l'effetto più regolare

Questo è il test con l'andamento più chiaro e prevedibile dei tre — all'aumentare delle epoche aumentano sia accuracy che fitness in modo monotono:

```
ep=50  → 75.56%
ep=150 → 90.00%  (+14.44%)
ep=300 → 92.22%  (+2.22%)
```

Questo è esattamente il comportamento atteso — più epoche di backpropagation permettono alla rete di convergere meglio durante la valutazione della fitness, producendo una stima più affidabile della qualità dell'architettura.

---

### Osservazione 2 — Con 50 epoche la fitness è troppo rumorosa

Con `ep=50` la best accuracy raggiunge solo 75.56%, appena sopra la baseline del 70%. Il motivo è che 50 epoche di backpropagation su un campione casuale per volta non sono sufficienti per addestrare correttamente la rete — i pesi non convergono e l'accuracy misurata riflette più la casualità dell'inizializzazione che la qualità reale dell'architettura.

Il grafico della **best fitness** per generazione conferma questo — la curva blu (ep=50) è la più bassa e la più rumorosa, con oscillazioni ampie per tutta la durata della run. Il GA riceve un segnale di fitness poco affidabile e fatica a distinguere architetture buone da quelle cattive.

Il grafico della **mean accuracy** è ancora più eloquente — la curva blu rimane intorno al 40-50% per quasi tutta la run, segno che la popolazione nel suo insieme non riesce a migliorare. La fitness calcolata con 50 epoche è troppo rumorosa per guidare correttamente la selezione.

---

### Osservazione 3 — Il salto da 50 a 150 epoche è enorme, da 150 a 300 è marginale

Il miglioramento più grande si ottiene passando da 50 a 150 epoche — +14.44% di accuracy. Il miglioramento da 150 a 300 epoche è invece molto più contenuto — solo +2.22%. Questo è un esempio classico di **rendimenti decrescenti** — ogni epoca aggiuntiva contribuisce sempre meno al miglioramento dopo un certo punto.

Questo ha un'implicazione pratica importante per il tradeoff tempo-qualità. Il tempo di esecuzione totale scala linearmente con le epoche:

```
ep=50  → tempo base
ep=150 → 3x il tempo base    (+14.44% accuracy)
ep=300 → 6x il tempo base    (+16.66% accuracy totale vs ep=50)
```

Raddoppiare le epoche da 150 a 300 richiede il doppio del tempo ma porta solo +2.22% di miglioramento. La scelta di `ep=150` è quindi il punto di equilibrio ottimale tra qualità della valutazione e tempo di esecuzione.

---

### Osservazione 4 — Tutte e tre le run trovano architetture con 1 solo layer

È interessante notare che tutte e tre le run trovano architetture con un solo hidden layer, nonostante il numero di epoche diverso. Questo è coerente con i risultati del test 2 dove `lambda=0.00005` spingeva verso architetture semplici. La penalità di complessità è attiva anche in questo test con lo stesso valore, quindi il GA premia architetture snelle indipendentemente dalle epoche.

La differenza tra le architetture trovate è nel numero di neuroni — ep=300 trova 27 neuroni invece di 31, segno che con più epoche il GA riesce a valutare con maggiore precisione che un layer leggermente più piccolo è sufficiente per ottenere alta accuracy su Iris.

---

### Osservazione 5 — Analisi del grafico Best accuracy per generazione

Le tre curve mostrano comportamenti molto diversi nelle prime generazioni. La curva verde (ep=300) parte già da valori alti — intorno a 0.75 alla generazione 1 — e si stabilizza rapidamente sopra 0.90. La curva arancione (ep=150) parte più in basso ma converge verso valori simili entro la generazione 10. La curva blu (ep=50) oscilla intorno alla baseline per tutta la run senza mai convergere stabilmente.

Questo conferma che il numero di epoche determina la **qualità del segnale** che il GA usa per evolvere — con poche epoche il segnale è troppo rumoroso per guidare l'evoluzione efficacemente.

---

### Osservazione 6 — Analisi del grafico Mean accuracy per generazione

Il grafico della mean accuracy è quello più rivelatore. La curva verde (ep=300) è stabilmente sopra il 70% già dalla generazione 2 e raggiunge valori intorno all'80% nella seconda metà della run — significa che l'intera popolazione è di buona qualità, non solo il miglior individuo.

La curva arancione (ep=150) è molto più instabile — oscilla tra 60% e 72% con picchi e cadute frequenti. Questo suggerisce che con 150 epoche la valutazione della fitness ha ancora una componente di rumore significativa che rende la selezione meno precisa.

La curva blu (ep=50) rimane intorno al 45-50% per quasi tutta la run — la popolazione non riesce a migliorare perché il segnale di fitness è troppo debole per distinguere architetture buone da quelle cattive.

---

### Osservazione 7 — Considerazione sul tempo di esecuzione

Questo test ha un'implicazione pratica importante che vale la pena menzionare nella documentazione. Il tempo totale di esecuzione dell'intero GA è:

```
population_size * K * epochs * generations
= 20 * 3 * epochs * 30
```

Con ep=300 stai eseguendo 54.000 training invece di 27.000 — il doppio del tempo per un guadagno di solo 2.22% di accuracy. Per un progetto con risorse computazionali limitate come il tuo, `ep=150` è la scelta più razionale.

---

### Conclusione del Test 3

Il numero di epoche è il parametro che controlla la **qualità del segnale di fitness** ricevuto dal GA. Troppo poche epoche (50) rendono la valutazione inaffidabile e il GA non riesce a evolvere efficacemente. Troppe epoche (300) portano miglioramenti marginali a fronte di un costo computazionale doppio. Il valore ottimale per questo progetto è **150 epoche**, confermando la scelta iniziale della configurazione base.

Questo test dimostra un principio generale importante nel machine learning — la qualità del processo di valutazione è tanto importante quanto la qualità dell'algoritmo di ottimizzazione. Un GA perfetto con una fitness mal calcolata non convergerà mai verso buone soluzioni.


## Analisi del Test 4 — Effetto del Learning Rate

### Risultati numerici

| learning_rate | best_accuracy | best_fitness | n_layer | n_params | architettura |
|---|---|---|---|---|---|
| 0.001 | 60.00% | 59.35 | 1 | 131 | [(16, leaky_relu)] |
| 0.010 | 90.00% | 85.36 | 2 | 928 | [(23, leaky_relu), (30, leaky_relu)] |
| 0.050 | 96.67% | 95.65 | 1 | 203 | [(25, relu)] |
| baseline | 70.00% | — | 2 | — | — |

---

### Osservazione 1 — Il learning rate è il parametro più critico di tutti

Questo test produce il risultato più estremo tra tutti quelli eseguiti. La differenza tra il caso peggiore e quello migliore è enorme:

```
lr=0.001 → 60.00%   ← peggiore della baseline
lr=0.010 → 90.00%   ← +30% rispetto a lr=0.001
lr=0.050 → 96.67%   ← +6.67% rispetto a lr=0.01
```

Nessun altro parametro testato ha prodotto una variazione così ampia. Questo dimostra che il learning rate è il parametro più sensibile dell'intero sistema — una scelta sbagliata può rendere il GA completamente inutile.

---

### Osservazione 2 — Con lr=0.001 il GA non supera la baseline

Con `lr=0.001` il GA trova un'architettura con accuracy del 60% — uguale alla baseline e inferiore a quella ottenuta con tutti gli altri parametri testati. È il risultato peggiore in assoluto tra tutti i test eseguiti.

Il motivo è che con un learning rate così piccolo la backpropagation aggiorna i pesi con passi microscopici. In sole 150 epoche la rete non riesce a convergere — i pesi partono casuali e rimangono quasi casuali alla fine del training. La fitness calcolata riflette quindi quasi esclusivamente la casualità dell'inizializzazione, non la qualità dell'architettura.

Il GA riceve un segnale completamente inutile — tutte le architetture sembrano ugualmente buone o cattive perché nessuna converge — e non riesce a evolvere verso soluzioni migliori. Questo spiega la curva blu completamente piatta nel grafico della mean accuracy, che rimane intorno al 35% per tutta la run senza nessun miglioramento.

---

### Osservazione 3 — Con lr=0.05 il GA trova la migliore architettura in assoluto

Con `lr=0.05` il GA trova la best accuracy più alta di tutti i test — 96.67% — con una rete di un solo layer da 25 neuroni e soli 203 parametri. È il risultato più efficiente dell'intero progetto: alta accuracy, bassa complessità.

Un learning rate più alto permette alla rete di fare passi più grandi durante il training e convergere più velocemente. Con 150 epoche e lr=0.05 la rete riesce ad addestrare efficacemente, producendo una stima affidabile della qualità dell'architettura. Il GA riceve quindi un segnale pulito e riesce a guidare l'evoluzione verso architetture realmente buone.

Il grafico della mean accuracy conferma questo in modo netto — la curva verde (lr=0.05) è stabilmente sopra l'85% già dalla generazione 3 e rimane lì per tutta la run. Significa che non solo il miglior individuo è buono, ma tutta la popolazione è di alta qualità fin dalle prime generazioni.

---

### Osservazione 4 — C'è però un rischio con learning rate troppo alto

Un learning rate di 0.05 funziona bene su Iris perché il problema è semplice e ben condizionato. Su problemi più complessi un learning rate così alto potrebbe causare oscillazioni durante il training — i passi di aggiornamento sono così grandi che i pesi "saltano" oltre il minimo della funzione di errore invece di convergere verso di esso.

Questo fenomeno si chiama **divergenza** e si manifesta come un errore che invece di scendere durante il training oscilla o addirittura cresce. Non lo vedi in questo test perché Iris è troppo semplice per causarlo, ma è un rischio reale che vale la pena menzionare nella documentazione come limitazione.

---

### Osservazione 5 — lr=0.01 produce un'architettura più complessa

Con `lr=0.01` il GA trova l'unica architettura a 2 layer tra tutti i valori testati — `[(23, leaky_relu), (30, leaky_relu)]` con 928 parametri. Questo è un risultato anomalo rispetto agli altri test dove le architetture trovate avevano sempre 1 layer.

Una possibile spiegazione è che con lr=0.01 la convergenza in 150 epoche è sufficiente ma non ottimale — la rete non raggiunge ancora il massimo della sua capacità. Il GA compensa quindi aggiungendo più neuroni e layer per aumentare l'accuracy, anche a costo di una maggiore complessità. Con lr=0.05 invece la rete converge così bene che anche un singolo layer è sufficiente per ottenere accuracy alta.

---

### Osservazione 6 — Analisi del grafico Best fitness per generazione

Questo grafico mostra qualcosa di molto interessante — la curva verde (lr=0.05) è praticamente piatta intorno a 0.91-0.92 per quasi tutta la run, con pochissime oscillazioni. Questo è il segnale di una convergenza rapida e stabile — il GA trova buone architetture già nelle prime generazioni e le mantiene grazie all'elitismo.

La curva arancione (lr=0.01) oscilla molto di più — segno che con questo learning rate la valutazione della fitness ha ancora una componente di rumore significativa. La curva blu (lr=0.001) è la più bassa e rumorosa — fitness intorno a 0.45-0.55 per tutta la run, senza nessuna tendenza al miglioramento.

---

### Osservazione 7 — Parallelismo con il test sulle epoche

C'è un parallelismo interessante tra questo test e quello sulle epoche. Sia un learning rate troppo basso che un numero di epoche troppo basso producono lo stesso effetto — la rete non converge durante la valutazione della fitness e il GA riceve un segnale inutile.

I due parametri sono in realtà complementari — puoi compensare un learning rate basso aumentando le epoche, o viceversa. In entrambi i casi l'obiettivo è lo stesso: fare in modo che la backpropagation converga sufficientemente bene in un tempo ragionevole per dare al GA una stima affidabile della qualità dell'architettura.

```
convergenza = f(learning_rate, epochs)

lr=0.001, ep=150  → non converge
lr=0.01,  ep=150  → converge parzialmente
lr=0.05,  ep=150  → converge bene
lr=0.001, ep=5000 → convergerebbe, ma troppo lento
```

---

### Conclusione del Test 4

Il learning rate è il parametro più critico del sistema. Un valore troppo basso (0.001) rende il GA completamente inutile perché le reti non convergono in 150 epoche e la fitness è priva di significato. Un valore adeguato (0.05) permette una convergenza rapida e produce il miglior risultato in assoluto tra tutti i test — 96.67% di accuracy con una rete semplice da 1 layer.

Per la configurazione finale del progetto si raccomanda `lr=0.05` come valore ottimale, in combinazione con `ep=150` per il numero di epoche. Questa combinazione garantisce convergenza affidabile senza richiedere un tempo di esecuzione eccessivo. Va però sottolineato che questo valore è ottimale per Iris — su problemi più complessi potrebbe essere necessario ridurlo per evitare instabilità durante il training.



