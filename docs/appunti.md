## x_train, y_train, x_val e y_val

Quando hai un dataset come Iris devi dividerlo in due parti prima di iniziare qualsiasi training. Questa divisione è necessaria per misurare quanto la rete **generalizza** su dati che non ha mai visto durante l'allenamento.

X_train e y_train sono i dati di addestramento. X_train contiene le feature (le 4 misure del fiore) e y_train contiene le etichette corrispondenti (la specie, in formato one-hot). La rete neurale vede questi dati durante il training con backpropagation e aggiorna i pesi su di essi.

X_val e y_val sono i dati di validazione, che la rete non vede mai durante il training. Dopo che la rete è stata addestrata su X_train, la valuti su X_val e misuri quante predizioni sono corrette. Questa accuratezza è la fitness dell'individuo.

La distinzione è cruciale perché una rete potrebbe imparare a memoria i dati di training e fare 100% di accuratezza su di essi, ma poi fallire su dati nuovi. Misurare la fitness su X_val ti dà una stima onesta di quanto vale quell'architettura.

One-hot encoding
Il nome "one-hot" viene dal fatto che c'è sempre un solo 1 ("one hot") e tutti gli altri valori sono 0. Ogni posizione del vettore corrisponde a una classe.
Per Iris le tre classi sono tre specie di fiore, quindi il vettore ha 3 posizioni:

```text
[1, 0, 0]  →  Iris Setosa
[0, 1, 0]  →  Iris Versicolor
[0, 0, 1]  →  Iris Virginica
```

Perché si usa invece di un numero intero
Potresti pensare di usare semplicemente 0, 1, 2 come etichette. Il problema è che questo introdurrebbe un ordine artificiale tra le classi — la rete potrebbe imparare che Virginica (2) è "più grande" di Setosa (0), il che non ha nessun senso biologico. Con il one-hot encoding le tre classi sono trattate in modo simmetrico e indipendente, ognuna con il suo neurone di output dedicato.

## perché utilizzo un approssimazione della derivata della softmax

La derivata completa della Softmax sarebbe una matrice (il Jacobiano) perché ogni output dipende da tutti gli input. Nella tua implementazione usi solo la diagonale `S * (1 - S)`, che è la stessa formula della Sigmoid. Questa è un'approssimazione, ma per il tuo caso è sufficiente perché la backpropagation con questa approssimazione converge comunque correttamente su Iris.

L'approssimazione diagonale ignora le interazioni tra neuroni di output diversi — cioè assume che la variazione dell'output i dipenda solo dall'input i e non dagli altri. Su Iris questa semplificazione non causa problemi perché le classi sono ben separabili e la rete converge comunque. È una scelta comune anche in implementazioni professionali quando si vuole mantenere il codice semplice e uniforme.

## perché in mutazione devo lavorare su una copia:

Perché in Python le liste sono passate **per riferimento**, non per valore.

---

## Cosa significa in pratica

Quando scrivi:

```python
def _mutazione(self, individuo):
    del individuo[0]
```

Non stai lavorando su una copia locale di `individuo` — stai modificando direttamente la lista originale che esiste nella popolazione. Questo significa che dopo aver chiamato `_mutazione()`, il cromosoma nella popolazione è già stato alterato, anche se non volevi farlo.

---

## Perché è un problema nel tuo caso specifico

Nel ciclo principale di `run()` farai qualcosa del genere:

```python
genitore1 = self._selezione(popolazione, fitness_scores)
genitore2 = self._selezione(popolazione, fitness_scores)
figlio = self._crossover(genitore1, genitore2)
figlio = self._mutazione(figlio)
```

Se `_mutazione()` modifica direttamente `figlio` senza copiarlo, e `figlio` condivide dei riferimenti con `genitore1` o `genitore2`, potresti alterare involontariamente i genitori nella popolazione. Con le tuple questo rischio è ridotto perché le tuple sono immutabili, ma la lista esterna che le contiene non lo è.

---

## L'analogia intuitiva

Immagina di avere un foglio con scritto un cromosoma. Passare per riferimento significa passare quel foglio fisico alla funzione — se la funzione ci scrive sopra, il foglio originale è modificato. Lavorare su una copia significa fotocopiare il foglio prima di passarlo — la funzione può scrivere sulla fotocopia senza toccare l'originale.

---


## Perché normalizzare i dati in input

Il motivo fondamentale è che la rete neurale apprende modificando i pesi tramite il gradiente. Se le feature hanno scale molto diverse, i gradienti associati alle feature grandi dominano l'aggiornamento dei pesi e quelli associati alle feature piccole vengono quasi ignorati.

Esempio concreto con Iris:
```
feature 1 — lunghezza sepalo: valori tra 4.3 e 7.9
feature 2 — larghezza sepalo: valori tra 2.0 e 4.4
feature 3 — lunghezza petalo: valori tra 1.0 e 6.9
feature 4 — larghezza petalo: valori tra 0.1 e 2.5
```

Le scale sono simili in questo caso, ma immagina un dataset dove una feature vale 0.001 e un'altra vale 10000. Il peso associato alla feature grande verrebbe aggiornato con passi enormi durante la backpropagation, mentre il peso della feature piccola verrebbe aggiornato con passi microscopici. Il risultato è che la rete fatica a convergere, oscilla molto durante il training e in certi casi non converge mai.

Dopo la normalizzazione tutte le feature hanno media 0 e deviazione standard 1, quindi i gradienti hanno tutti la stessa scala e i pesi vengono aggiornati in modo equilibrato. La rete converge più velocemente e in modo più stabile.

---

## Perché non puoi usare le statistiche del validation set

Questa è una delle regole più importanti del machine learning e vale la pena capirla bene.

Immagina di essere uno studente che deve preparare un esame. Il training set è il materiale di studio che hai a disposizione prima dell'esame. Il validation set è il compito d'esame che il professore ha già preparato ma che non ti ha ancora consegnato.

Se potessi vedere le domande dell'esame in anticipo e preparare le risposte su quelle domande specifiche, il tuo voto non misurerebbe la tua preparazione reale — misurerebbe solo quanto bene hai memorizzato quelle domande specifiche. Il professore non saprebbe se sei davvero preparato o se hai semplicemente spiato.

Lo stesso vale per la normalizzazione. Se calcoli media e deviazione standard includendo il validation set, stai usando informazioni sui dati futuri per trasformare i dati presenti. In un contesto reale questo è impossibile — i dati di validazione arrivano dopo, quando il modello è già stato addestrato.

Esempio numerico per rendere il problema concreto:
```
X_train feature 1: [5.1, 4.9, 6.3, 5.8]
X_val   feature 1: [9.0, 8.5]          ← valori molto più grandi

media calcolata solo su X_train = 5.5
media calcolata su train + val  = 6.6   ← diversa!
```

Se usi la media combinata per normalizzare X_train, stai modificando la trasformazione applicata ai dati di training basandoti su informazioni che in teoria non dovresti ancora avere. Il modello viene addestrato su dati trasformati in modo "ottimistico" rispetto ai dati reali che vedrà in produzione, e la sua accuratezza sul validation set risulterà falsata — tipicamente più alta di quello che otterresti nella realtà.

La regola è quindi sempre la stessa: calcola tutto sul training set, applica le stesse trasformazioni al validation set senza ricalcolare niente. Così la valutazione sul validation set è una misura onesta di quanto il modello generalizza su dati che non ha mai visto.

```py
# IMPORT
importa GeneticAlgorithm da genetic_algorithm
importa numpy, matplotlib, sklearn

# 1 — CARICAMENTO E PREPROCESSING DEI DATI
dataset ← carica Iris da sklearn
X ← feature del dataset (150 campioni, 4 feature)
y ← etichette del dataset (150 campioni)

# normalizzazione
X ← (X - media(X)) / std(X)

# one-hot encoding delle etichette
Y ← matrice di zeri (150, 3)
PER ogni campione i:
    Y[i][y[i]] ← 1

# split train/validation 80-20
X_train, X_val, Y_train, Y_val ← split(X, Y, test_size=0.2)

# 2 — BASELINE CON BACKPROPAGATION CLASSICA
# addestra una rete con architettura fissa di riferimento
rete_baseline ← neural_network([(8, relu), (8, relu)], 4, 3, 0.01, softmax)

PER ogni epoca in range(EPOCHS_BASELINE):
    x ← indice casuale nel training set
    rete_baseline.feedback(X_train[x], Y_train[x])

# valuta la baseline sul validation set
correct ← 0
PER ogni i in range(len(X_val)):
    guess ← rete_baseline.feedforward(X_val[i])
    SE argmax(guess) == argmax(Y_val[i]):
        correct += 1
accuracy_baseline ← correct / len(X_val)
stampa accuracy_baseline

# 3 — ESECUZIONE DEL GA
ga ← GeneticAlgorithm(
    population_size = 20,
    generations = 25,
    mutation_rate = 0.15,
    tournament_size = 3,
    epochs = 150,
    learning_rate = 0.01,
    X_train, Y_train, X_val, Y_val
)

best_individuo, best_fitness, storia_best, storia_mean ← ga.run()

stampa "Migliore architettura trovata:"
stampa best_individuo
stampa "Fitness:" best_fitness

# 4 — GRAFICI
# Grafico 1 — evoluzione della fitness nel tempo
figura 1:
    asse x ← generazioni
    asse y ← fitness
    linea 1 ← storia_best   (best fitness per generazione)
    linea 2 ← storia_mean   (mean fitness per generazione)
    linea orizzontale tratteggiata ← accuracy_baseline
    legenda, titolo, etichette assi

# Grafico 2 — architettura migliore trovata
figura 2:
    per ogni layer in best_individuo:
        stampa in forma leggibile:
        "Layer i: N neuroni, funzione X"
    titolo con best_fitness
```


Ecco una spiegazione dettagliata di ogni parametro, con indicazione del valore che hai scelto e del ragionamento dietro.

---

## `population_size = 20`

È il numero di individui nella popolazione ad ogni generazione. Con 20 individui hai abbastanza diversità genetica per esplorare lo spazio delle architetture senza rendere il calcolo troppo lento. Valori troppo bassi (sotto 10) rischiano di convergere prematuramente, valori troppo alti (sopra 50) rallentano molto l'esecuzione perché ogni individuo richiede K training completi.

---

## `generations = 30`

È il numero di cicli evolutivi. Ad ogni generazione la popolazione viene valutata, selezionata, riprodotta e mutata. Con 30 generazioni e 20 individui stai valutando in totale 600 architetture diverse — sufficiente per Iris. Aumentare le generazioni migliora la qualità della soluzione ma aumenta proporzionalmente il tempo di esecuzione.

---

## `mutation_rate = 0.2`

È la probabilità che ogni gene di un individuo venga modificato durante la mutazione. Con 0.2 ogni gene ha il 20% di probabilità di mutare. Valori troppo bassi rendono il GA simile a una ricerca puramente per selezione e crossover, perdendo diversità. Valori troppo alti (sopra 0.5) rendono il GA caotico — ogni generazione è quasi casuale e non si accumula miglioramento.

---

## `tournament_size = 3`

È il numero di individui che partecipano ad ogni torneo di selezione. Con 3 hai un buon equilibrio tra pressione selettiva ed esplorazione — il migliore dei 3 estratti vince, ma individui mediocri hanno ancora qualche possibilità. Con `tournament_size = 2` la pressione è bassa, con `tournament_size = 10` su una popolazione di 20 il GA diventa quasi greedy e converge troppo velocemente.

---

## `epochs = 150`

È il numero di aggiornamenti di backpropagation eseguiti per addestrare ogni rete durante la valutazione della fitness. Con 150 epoche la rete ha abbastanza training per dare una stima affidabile della sua accuratezza senza richiedere troppo tempo. Ricorda che questo valore moltiplica direttamente il tempo totale — con 20 individui, K=3 e 30 generazioni stai eseguendo `20 * 3 * 30 = 1800` training, ognuno da 150 epoche.

---

## `learning_rate = 0.01`

È il passo di aggiornamento dei pesi durante la backpropagation. Con 0.01 il training è stabile — non troppo lento da non convergere in 150 epoche, non troppo veloce da oscillare senza convergere. È il valore più classico e collaudato per reti neurali semplici come la tua.

---

## `lambda_ = 0.00005`

È il coefficiente della penalità di complessità nella funzione di fitness. Un valore molto piccolo perché la complessità in termini di parametri può arrivare a diverse centinaia — con `lambda_ = 0.00005` e una rete da 200 parametri la penalità vale `0.00005 * 200 = 0.01`, che è piccola ma sufficiente a fare differenza tra architetture simili. Aumentare questo valore spinge il GA verso reti più semplici, diminuirlo lo rende quasi indifferente alla complessità.

---

## `K = 3` (attributo interno, non parametro)

È il numero di volte che ogni individuo viene valutato con pesi iniziali diversi. La fitness finale è la media delle K valutazioni. Con K=3 riduci significativamente la varianza dovuta all'inizializzazione casuale dei pesi senza triplicare completamente il tempo di esecuzione. Potresti esporlo come parametro del costruttore se vuoi sperimentare con valori diversi.

## `__init__`

È il costruttore della classe. Si limita a salvare tutti i parametri come attributi dell'istanza e a definire due liste costanti — `hidden_functions` con le funzioni utilizzabili negli hidden layer e `output_functions` con quelle per l'output layer. Definisce anche `self.K = 3` come costante interna. Non esegue nessun calcolo — prepara solo lo stato iniziale dell'oggetto.

---

## `_genera_individuo`

Genera un cromosoma casuale che rappresenta un'architettura di rete neurale. Sceglie casualmente un numero di hidden layer tra 1 e 4, poi per ognuno sceglie casualmente un numero di neuroni tra 4 e 32 e una funzione di attivazione dalla lista `hidden_functions`. Restituisce una lista di tuple `(n_neuroni, funzione)`. Viene chiamata durante l'inizializzazione della popolazione in `run()` e occasionalmente come fallback in casi degeneri.

---

## `_complessita`

Calcola il numero totale di parametri della rete — pesi e bias — dato un cromosoma. Scorre i layer uno alla volta tenendo traccia del numero di neuroni del layer precedente (`prev`), che parte da 4 (gli input di Iris). Per ogni layer aggiunge `n_neuroni * prev` (i pesi) più `n_neuroni` (i bias). Alla fine aggiunge i parametri dell'output layer fisso. Restituisce un intero che rappresenta la complessità dell'architettura — più è grande, maggiore è la penalità applicata alla fitness.

---

## `_fitness`

È il cuore del sistema. Riceve un cromosoma e restituisce una tupla `(fitness, accuracy)`. Ripete K=3 volte la stessa procedura: costruisce una rete neurale con quell'architettura (con pesi inizializzati casualmente), la addestra sul training set per `self.epochs` epoche scegliendo campioni casuali, poi la valuta sul validation set contando le predizioni corrette tramite `argmax`. Calcola l'accuracy media sulle K run, sottrae la penalità di complessità per ottenere la fitness, e restituisce entrambi i valori separatamente.

---

## `_selezione`

Implementa la selezione tramite torneo. Estrae casualmente `tournament_size` indici dalla popolazione, li ordina in base alla loro fitness dal più alto al più basso, e restituisce l'individuo corrispondente all'indice con fitness più alta — il vincitore del torneo. Viene chiamata due volte per ogni figlio da generare, una per ogni genitore.

---

## `_crossover`

Combina due genitori per generare un figlio. Sceglie un punto di taglio casuale indipendente per ciascun genitore — `t1` per il primo e `t2` per il secondo — e costruisce il figlio prendendo i primi `t1` layer dal genitore 1 e i layer da `t2` in poi dal genitore 2. Gestisce il caso degenere in cui il figlio risulta vuoto scegliendo un gene casuale dal pool dei due genitori. Restituisce il nuovo cromosoma senza modificare i genitori.

---

## `_mutazione`

Introduce modifiche casuali in un cromosoma per mantenere diversità nella popolazione. Prima crea una copia superficiale dell'individuo per non modificare l'originale. Poi scorre ogni gene — ogni tupla `(n_neuroni, funzione)` — e con probabilità `mutation_rate` applica una delle tre mutazioni possibili con pesi uguali: cambiare il numero di neuroni mantenendo la funzione, cambiare la funzione mantenendo i neuroni, oppure aggiungere o rimuovere un layer. Nel caso di rimozione controlla che l'individuo abbia più di un layer, e usa `break` dopo il `del` perché gli indici del loop non sono più validi dopo una cancellazione.

---

## `run`

È l'unico metodo pubblico e orchestra l'intero ciclo evolutivo. Inizializza la popolazione con `population_size` individui casuali e le strutture dati per le storie. Ad ogni generazione valuta tutti gli individui chiamando `_fitness()` una volta sola per ciascuno e salvando fitness e accuracy separatamente, aggiorna il miglior individuo assoluto trovato fino a quel momento, salva le statistiche della generazione corrente nelle storie, stampa un report di avanzamento, e genera la nuova popolazione tramite elitismo — il miglior individuo passa diretto — più crossover e mutazione fino a raggiungere `population_size`. Restituisce il miglior individuo trovato, la sua fitness, la sua accuracy e le tre storie complete per i grafici.




L'overfitting è uno dei problemi più importanti del machine learning. Ti lo spiego partendo dall'intuizione.

---

## L'idea di base

Un modello in overfitting ha **imparato a memoria** i dati di training invece di imparare le regole generali sottostanti. Funziona perfettamente sui dati che ha visto durante l'addestramento, ma fallisce su dati nuovi.

---

## L'analogia più intuitiva

Immagina uno studente che deve prepararsi a un esame di matematica. Ha a disposizione 100 esercizi di pratica.

**Studente che generalizza bene** — studia i concetti, capisce come funzionano le equazioni, e riesce a risolvere qualsiasi esercizio nuovo anche se non lo ha mai visto prima.

**Studente in overfitting** — memorizza le soluzioni dei 100 esercizi parola per parola senza capire il ragionamento. All'esame ottiene 10 sugli esercizi identici a quelli studiati, ma fallisce su qualsiasi variazione.

---

## Come si manifesta numericamente

Il segnale classico dell'overfitting è una **discrepanza** tra accuracy sul training set e accuracy sul validation set:

```
accuracy su training set:   99%  ← il modello ha memorizzato i dati
accuracy su validation set: 65%  ← il modello non generalizza
```

Se le due accuracy sono simili il modello sta generalizzando bene. Se quella sul training set è molto più alta, il modello è in overfitting.

---

## Perché succede

Il modello ha troppa **capacità** rispetto alla complessità del problema — troppi parametri rispetto ai dati disponibili. Con abbastanza parametri una rete neurale può memorizzare qualsiasi dataset, esattamente come un polinomio di grado sufficientemente alto passa per qualsiasi insieme di punti.

Esempio visivo con una curva di regressione:

```
dati reali:        •  •    •  •   •

modello semplice:  ────────────────   ← generalizza bene, ignora il rumore
modello in overfit: ∿∿∿∿∿∿∿∿∿∿∿∿∿   ← passa per ogni punto ma è inutile
```

---

## Il collegamento con il tuo progetto

Quando con `lambda=0` il GA trova una rete con 7 layer e 5447 parametri su un dataset di 120 campioni di training, stai creando le condizioni ideali per l'overfitting — hai molti più parametri che esempi. La rete potrebbe aver imparato a memoria i 120 campioni di training invece di generalizzare.

Il fatto che l'accuracy sul validation set sia comunque 98.89% suggerisce che Iris è un problema così semplice che anche una rete enorme non va in overfitting grave — le classi sono molto ben separabili. Ma su un dataset più complesso e rumoroso quella stessa architettura probabilmente fallirebbe.

Questo è esattamente il motivo per cui hai introdotto la penalità di complessità `lambda` — non solo per trovare reti più efficienti, ma anche per ridurre il rischio di overfitting favorendo architetture più semplici.

---

## Come si combatte

Le tecniche principali sono tre, anche se nel tuo progetto ne usi implicitamente solo una:

**Regolarizzazione** — penalizzare la complessità del modello, esattamente come fa il tuo `lambda_`. È quello che hai implementato.

**Dropout** — durante il training si "spengono" casualmente alcuni neuroni, costringendo la rete a non affidarsi troppo a nessun neurone specifico. Non implementato nel tuo progetto.

**Early stopping** — si interrompe il training quando l'accuracy sul validation set smette di migliorare, anche se quella sul training set continua a salire. Non implementato ma sarebbe un'aggiunta interessante da menzionare come lavoro futuro nella documentazione.