# Neural Network Python

Questo progetto è un'implementazione didattica di una rete neurale (Multi-Layer Perceptron) scritta completamente da zero. L'obiettivo è mostrare in modo pratico come funziona il processo di apprendimento "sotto il cofano", usando logica matriciale nuda e cruda con `numpy` invece di appoggiarsi ai soliti framework pronti (come TensorFlow o PyTorch).d

## Struttura del codice

Il progetto è diviso in due file principali, uno che contiene la logica matematica e uno che gestisce la simulazione visiva:

### 1. `nn.py` - Il motore della rete
Qui dentro c'è la classe `neural_network`, che fa tutto il lavoro sporco. Le cose più importanti da tenere d'occhio per capire il codice:
- **Inizializzazione (`__init__`)**: Si passano i parametri chiave (numero di layer, dimensione dei layer nascosti, input/output e learning rate). Le matrici dei pesi e dei bias vengono inizializzate con valori casuali compresi tra -0.5 e 0.5.
- **Funzione di attivazione**: Viene usata la classica **sigmoide**. Un dettaglio furbo nel codice è l'uso di `np.clip(x, -500, 500)` per tagliare i valori estremi: serve a evitare che `np.exp` vada in overflow e faccia crashare tutto con numeri troppo grandi. C'è ovviamente anche la sua derivata (`d_sigmoid`) per calcolare i gradienti.
- **Feedforward (`prediction`)**: Prende l'input (assicurandosi che abbia la forma giusta per le moltiplicazioni matriciali), lo fa passare per tutti i pesi e aggiunge i bias layer dopo layer. Oltre al risultato finale, si salva anche i risultati intermedi (`hidden_outputs`), che sono indispensabili poi per aggiustare il tiro.
- **Addestramento (`train` e `backpropagation_hidden`)**: Qui avviene l'apprendimento. Calcola l'errore rispetto al target, vede quanto questo errore dipenda dall'output usando la derivata, e poi va a ritroso (backpropagation) aggiornando pesi e bias proporzionalmente al *learning rate*.

### 2. `sin.py` - L'addestramento visivo
È il main script da cui far partire tutto. Il suo compito è creare una rete temporanea e addestrarla per fargli imparare la forma di una bella sinusoide (traslata tra 0 e 1).

La parte più interessante di questo file è l'animazione grafica in tempo reale:
- Prepara i dati di base per l'onda.
- Crea l'oggetto rete (di base bello grosso: 3 layer, input a 1, output a 1 e ben 1000 nodi nel layer nascosto).
- Usa `plt.ion()` di matplotlib per attivare la modalità interattiva.
- Nel ciclo principale, si pescano punti a caso per fare allenare la rete (un classico approccio stocastico). Poi, ogni tot epoche (per non bloccare il rendering), lo script calcola la curva di predizione della rete e aggiorna direttamente la linea sul grafico (`linea_predizione.set_data(...)`), resettando poi la tela. Il risultato è che si vede letteralmente la curva rossa adattarsi ai dati blu.

*(Nota: Se spulci bene nel file, noterai un pezzo commentato per far fittare alla rete un "treno di gradini" anziché un'onda. Ottimo per testare quanto la rete riesca a curvare).

## Requisiti ed esecuzione

Moduli necessari:
- `numpy`
- `matplotlib`

Per avviarlo:
```bash
python sin.py
```

## Consigli per smanettarci su
- Prova a ridurre drasticamente il numero di nodi nascosti in `sin.py` (da 1000 a, per dire, 10 o 20) e guarda come l'animazione fatica di più ad approssimare la curva in modo pulito.
- Modifica il `learning_rate` (`lr`). Se è troppo grande, la curva rossa impazzirà saltando da tutte le parti; se è troppo piccolo, ci metterà una vita a convergere.
- Prova a decommentare il treno di gradini e notare come una rete con attivazione sigmoide (che arrotonda molto le curve) gestisca le discontinuità e gli spigoli.
