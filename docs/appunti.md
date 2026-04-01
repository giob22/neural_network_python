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

Passiamo a `run()`?