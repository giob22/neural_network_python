import numpy as np
import math as m
import matplotlib.pyplot as plt
from nn import neural_network

# creo il dataset della sinusoide

X_train = np.linspace(0,1,100).tolist();
# faccio il tolist() altrimenti sarebbe un numpy.ndarray
Y_train = []

for x in X_train:
    # moltiplico x per 2*pi per avere un periodo completo del sin
    valore_seno = (m.sin(x*2*m.pi) + 1)/2


    # caso: treno di gradini
    # if x<0.2:
    #     valore_seno = 0
    # elif x < 0.4:
    #     valore_seno = 0.2
    # elif x < 0.6:
    #     valore_seno = 0.4
    # elif x < 0.8:
    #     valore_seno = 0.6
    # elif x < 1:
    #     valore_seno = 0.8
    # else:
    #     valore_seno = 1

    #sommo 1 e divido per 2 per avere tutti valori compresi tra 0 e 1
    Y_train.append(valore_seno)

# preparo la rete e il grafico

'''
input (x) = 1
output (y) = 1
hidden_size = 32

1 layer intermedio
1 layer di input
1 layer di output
'''
p = neural_network(n_layer=3, n_input=1, n_output=1, lr=0.1, hidden_size=1000)

plt.ion() 
# per permettere al codice sottostante di continuare ad eseguire dopo aver fatto il plot
# necessario per l'animazione

fig, ax = plt.subplots(figsize=(10,6))
# fig: rappresenta l'intera finestra o contenitore globale
# ax: è il sigolo grafico all'interno della figure

linea_predizione, = ax.plot([],[], color='red', linewidth=3, label='Predizione della rete')

ax.plot(X_train, Y_train, color='blue',alpha=0.5, label='Sinusoide Reale (Target)')

ax.set_xlim(0,1)
ax.set_ylim(-0.1, 1.1)
ax.grid(True, linestyle='-',alpha=0.6)
ax.legend()

# animazione dell'addestramento

epoche_per_frame = 200
frame_totali = 10000

# creiamo i punti per una predizione fluida
x_plot = np.linspace(0,1,1000)

for frame in range(frame_totali):

    for _ in range(epoche_per_frame):
        # scegliamo un punto a caso delle x e addestriamo la rete
        x_casuale = np.random.randint(0,len(X_train))
        p.train(X_train[x_casuale], Y_train[x_casuale])
    
    # calcolo la curva predetta dalla rete
    y_plot = []
    for val_x in x_plot:
        output, _ = p.prediction(np.array([val_x]).reshape(-1,1))
        y_plot.append(output[0][0])
    
    # aggiornamento della linea sul grafico
    linea_predizione.set_data(x_plot, y_plot)
    ax.set_title(f"Simulazione di una rete neurale che impara un periodo di una sinusoide\nEpoca {frame * epoche_per_frame}")

    fig.canvas.draw() # ridisegna la tela
    fig.canvas.flush_events() # forza l'interfaccia a elaborare tutti gli eventi in sospeso
    plt.pause(0.01) # mette in pausa l'esecuzione dello script python → stiamo dando tempo al motore grafico di renderizzare e mostrare il risultato

plt.ioff()
print("Addestramento completato")
plt.show()