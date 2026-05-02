import csv
import sys
import os
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from main import run
from sklearn.datasets import load_digits

os.makedirs('tests_img', exist_ok=True)

FIXED = dict(
    population=20, mutation_rate=0.2,
    tournament_size=5, epochs=500, learning_rate=0.05,
    lambda_=0.05, K=3, seed=42, plot=False,
    dataset=load_digits(), max_workers=2
    
)

VALUES = [5, 10, 20, 30, 50, 70, 100]
PARAM  = 'generations'

if __name__ == '__main__':
    csv_rows  = []
    all_results = []
    
    for v in VALUES:
        print(f"Esecuzione con {PARAM}={v}...")
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
    with open(f'tests_img/test_{PARAM}.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
        writer.writeheader()
        writer.writerows(csv_rows)
    
    # ── grafico comparativo ───────────────────────────────────────────────
    fig, axes = plt.subplot_mosaic(
        [['best_acc', 'mean_acc'],
         ['best_fit', 'n_params']],
        figsize=(14, 9)
    )
    
    for r in all_results:
        gen = range(1, len(r['storia_best_accuracy']) + 1)
        axes['best_acc'].plot(gen, r['storia_best_accuracy'], label=r['label'])
        axes['best_fit'].plot(gen, r['storia_best_fitness'],  label=r['label'])
        axes['mean_acc'].plot(gen, r['storia_mean_accuracy'], label=r['label'])
    
    baseline = all_results[0]['accuracy_baseline']
    for key in ['best_acc', 'mean_acc']:
        axes[key].axhline(baseline, color='red', linestyle='dashed', label='baseline')
    
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
    plt.savefig(f'tests_img/test_{PARAM}.png', dpi=150)
    plt.close()
    
    # Grafico normalizzato: stesso asse x % per tutte le run
    fig2, ax = plt.subplots(figsize=(10, 5))
    for r in all_results:
        n = len(r['storia_best_fitness'])
        x_norm = [i / max(1, n - 1) * 100 for i in range(n)]  # asse x in %
        ax.plot(x_norm, r['storia_best_fitness'], label=r['label'])
    ax.set_xlabel('Avanzamento generazioni (%)')
    ax.set_ylabel('Best fitness (%)')
    ax.set_title('Convergenza normalizzata — tutte le configurazioni a confronto')
    ax.legend()
    plt.tight_layout()
    plt.savefig('tests_img/test_generations_norm.png', dpi=150)
    plt.close()

    # Griglia di subplot: ogni run ha il proprio asse x — mostra chiaramente dove si stabilizza
    ncols = 4
    nrows = -(-len(all_results) // ncols)  # divisione con arrotondamento verso l'alto
    fig3, axes3 = plt.subplots(nrows, ncols, figsize=(16, 4 * nrows))
    axes3 = axes3.flatten()
    for i, r in enumerate(all_results):
        gen = range(1, len(r['storia_best_fitness']) + 1)
        axes3[i].plot(gen, r['storia_best_fitness'], color='steelblue')
        axes3[i].set_title(r['label'])
        axes3[i].set_xlabel('Generazione')
        axes3[i].set_ylabel('Best fitness (%)')
        axes3[i].grid(True, alpha=0.3)
    for j in range(i + 1, len(axes3)):
        axes3[j].set_visible(False)
    plt.suptitle('Best fitness per run — individua il gomito di stabilizzazione', fontsize=13)
    plt.tight_layout()
    plt.savefig('tests_img/test_generations_grid.png', dpi=150)
    plt.close()

    print("Test 6 completato.")
