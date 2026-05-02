import csv
import sys
import os
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from main import run
from sklearn.datasets import load_digits

os.makedirs('tests_img', exist_ok=True)

FIXED = dict(
    generations=20, mutation_rate=0.2,
    tournament_size=5, epochs=500, learning_rate=0.05,
    lambda_=0.05, K=3, seed=42, plot=False,
    dataset=load_digits(),
)

VALUES = [5, 10, 20, 30, 50, 70, 100]
PARAM  = 'population'

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

    # Grafico test_accuracy vs population (necessario per individuare il punto di plateau)
    fig2, ax = plt.subplots(figsize=(8, 5))
    ax.plot(VALUES, [r['test_accuracy'] for r in all_results], marker='o', label='test accuracy')
    ax.plot(VALUES, [r['best_accuracy'] for r in all_results], marker='s', linestyle='--', label='val accuracy')
    ax.axhline(baseline, color='red', linestyle='dashed', label='baseline')
    ax.set_xlabel('population')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Test accuracy vs population size')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'tests_img/test_{PARAM}_accuracy.png', dpi=150)
    plt.close()

    print("Test 5 completato.")
