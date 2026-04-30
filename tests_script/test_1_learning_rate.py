import csv
import sys
import os
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from main import run
from sklearn.datasets import load_digits

os.makedirs('tests_img', exist_ok=True)

FIXED = dict(
    population=20, generations=20, mutation_rate=0.2,
    tournament_size=5, epochs=500, learning_rate=0.01,
    lambda_=0.05, K=5, seed=42, plot=False,
    dataset=load_digits(),
)

VALUES = [0.001, 0.005, 0.01, 0.05, 0.1]
PARAM  = 'learning_rate'

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
    
    # Grafico aggiuntivo consigliato per Test 1
    fig2, ax = plt.subplots(figsize=(8, 5))
    test_accs = [r['test_accuracy'] for r in all_results]
    ax.bar(labels, test_accs, color='#344966')
    ax.axhline(baseline, color='red', linestyle='dashed', label=f'baseline ({baseline:.1f}%)')
    ax.set_title('Test accuracy per learning rate')
    ax.set_ylabel('Test accuracy (%)')
    ax.legend()
    plt.tight_layout()
    plt.savefig('tests_img/test_learning_rate_bar.png', dpi=150)
    plt.close()
    print(f"Test 1 completato.")
