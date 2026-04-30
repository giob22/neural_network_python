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
    lambda_=0.05, plot=False,
    dataset=load_digits(),
)

VALUES = [1, 3, 5, 10, 20]
PARAM  = 'K'

if __name__ == '__main__':
    csv_rows  = []
    all_results = []
    SEEDS = [42, 123, 7, 99, 0]
    
    for v in VALUES:
        print(f"Esecuzione con {PARAM}={v}...")
        accs = []
        
        # Eseguiamo prima per i vari seeds solo per raccogliere le accuracy
        for s in SEEDS:
            r = run(**{**FIXED, 'K': v, 'seed': s})
            accs.append(r['test_accuracy'])
            if s == 42:
                # teniamo l'esecuzione con seed 42 per la storia
                result = r
                
        mean_test = sum(accs) / len(accs)
        std_test  = (sum((a - mean_test)**2 for a in accs) / len(accs))**0.5
    
        csv_rows.append({
            PARAM:               v,
            'mean_test':         round(mean_test, 4),
            'std_test':          round(std_test, 4),
            'min_test':          round(min(accs), 4),
            'max_test':          round(max(accs), 4),
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
    print("Test 4 completato.")
