import csv
import sys
import os
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from main import run
from sklearn.datasets import load_iris, load_wine, load_breast_cancer, load_digits

os.makedirs('tests_img', exist_ok=True)

FIXED = dict(
    population=20, generations=50, mutation_rate=0.2,
    tournament_size=5, epochs=500, learning_rate=0.05,
    lambda_=0.05, K=3, seed=42, plot=False,
)

DATASETS = {
    'iris':           load_iris(),
    'wine':           load_wine(),
    'breast_cancer':  load_breast_cancer(),
    'digits':         load_digits(),
}

if __name__ == '__main__':
    csv_rows   = []
    all_results = []

    for name, ds in DATASETS.items():
        print(f"Esecuzione dataset {name}...")
        result = run(**{**FIXED, 'dataset': ds})

        csv_rows.append({
            'dataset':           name,
            'n_feature':         ds.data.shape[1],
            'n_classi':          len(set(ds.target)),
            'n_campioni':        len(ds.target),
            'test_accuracy':     result['test_accuracy'],
            'accuracy_baseline': result['accuracy_baseline'],
            'gap_val_test':      round(result['best_accuracy'] - result['test_accuracy'], 2),
            'n_params':          result['n_params'],
            'n_params_baseline': result['n_params_baseline'],
            'n_layer':           len(result['best_individuo']),
            'best_individuo':    str(result['best_individuo']),
        })

        all_results.append({'label': name, **result})

    with open('tests_img/test_dataset.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
        writer.writeheader()
        writer.writerows(csv_rows)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    names            = [r['label']              for r in all_results]
    test_accs        = [r['test_accuracy']      for r in all_results]
    baselines        = [r['accuracy_baseline']  for r in all_results]
    nparams          = [r['n_params']           for r in all_results]
    nparams_baseline = [r['n_params_baseline']  for r in all_results]

    x = np.arange(len(names))
    w = 0.35

    axes[0].bar(x - w/2, test_accs, w, label='GA test accuracy', color='#344966')
    axes[0].bar(x + w/2, baselines, w, label='Baseline',          color='#C30000B2')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names)
    axes[0].set_ylabel('Accuracy (%)')
    axes[0].set_title('GA vs Baseline per dataset')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')

    axes[1].bar(x - w/2, nparams,          w, label='GA',      color='steelblue')
    axes[1].bar(x + w/2, nparams_baseline, w, label='Baseline', color='#C30000B2')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names)
    axes[1].set_ylabel('#parametri')
    axes[1].set_title('#parametri GA vs Baseline per dataset')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.suptitle('Test dataset — architettura trovata dal GA', fontsize=13)
    plt.tight_layout()
    plt.savefig('tests_img/test_dataset.png', dpi=150)
    plt.close()
    print("Test 8 completato.")
