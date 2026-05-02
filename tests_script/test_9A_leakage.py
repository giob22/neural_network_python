import csv
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from main import run
from sklearn.datasets import load_digits

os.makedirs('tests_img', exist_ok=True)

FIXED = dict(
    population=20, generations=50, mutation_rate=0.2,
    tournament_size=5, epochs=500, learning_rate=0.05,
    lambda_=0.05, K=3, seed=42, plot=False,
    dataset=load_digits(),
)

SEEDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

if __name__ == '__main__':
    csv_rows = []
    val_accs  = []
    test_accs = []
    gaps      = []

    for s in SEEDS:
        print(f"Esecuzione seed={s}...")
        result = run(**{**FIXED, 'seed': s})

        val_acc  = result['best_accuracy']
        test_acc = result['test_accuracy']
        gap      = round(val_acc - test_acc, 2)

        val_accs.append(val_acc)
        test_accs.append(test_acc)
        gaps.append(gap)

        csv_rows.append({
            'seed':              s,
            'val_accuracy':      val_acc,
            'test_accuracy':     test_acc,
            'gap_val_test':      gap,
            'accuracy_baseline': result['accuracy_baseline'],
            'n_params':          result['n_params'],
        })

    mean_gap = round(sum(gaps) / len(gaps), 2)
    std_gap  = round(np.std(gaps), 2)

    # aggiungi riga di riepilogo
    csv_rows.append({
        'seed':              'MEDIA',
        'val_accuracy':      round(sum(val_accs)  / len(val_accs),  2),
        'test_accuracy':     round(sum(test_accs) / len(test_accs), 2),
        'gap_val_test':      mean_gap,
        'accuracy_baseline': csv_rows[0]['accuracy_baseline'],
        'n_params':          '',
    })

    with open('tests_img/test_data_leakage_A.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
        writer.writeheader()
        writer.writerows(csv_rows)

    # ── grafico ──────────────────────────────────────────────────────────
    x = np.arange(len(SEEDS))
    w = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ax1.bar(x - w/2, val_accs,  w, label='val accuracy (GA, ottimistica)', color='#8B008B', alpha=0.85)
    ax1.bar(x + w/2, test_accs, w, label='test accuracy (stima reale)',     color='#344966', alpha=0.85)
    ax1.axhline(csv_rows[0]['accuracy_baseline'], color='red', linestyle='dashed', label='baseline')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'seed={s}' for s in SEEDS], rotation=30)
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title('Val vs Test accuracy per seed\n(gap = bias ottimistico)')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3, axis='y')

    ax2.bar(x, gaps, color='#C30000B2')
    ax2.axhline(mean_gap, color='black', linestyle='dashed',
                label=f'gap medio = {mean_gap}% ± {std_gap}%')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'seed={s}' for s in SEEDS], rotation=30)
    ax2.set_ylabel('Gap val - test (%)')
    ax2.set_title('Bias ottimistico per seed\n(sempre positivo = leakage sistematico)')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3, axis='y')

    plt.suptitle(
        f'Leakage A — bias ottimistico GA: gap medio {mean_gap}% ± {std_gap}%',
        fontsize=12
    )
    plt.tight_layout()
    plt.savefig('tests_img/test_data_leakage_A.png', dpi=150)
    plt.close()
    print("Test 9A completato.")
