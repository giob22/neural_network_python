import csv
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from main import one_hot, mean_accuracy_on_K_runs, number_params
from neural_network.nn_layer import leaky_relu
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

os.makedirs('tests_img', exist_ok=True)

SEED  = 42
K     = 5
LR    = 0.01
EPOCHS = 150
CROMOSOMA = [(8, leaky_relu), (8, leaky_relu)]

if __name__ == '__main__':
    print("Esecuzione test Leakage B...")
    dataset = load_iris()
    x = dataset.data
    y = dataset.target
    n_classi = len(set(y))
    
    x_trainval, x_test, y_trainval, y_test = train_test_split(
        x, y, test_size=0.2, random_state=SEED
    )
    x_train, x_val, y_train, y_val = train_test_split(
        x_trainval, y_trainval, test_size=0.25, random_state=SEED
    )
    
    # ── Scenario CORRETTO: scaler fittato solo su x_train ────────────────
    scaler_ok = StandardScaler()
    x_train_ok = scaler_ok.fit_transform(x_train)
    x_val_ok   = scaler_ok.transform(x_val)
    x_test_ok  = scaler_ok.transform(x_test)
    
    y_train_oh = one_hot(y_train, n_classi)
    y_test_oh  = one_hot(y_test,  n_classi)
    
    acc_corretto = mean_accuracy_on_K_runs(
        CROMOSOMA, x.shape[1], n_classi, LR, EPOCHS,
        y_train_oh, x_train_ok, y_test_oh, x_test_ok,
        K=K, seed=SEED
    )
    
    # ── Scenario SBAGLIATO: scaler fittato su tutto x (train+val+test) ───
    scaler_leak = StandardScaler()
    x_all_scaled = scaler_leak.fit_transform(x)          # ← leakage qui
    
    x_tv_leak, x_test_leak, y_tv_leak, y_test_leak = train_test_split(
        x_all_scaled, y, test_size=0.2, random_state=SEED
    )
    x_train_leak, _, y_train_leak, _ = train_test_split(
        x_tv_leak, y_tv_leak, test_size=0.25, random_state=SEED
    )
    
    y_train_leak_oh = one_hot(y_train_leak, n_classi)
    y_test_leak_oh  = one_hot(y_test_leak,  n_classi)
    
    acc_leakage = mean_accuracy_on_K_runs(
        CROMOSOMA, x.shape[1], n_classi, LR, EPOCHS,
        y_train_leak_oh, x_train_leak, y_test_leak_oh, x_test_leak,
        K=K, seed=SEED
    )
    
    # ── risultati ─────────────────────────────────────────────────────────
    acc_corretto_pct = round(acc_corretto * 100, 2)
    acc_leakage_pct  = round(acc_leakage  * 100, 2)
    gap              = round(acc_leakage_pct - acc_corretto_pct, 2)
    
    print(f"Scaler corretto (fit su train): {acc_corretto_pct}%")
    print(f"Scaler sbagliato (fit su tutto): {acc_leakage_pct}%")
    print(f"Gonfiamento artificiale:         +{gap}%")
    
    csv_rows = [
        {'scenario': 'scaler_corretto',  'test_accuracy': acc_corretto_pct, 'gap': 0},
        {'scenario': 'scaler_leakage',   'test_accuracy': acc_leakage_pct,  'gap': gap},
    ]
    with open('tests_img/test_data_leakage_B.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
        writer.writeheader()
        writer.writerows(csv_rows)
    
    # ── grafico ──────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.bar(['Corretto\n(fit su train)', 'Sbagliato\n(fit su tutto)'],
           [acc_corretto_pct, acc_leakage_pct],
           color=['#344966', '#C30000'])
    ax.set_ylabel('Test accuracy (%)')
    ax.set_title(f'Leakage B — scaler\nGonfiamento artificiale: +{gap}%')
    ax.set_ylim(max(0, min(acc_corretto_pct, acc_leakage_pct) - 10), 100)
    ax.grid(True, alpha=0.3, axis='y')
    for i, v in enumerate([acc_corretto_pct, acc_leakage_pct]):
        ax.text(i, v + 0.5, f'{v}%', ha='center', fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig('tests_img/test_data_leakage_B.png', dpi=150)
    plt.close()
    print("Test 9B completato.")
