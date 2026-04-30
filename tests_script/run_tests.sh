#!/usr/bin/env bash
# Avvia un test alla volta e salva output in tests_img/run_YYYYMMDD_HHMMSS/<test>/
# Uso:
#   ./run_tests.sh          → menu interattivo
#   ./run_tests.sh 1        → esegue test 1
#   ./run_tests.sh 9A       → esegue test 9A
#   ./run_tests.sh all      → esegue tutti in sequenza

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TESTS_DIR="$PROJECT_DIR/tests_script"
IMG_DIR="$PROJECT_DIR/tests_img"
RUN_ID="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="$IMG_DIR/run_$RUN_ID"

declare -A TEST_FILE=(
    [1]="test_1_learning_rate.py"
    [2]="test_2_epochs.py"
    [3]="test_3_lambda.py"
    [4]="test_4_k.py"
    [5]="test_5_population.py"
    [6]="test_6_generations.py"
    [7]="test_7_mutation_rate.py"
    [8]="test_8_dataset.py"
    [9A]="test_9A_leakage.py"
    [9B]="test_9B_leakage.py"
)

declare -A TEST_DESC=(
    [1]="Learning Rate"
    [2]="Epochs"
    [3]="Lambda (penalità complessità)"
    [4]="K (run per individuo)"
    [5]="Population size"
    [6]="Generations"
    [7]="Mutation rate"
    [8]="Dataset comparison"
    [9A]="Leakage A — bias ottimistico GA"
    [9B]="Leakage B — scaler leakage"
)

TEST_ORDER=(1 2 3 4 5 6 7 8 9A 9B)

print_menu() {
    echo ""
    echo "=============================="
    echo "  Test disponibili"
    echo "=============================="
    for k in "${TEST_ORDER[@]}"; do
        printf "  %-4s %s\n" "$k" "${TEST_DESC[$k]}"
    done
    echo "  all  Esegui tutti in sequenza"
    echo "=============================="
    echo ""
}

run_one() {
    local key="$1"
    local file="${TEST_FILE[$key]:-}"

    if [[ -z "$file" ]]; then
        echo "[ERRORE] Test '$key' non esiste."
        return 1
    fi

    local out_dir="$RUN_DIR/test_$key"
    mkdir -p "$out_dir"

    echo ""
    echo ">>> Test $key — ${TEST_DESC[$key]}"
    echo "    Output: $out_dir"
    echo ""

    # marker per trovare file generati durante questo test
    local marker
    marker=$(mktemp "$IMG_DIR/.marker_XXXXXX")

    cd "$PROJECT_DIR"
    python "$TESTS_DIR/$file" 2>&1 | tee "$out_dir/run.log"

    # sposta png e csv generati dopo il marker
    find "$IMG_DIR" -maxdepth 1 -newer "$marker" \( -name "*.png" -o -name "*.csv" \) \
        -exec mv {} "$out_dir/" \; 2>/dev/null || true

    rm -f "$marker"

    echo ""
    echo ">>> Test $key completato. File salvati in: $out_dir"
}

# ── punto di ingresso ──────────────────────────────────────────────────────────

mkdir -p "$RUN_DIR"

CHOICE="${1:-}"

if [[ -z "$CHOICE" ]]; then
    print_menu
    read -rp "Inserisci numero test (es. 1, 9A, all): " CHOICE
fi

if [[ "$CHOICE" == "all" ]]; then
    for k in "${TEST_ORDER[@]}"; do
        run_one "$k"
    done
    echo ""
    echo "=============================="
    echo "  Tutti i test completati."
    echo "  Risultati in: $RUN_DIR"
    echo "=============================="
else
    run_one "$CHOICE"
fi
