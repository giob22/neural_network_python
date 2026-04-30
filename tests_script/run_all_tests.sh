#!/bin/bash

cd $(dirname $0)/..

TESTS=(
    "tests_script/test_1_learning_rate.py"
    "tests_script/test_2_epochs.py"
    "tests_script/test_3_lambda.py"
    "tests_script/test_4_k.py"
    "tests_script/test_5_population.py"
    "tests_script/test_6_generations.py"
    "tests_script/test_7_mutation_rate.py"
    "tests_script/test_8_dataset.py"
    "tests_script/test_9A_leakage.py"
    "tests_script/test_9B_leakage.py"
)

if [ -n "$1" ]; then
    echo "Cerco il test corrispondente a: $1"
    FILTERED_TESTS=()
    for t in "${TESTS[@]}"; do
        if [[ "$t" == *"$1"* ]]; then
            FILTERED_TESTS+=("$t")
        fi
    done
    
    if [ ${#FILTERED_TESTS[@]} -eq 0 ]; then
        echo "Nessun test trovato corrispondente a '$1'."
        echo "Usa ad esempio: 1, 9A, epochs, leakage"
        exit 1
    fi
    TESTS=("${FILTERED_TESTS[@]}")
else
    echo "Esecuzione di tutti i test..."
fi

for test_script in "${TESTS[@]}"; do
    echo "-> Avvio: $test_script"
    .venv/bin/python "$test_script"
    if [ $? -ne 0 ]; then
        echo "Errore durante l'esecuzione di $test_script"
        exit 1
    fi
done

echo "Tutti i test richiesti sono completati!"
