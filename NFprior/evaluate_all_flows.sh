#!/bin/bash

# Flag to skip directories containing "flowjax" in their name
skip_flowjax=true

MODELS_BASE_PATH="./models"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Evaluating all flows for all population types"
echo "Models base path: $MODELS_BASE_PATH"
if [ "$skip_flowjax" = true ]; then
    echo "Skip FlowJAX mode: ON (skipping flowjax directories)"
else
    echo "Skip FlowJAX mode: OFF"
fi

# Iterate over all subdirectories in models (each is a population type)
for POPULATION_TYPE in "$MODELS_BASE_PATH"/uniform "$MODELS_BASE_PATH"/gaussian "$MODELS_BASE_PATH"/double_gaussian; do
    if [ -d "$POPULATION_TYPE" ]; then
        POPULATION_NAME=$(basename "$POPULATION_TYPE")
        
        echo ""
        echo "========================================"
        echo "Evaluating population: $POPULATION_NAME"
        echo "========================================"

        # Find all directories containing model.pt or model.eqx files
        MODEL_DIRS=$(find "$POPULATION_TYPE" \( -name "model.pt" -o -name "model.eqx" \) -exec dirname {} \;)
        
        if [ -z "$MODEL_DIRS" ]; then
            echo "No model directories found for $POPULATION_NAME"
            continue
        fi

        echo "Found model directories:"
        echo "$MODEL_DIRS"
        echo ""

        # Test and evaluate each model directory
        for MODEL_DIR in $MODEL_DIRS; do
            echo "Model dir: $MODEL_DIR"
            
            # Skip directories that contain "flowjax" if flag is set
            if [ "$skip_flowjax" = true ] && [[ "$MODEL_DIR" == *"flowjax"* ]]; then
                echo "Skipping $MODEL_DIR (contains flowjax)"
                continue
            fi
            echo ""
            echo "Testing: $MODEL_DIR"
            echo "----------------------------------------"
            python3 "$SCRIPT_DIR/evaluate_flows.py" "$MODEL_DIR" --test-only

            echo ""
            echo "Evaluating: $MODEL_DIR"
            echo "----------------------------------------"
            python3 "$SCRIPT_DIR/evaluate_flows.py" "$MODEL_DIR" --n-samples 200_000
        done
    fi
done