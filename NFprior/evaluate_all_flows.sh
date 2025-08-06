# #!/bin/bash

# # Script to evaluate all normalizing flows for a given population type
# # Usage: ./evaluate_all_flows.sh [population_type]

# POPULATION_TYPE=${1:-"GW190425"}
# MODELS_BASE_PATH="./models"
# SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# echo "Evaluating all flows for population: $POPULATION_TYPE"
# echo "Models base path: $MODELS_BASE_PATH"

# # Check if population directory exists
# if [ ! -d "$MODELS_BASE_PATH/$POPULATION_TYPE" ]; then
#     echo "Error: Population directory $MODELS_BASE_PATH/$POPULATION_TYPE does not exist"
#     exit 1
# fi

# # Find all directories containing model.pt files
# echo "Searching for model directories..."
# MODEL_DIRS=$(find "$MODELS_BASE_PATH/$POPULATION_TYPE" -name "model.pt" -exec dirname {} \;)

# if [ -z "$MODEL_DIRS" ]; then
#     echo "No model directories found for population $POPULATION_TYPE"
#     exit 1
# fi

# echo "Found model directories:"
# echo "$MODEL_DIRS"
# echo ""

# # Test each model directory
# echo "Testing models..."
# echo "=================="

# for MODEL_DIR in $MODEL_DIRS; do
#     # Build full path to the Python script and run test
#     echo ""
#     echo "Testing: $MODEL_DIR"
#     echo "----------------------------------------"
#     python3 "$SCRIPT_DIR/evaluate_flows.py" "$MODEL_DIR" --test-only
    
#     # If test passes then do the full evaluate
#     echo ""
#     echo "Evaluating: $MODEL_DIR"
#     echo "----------------------------------------"
#     python3 "$SCRIPT_DIR/evaluate_flows.py" "$MODEL_DIR"
# done

#!/bin/bash

# Script to evaluate all normalizing flows for all population types
# Usage: ./evaluate_all_flows.sh

MODELS_BASE_PATH="./models"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Evaluating all flows for all population types"
echo "Models base path: $MODELS_BASE_PATH"

# Iterate over all subdirectories in models (each is a population type)
for POPULATION_TYPE in "$MODELS_BASE_PATH"/*; do
    if [ -d "$POPULATION_TYPE" ]; then
        POPULATION_NAME=$(basename "$POPULATION_TYPE")
        echo ""
        echo "========================================"
        echo "Evaluating population: $POPULATION_NAME"
        echo "========================================"

        # Find all directories containing model.pt files
        MODEL_DIRS=$(find "$POPULATION_TYPE" -name "model.pt" -exec dirname {} \;)

        if [ -z "$MODEL_DIRS" ]; then
            echo "No model directories found for $POPULATION_NAME"
            continue
        fi

        echo "Found model directories:"
        echo "$MODEL_DIRS"
        echo ""

        # Test and evaluate each model directory
        for MODEL_DIR in $MODEL_DIRS; do
            echo ""
            echo "Testing: $MODEL_DIR"
            echo "----------------------------------------"
            python3 "$SCRIPT_DIR/evaluate_flows.py" "$MODEL_DIR" --test-only

            echo ""
            echo "Evaluating: $MODEL_DIR"
            echo "----------------------------------------"
            python3 "$SCRIPT_DIR/evaluate_flows.py" "$MODEL_DIR"
        done
    fi
done
