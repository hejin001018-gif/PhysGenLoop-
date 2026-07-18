#!/bin/bash

# ==========================
# WanPhysics Batch Generator
# Mac Controller
# ==========================


PROJECT_DIR="$HOME/Desktop/WanPhysics"

PROMPT_DIR="$PROJECT_DIR/prompts"


if [ ! -d "$PROMPT_DIR" ]; then
    echo "Prompt directory not found:"
    echo "$PROMPT_DIR"
    exit 1
fi


echo "================================="
echo "WanPhysics Batch Generation"
echo "================================="


for FILE in "$PROMPT_DIR"/*.txt
do

    if [ ! -f "$FILE" ]; then
        echo "No prompt files found."
        exit 0
    fi

    ID=$(basename "$FILE" .txt)
    PROMPT=$(cat "$FILE")
    echo ""
    echo "================================="
    echo "Task ID:"
    echo "$ID"
    echo "Prompt:"
    echo "$PROMPT"
    echo "================================="
    ./scripts/run_wan.sh "$ID" "$PROMPT"
    if [ $? -ne 0 ]; then
        echo "Task failed:"
        echo "$ID"
        exit 1
    fi
done
echo ""
echo "================================="
echo "All tasks finished!"
echo "================================="