#!/bin/bash

TEMP_MESSAGE_FILE=$(mktemp)
STAGED_DIFF=$(git diff --staged)

# Prompt for generating the commit message
# Adjust the prompt as needed
PROMPT="Based on the following changes, generate a concise and clear Git commit message. The commit message should have a subject line of up to 72 characters, followed by a detailed body. You must not append explicit 'Subject' or 'Body' inside the commit message. \n\nChanges:\n$STAGED_DIFF"

# Use Ollama to generate the commit message and save it to the temporary file
if ! ollama run qwen2.5-coder:3b-instruct "$PROMPT" > "$TEMP_MESSAGE_FILE"; then
    echo "Error: Failed to generate commit message with Ollama."
    rm "$TEMP_MESSAGE_FILE"
    exit 1
fi

# Open the generated commit message in an editor for editing
echo "---"
echo "Editing the commit message. Please save the file and close the editor when done."
echo "---"

# Launch the editor with git commit -eF using the generated message file
if ! git commit -eF "$TEMP_MESSAGE_FILE"; then
    echo "Error: Commit failed."
    rm "$TEMP_MESSAGE_FILE"
    exit 1
fi

# Remove the temporary file
rm "$TEMP_MESSAGE_FILE"

echo "Commit completed successfully."
