#!/usr/bin/env python3

import subprocess
import sys
import json
import requests
import logging
import argparse
import re
from typing import Dict, Any, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(asctime)s %(process)d %(filename)s:%(lineno)d] %(message)s",
    datefmt="%m%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def get_staged_diff() -> str:
    try:
        result = subprocess.run(
            ["git", "diff", "--staged"], capture_output=True, text=True, check=True
        )
        return result.stdout
    except subprocess.CalledProcessError:
        logger.error("Failed to get git diff --staged")
        sys.exit(1)


def check_staged_changes(diff: str) -> bool:
    if not diff.strip():
        return False
    return True


def create_prompt(staged_diff: str) -> str:
    prompt = (
        """Based on the following changes, generate a Git commit message.
- The commit message must be one line.
- Markdown formatting is not allowed.

Changes:
```
$ git diff --staged
"""
        + staged_diff
        + """```"""
    )
    return prompt


def pull_model(model_name: str, server_url: str) -> bool:
    payload = {"name": model_name}

    try:
        logger.info(f"Pulling model {model_name}")
        response = requests.post(
            f"{server_url}/api/pull",
            headers={"Content-Type": "application/json"},
            json=payload,
        )
        response.raise_for_status()
        logger.info(f"Successfully pulled model {model_name}")
        return True
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to pull model {model_name}: {e}")
        return False


def generate_commit_message(
    prompt: str, model_name: str, server_url: str
) -> Dict[str, Any]:
    payload = {"model": model_name, "prompt": prompt, "stream": False}

    try:
        response = requests.post(
            f"{server_url}/api/generate",
            headers={"Content-Type": "application/json"},
            json=payload,
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to connect to Ollama server: {e}")
        sys.exit(1)


def extract_commit_message(
    response_data: Dict[str, Any], model_name: str
) -> Optional[str]:
    """Extract commit message from Ollama response, removing <think> tags for qwen3:8b."""
    try:
        response_text = response_data.get("response", "")
        if not response_text:
            logger.error("No response text found in Ollama response")
            return None

        if model_name == "qwen3:8b":
            cleaned_text = re.sub(
                r"<think>.*?</think>", "", response_text, flags=re.DOTALL
            )
        else:
            cleaned_text = response_text

        commit_message = cleaned_text.strip()

        if not commit_message:
            logger.error("No commit message found")
            return None

        return commit_message

    except Exception as e:
        logger.error(f"Failed to extract commit message: {e}")
        return None


def execute_git_commit(commit_message: str) -> None:
    """Execute git commit with the generated message."""
    try:
        logger.info("Executing git commit with generated message")
        subprocess.run(["git", "commit", "-m", commit_message, "-e"], check=True)
        logger.info("Successfully committed changes")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to commit changes: {e}")
        sys.exit(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Git commit message using Ollama"
    )
    parser.add_argument(
        "--server",
        default="http://localhost:11434",
        help="Ollama server URL (default: http://localhost:11434)",
    )
    parser.add_argument(
        "--model", default="qwen3:8b", help="Ollama model name (default: qwen3:8b)"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    staged_diff = get_staged_diff()
    if not check_staged_changes(staged_diff):
        logger.info("No staged changes found. Exiting.")
        sys.exit(0)
    if not pull_model(args.model, args.server):
        logger.error(
            f"Failed to pull model {args.model}. Please check the server and model name."
        )
        sys.exit(1)
    prompt = create_prompt(staged_diff)
    logger.debug("Generated prompt:")
    logger.debug(prompt)
    response = generate_commit_message(prompt, args.model, args.server)

    commit_message = extract_commit_message(response, args.model)
    if commit_message:
        logger.info("Generated commit message:")
        logger.info(commit_message)
        execute_git_commit(commit_message)
    else:
        logger.error("Failed to extract commit message from response")
        logger.error("Raw response:")
        logger.error(json.dumps(response, indent=2))
        sys.exit(1)


if __name__ == "__main__":
    main()
