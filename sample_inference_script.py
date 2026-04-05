"""
Legacy sample inference script.

This repository's baseline entrypoint is `inference.py` (repo root), which runs
EmailTriageEnv over 3 tasks using the configured OpenAI model (requires API key + model name).
"""

from inference import main


if __name__ == "__main__":
    main()