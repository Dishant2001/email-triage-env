"""
Legacy sample inference script.

This repository's required baseline entrypoint is `inference.py` (repo root),
which runs the my_env inbox triage environment over 3 tasks (easy/medium/hard).
"""

from inference import main


if __name__ == "__main__":
    main()