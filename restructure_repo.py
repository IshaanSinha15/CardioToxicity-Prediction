#!/usr/bin/env python3
"""
Repository Restructuring Script

This script reorganizes the CardioToxicity-Prediction repository by moving
ML inference modules into a 'prediction_backend' folder and updating imports.

Usage: python restructure_repo.py
"""

import os
import re
import shutil
from pathlib import Path


def log(message: str):
    """Print a log message with [INFO] prefix."""
    print(f"[INFO] {message}")


def get_project_root() -> Path:
    """Detect the project root by finding the directory containing this script."""
    return Path(__file__).parent.resolve()


def create_prediction_backend(root: Path):
    """Create the prediction_backend directory if it doesn't exist."""
    backend_dir = root / "prediction_backend"
    if not backend_dir.exists():
        backend_dir.mkdir()
        log("Created prediction_backend/")
    else:
        log("prediction_backend/ already exists")


def move_directories(root: Path):
    """Move specified directories into prediction_backend."""
    directories_to_move = [
        "embeddings",
        "features",
        "molecular_processing",
        "models",
        "inference",
        "evaluation",
        "tests"
    ]

    backend_dir = root / "prediction_backend"

    for dir_name in directories_to_move:
        src = root / dir_name
        dst = backend_dir / dir_name

        if src.exists() and src.is_dir():
            if dst.exists():
                log(f"{dir_name} already in prediction_backend, skipping move")
            else:
                shutil.move(str(src), str(dst))
                log(f"Moved {dir_name} → prediction_backend/")
        else:
            log(f"{dir_name} not found, skipping")


def create_init_file(root: Path):
    """Create __init__.py in prediction_backend if it doesn't exist."""
    init_file = root / "prediction_backend" / "__init__.py"
    if not init_file.exists():
        init_file.write_text("# Prediction Backend Package\n")
        log("Created prediction_backend/__init__.py")
    else:
        log("prediction_backend/__init__.py already exists")


def should_skip_path(path: Path) -> bool:
    """Check if a path should be skipped during file scanning."""
    skip_dirs = {".git", ".venv", ".pytest_cache", ".vscode", "__pycache__", "node_modules"}
    return any(part in skip_dirs for part in path.parts)


def update_imports_in_file(file_path: Path):
    """Update import statements in a single Python file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Patterns to replace
        replacements = {
            r'\bfrom embeddings\.': 'from prediction_backend.embeddings.',
            r'\bfrom features\.': 'from prediction_backend.features.',
            r'\bfrom molecular_processing\.': 'from prediction_backend.molecular_processing.',
            r'\bfrom models\.': 'from prediction_backend.models.',
            r'\bfrom inference\.': 'from prediction_backend.inference.',
            r'\bfrom evaluation\.': 'from prediction_backend.evaluation.',
            r'\bfrom tests\.': 'from prediction_backend.tests.',
            r'\bimport embeddings\.': 'import prediction_backend.embeddings.',
            r'\bimport features\.': 'import prediction_backend.features.',
            r'\bimport molecular_processing\.': 'import prediction_backend.molecular_processing.',
            r'\bimport models\.': 'import prediction_backend.models.',
            r'\bimport inference\.': 'import prediction_backend.inference.',
            r'\bimport evaluation\.': 'import prediction_backend.evaluation.',
            r'\bimport tests\.': 'import prediction_backend.tests.',
        }

        original_content = content
        for pattern, replacement in replacements.items():
            content = re.sub(pattern, replacement, content)

        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            log(f"Updated imports in {file_path.name}")

    except Exception as e:
        log(f"Error updating {file_path}: {e}")


def update_all_imports(root: Path):
    """Scan all .py files and update imports."""
    log("Scanning for Python files to update imports...")

    for py_file in root.rglob("*.py"):
        if not should_skip_path(py_file):
            update_imports_in_file(py_file)

    log("Import updates complete")


def main():
    """Main function to run the restructuring."""
    log("Starting repository restructuring...")

    root = get_project_root()
    log(f"Project root: {root}")

    create_prediction_backend(root)
    move_directories(root)
    create_init_file(root)
    update_all_imports(root)

    log("Refactoring complete")


if __name__ == "__main__":
    main()