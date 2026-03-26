#!/usr/bin/env python3
"""Restructure script for cardiotoxicity-prediction repository.

Moves core backend modules into prediction_backend/ and rewrites Python imports.
Idempotent: safe to run multiple times.
"""

import os
import re
import shutil
from pathlib import Path

EXCLUDE_DIRS = {'.git', '.venv', '.pytest_cache', '.vscode'}
MODULES_TO_MOVE = [
    'embeddings',
    'features',
    'molecular_processing',
    'models',
    'inference',
    'evaluation',
    'tests',
]
TARGET_DIR_NAME = 'prediction_backend'


def log(msg: str):
    print(f"[INFO] {msg}")


def is_excluded(path: Path):
    return any(part in EXCLUDE_DIRS for part in path.parts)


def move_directories(root: Path):
    target_root = root / TARGET_DIR_NAME
    target_root.mkdir(exist_ok=True)

    for module in MODULES_TO_MOVE:
        src = root / module
        dst = target_root / module

        if src.exists() and src.is_dir():
            if dst.exists():
                log(f"Skipping move because destination exists: {module} -> {TARGET_DIR_NAME}/{module}")
            else:
                log(f"Moving {module} → {TARGET_DIR_NAME}")
                shutil.move(str(src), str(dst))
        else:
            if dst.exists() and dst.is_dir():
                log(f"Already moved: {module} is in {TARGET_DIR_NAME}")
            else:
                log(f"Directory not found (skipping): {module}")


def ensure_init_file(root: Path):
    target_root = root / TARGET_DIR_NAME
    target_root.mkdir(exist_ok=True)
    init_file = target_root / '__init__.py'

    if not init_file.exists():
        log(f"Creating {TARGET_DIR_NAME}/__init__.py")
        init_file.write_text('# Auto-generated package init\n', encoding='utf-8')
    else:
        log(f"{TARGET_DIR_NAME}/__init__.py already exists")


def update_imports(root: Path):
    for py_file in root.rglob('*.py'):
        if is_excluded(py_file):
            continue

        if py_file.name == os.path.basename(__file__):
            # Don't rewrite imports in this script itself
            continue

        text = py_file.read_text(encoding='utf-8')

        lines = text.splitlines(keepends=True)
        changed = False
        new_lines = []
        for line in lines:
            stripped = line.strip()
            if not stripped:
                new_lines.append(line)
                continue

            updated_line = line
            for module in MODULES_TO_MOVE:
                if re.match(r'^(\s*)(from|import)\s+' + re.escape(module) + r'\b', line):
                    if re.match(r'^(\s*)(from|import)\s+' + re.escape(TARGET_DIR_NAME) + r'\.' + re.escape(module) + r'\b', line):
                        # already rewritten
                        continue
                    updated_line = re.sub(
                        r'^(\s*)(from|import)(\s+)' + re.escape(module) + r'(\b)',
                        r'\1\2\3' + TARGET_DIR_NAME + '.' + module + r'\4',
                        line,
                    )
                    break

            if updated_line != line:
                changed = True
                log(f"Updating imports in {py_file}")
            new_lines.append(updated_line)

        if changed:
            py_file.write_text(''.join(new_lines), encoding='utf-8')


        new_lines = []
        for line in lines:
            stripped = line.strip()
            if not stripped:
                new_lines.append(line)
                continue

            updated_line = line
            for module in MODULES_TO_MOVE:
                if re.match(r'^(\s*)(from|import)\s+' + re.escape(module) + r'\b', line):
                    if re.match(r'^(\s*)(from|import)\s+' + re.escape(TARGET_DIR_NAME) + r'\.' + re.escape(module) + r'\b', line):
                        # already rewritten
                        continue
                    updated_line = re.sub(r'^(\s*)(from|import)(\s+)'+re.escape(module)+r'(\b)',
                                          r'\1\2\3' + TARGET_DIR_NAME + '.' + module + r'\4', line)
                    break

            if updated_line != line:
                changed = True
                log(f"Updating imports in {py_file}")
            new_lines.append(updated_line)

        if changed:
            py_file.write_text(''.join(new_lines), encoding='utf-8')


def main():
    root = Path(__file__).resolve().parent

    log(f"Detected project root: {root}")

    move_directories(root)
    ensure_init_file(root)
    update_imports(root)

    log('Refactoring complete')


if __name__ == '__main__':
    main()
