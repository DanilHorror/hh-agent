"""Запускает старые проверки-скрипты (без сети и браузера) как часть pytest.

Каждый скрипт печатает OK/FAIL по пунктам и завершается с кодом 0 или 1.
"""
import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = ["test_keywords.py", "test_llm_keywords.py", "test_llm_gigachat.py"]


@pytest.mark.parametrize("script", SCRIPTS)
def test_offline_script(script):
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    proc = subprocess.run(
        [sys.executable, os.path.join(ROOT, "test", script)],
        cwd=ROOT, env=env, capture_output=True, timeout=180,
    )
    output = (proc.stdout + proc.stderr).decode("utf-8", errors="replace")
    assert proc.returncode == 0, output[-3000:]
