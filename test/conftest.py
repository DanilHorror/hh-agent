"""Общие настройки тестов."""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture(autouse=True)
def no_letter_guard(monkeypatch):
    """Страж письма ходит к настоящей модели-судье — в тестах он выключен (тесты стража включают его сами)."""
    from config import settings
    monkeypatch.setattr(settings, "letter_guard", False)
    # защита от мусора отсекла бы короткие тестовые вакансии; её тесты — test_safety.py
    monkeypatch.setattr(settings, "input_defense", False)
