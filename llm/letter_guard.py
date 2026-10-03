"""Страж письма: проверить черновик после генерации и исправить, пока в нём есть выдумки или штампы.

    письмо → судья (выдумки) + проверка кодом (штампы)
           ─ чисто → отдать как есть
           ─ есть проблемы → модель исправляет ровно их → снова проверка (не больше max_rounds раз)
           ─ не исправилось → отдать с предупреждением: человек увидит, что именно проверить

Модуль не знает, какая модель пишет и какая судит: check и fix передаются снаружи,
поэтому логика проверяется тестами с заглушками.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Awaitable, Callable

Check = Callable[[str], Awaitable[list[str]]]                 # письмо → выдуманные утверждения
Fix = Callable[[str, list[str], list[str]], Awaitable[str]]    # (письмо, выдумки, штампы) → исправленное письмо
FindCliches = Callable[[str], list[str]]


@dataclass
class GuardResult:
    letter: str
    clean: bool                     # выдумок и штампов не осталось
    rounds: int                     # сколько раз письмо исправляли
    invented: list[str] = field(default_factory=list)   # выдумки, которые остались
    cliches: list[str] = field(default_factory=list)    # штампы, которые остались
    history: list[dict] = field(default_factory=list)   # что находили на каждом круге

    @property
    def warning(self) -> str:
        """Текст предупреждения для человека; пусто, если выдумок не осталось (штампы — не повод тревожить)."""
        return "Возможно, этого нет в резюме: " + "; ".join(self.invented) if self.invented else ""

    def as_dict(self) -> dict:
        return {"clean": self.clean, "rounds": self.rounds, "invented": self.invented,
                "cliches": self.cliches, "history": self.history}


async def guard_letter(letter: str, *, check: Check, fix: Fix, find_cliches: FindCliches,
                       max_rounds: int = 2) -> GuardResult:
    current = letter
    history: list[dict] = []
    invented: list[str] = []
    cliches: list[str] = []
    for round_no in range(max_rounds + 1):
        invented = await check(current)
        cliches = find_cliches(current)
        history.append({"round": round_no, "invented": invented, "cliches": cliches})
        if not invented and not cliches:
            return GuardResult(current, True, round_no, [], [], history)
        if round_no == max_rounds:
            break
        fixed = (await fix(current, invented, cliches)).strip()
        if fixed:
            current = fixed
    return GuardResult(current, False, max_rounds, invented, cliches, history)
