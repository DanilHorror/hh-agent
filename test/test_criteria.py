"""Критерии кандидата: разбор файла, раздел промпта, фильтр компаний."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm import criteria  # noqa: E402

TEXT = """# комментарий
[нет]
1С
C# / .NET
[плюс]
- Linux
[формат]
удалёнка
[компании]
# скептически
МТС
ПСБ
Перфоманс Лаб
[неизвестный раздел]
что-то
"""


def test_parse_keeps_hash_inside_words():
    c = criteria.parse(TEXT)
    assert c["нет"] == ["1С", "C# / .NET"]          # «C#» не обрезается как комментарий
    assert c["плюс"] == ["Linux"] and c["формат"] == ["удалёнка"]
    assert c["компании"] == ["МТС", "ПСБ", "Перфоманс Лаб"]


def test_prompt_block_contains_sections_and_scale():
    block = criteria.prompt_block(criteria.parse(TEXT))
    assert "НЕ ПОДХОДИТ" in block and "- C# / .NET" in block and "ШКАЛА" in block
    assert criteria.prompt_block(criteria.parse("")) == ""


def test_company_filter_matches_whole_words():
    c = criteria.parse(TEXT)
    assert criteria.blocked_company('"МТС", Работа в IT', c) == "МТС"
    assert criteria.blocked_company("ПАО Банк ПСБ, Бизнес", c) == "ПСБ"
    assert criteria.blocked_company("ООО Перфоманс Лаб", c) == "Перфоманс Лаб"
    assert criteria.blocked_company("Мтсофт", c) is None           # не часть другого слова
    assert criteria.blocked_company("Сбер. IT", c) is None
    assert criteria.blocked_company("", c) is None


def test_missing_file_means_no_criteria(tmp_path):
    c = criteria.load(str(tmp_path / "nope.txt"))
    assert all(v == [] for v in c.values())
