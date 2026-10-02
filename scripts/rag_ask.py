"""Вопрос к базе вакансий из консоли.

    venv\\Scripts\\python.exe scripts\\rag_ask.py "в каких вакансиях нужен Playwright на TypeScript?"
    venv\\Scripts\\python.exe scripts\\rag_ask.py --search-only --mode hybrid "нагрузочное тестирование JMeter"
"""
import argparse
import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.answer import answer, format_sources  # noqa: E402
from rag.search import MODES, search  # noqa: E402


def _fmt_timings(timings: dict) -> str:
    return ", ".join(f"{k} {v:.2f} с" for k, v in timings.items() if isinstance(v, (int, float)))


async def main(question: str, mode: str, k: int, search_only: bool) -> int:
    if search_only:
        result = await search(question, k=k, mode=mode)
        for n, hit in enumerate(result.hits, start=1):
            score = f"rerank {hit.rerank_score:.3f}" if hit.rerank_score is not None else f"score {hit.score:.4f}"
            print(f"\n[{n}] {hit.metadata.get('title')} — {hit.metadata.get('company')} ({hit.section}, {score})")
            print(f"    {hit.metadata.get('url')}")
            print("    " + hit.content.replace("\n", " ")[:300])
        print(f"\nВремя: {_fmt_timings(result.timings)}")
        return 0

    result = await answer(question, k=k, mode=mode)
    print("\n" + result.text + "\n")
    print("Источники:")
    for line in format_sources(result.sources):
        print("  " + line)
    print(f"\nВремя: {_fmt_timings(result.timings)}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("question")
    parser.add_argument("--mode", choices=MODES, default="hybrid_rerank")
    parser.add_argument("-k", type=int, default=6, help="сколько фрагментов отдать модели")
    parser.add_argument("--search-only", action="store_true", help="только поиск, без ответа LLM")
    args = parser.parse_args()
    sys.exit(asyncio.run(main(args.question, args.mode, args.k, args.search_only)))
