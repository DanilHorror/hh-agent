"""Индексация вакансий для RAG: нарезка на фрагменты → векторы → таблица vacancy_chunks.

    venv\\Scripts\\python.exe scripts\\rag_index.py             только новые вакансии
    venv\\Scripts\\python.exe scripts\\rag_index.py --rebuild   переиндексировать всё
    venv\\Scripts\\python.exe scripts\\rag_index.py --dry-run   только показать нарезку, без модели и записи
"""
import argparse
import asyncio
import os
import sys
import time
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag import embeddings, store  # noqa: E402
from rag.chunking import chunk_vacancy  # noqa: E402

BATCH_VACANCIES = 32


async def main(rebuild: bool, dry_run: bool, limit: int | None) -> int:
    store.require_postgres()
    vacancies = await store.vacancies_to_index(rebuild=rebuild)
    if limit:
        vacancies = vacancies[:limit]
    if not vacancies:
        print("Новых вакансий для индексации нет.", await store.stats())
        return 0

    all_chunks = {v["vacancy_id"]: chunk_vacancy(v["vacancy_id"], v["title"] or "", v["company"] or "",
                                                 v["url"] or "", v["description"] or "", v["skills"])
                  for v in vacancies}
    total = sum(len(c) for c in all_chunks.values())
    sections = Counter(c.section for chunks in all_chunks.values() for c in chunks)
    lengths = sorted(len(c.content) for chunks in all_chunks.values() for c in chunks)
    print(f"Вакансий: {len(vacancies)}, фрагментов: {total} (в среднем {total / len(vacancies):.1f} на вакансию)")
    print(f"Длина фрагмента: медиана {lengths[len(lengths) // 2]} символов, максимум {lengths[-1]}")
    print("Разделы:", ", ".join(f"{k} {v}" for k, v in sections.most_common()))
    if dry_run:
        sample = next(iter(all_chunks.values()))
        for chunk in sample[:4]:
            print(f"\n--- [{chunk.section}] ---\n{chunk.content[:400]}")
        return 0

    embeddings.get_embedder()
    started = time.time()
    done = 0
    items = list(all_chunks.items())
    for start in range(0, len(items), BATCH_VACANCIES):
        batch = items[start:start + BATCH_VACANCIES]
        texts = [c.content for _, chunks in batch for c in chunks]
        vectors = embeddings.embed(texts, batch_size=16)
        offset = 0
        for vacancy_id, chunks in batch:
            await store.replace_chunks(vacancy_id, chunks, vectors[offset:offset + len(chunks)],
                                       embeddings.EMBED_MODEL)
            offset += len(chunks)
        done += len(batch)
        print(f"  {done}/{len(items)} вакансий, {time.time() - started:.0f} с")

    print(f"Готово за {time.time() - started:.0f} с.", await store.stats())
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rebuild", action="store_true", help="переиндексировать все вакансии")
    parser.add_argument("--dry-run", action="store_true", help="только нарезка и статистика")
    parser.add_argument("--limit", type=int, default=None, help="обработать не больше N вакансий")
    args = parser.parse_args()
    sys.exit(asyncio.run(main(args.rebuild, args.dry_run, args.limit)))
