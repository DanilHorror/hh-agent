"""Эталонный набор: выборка по корзинам оценки, разметка, API разметки."""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quality import golden  # noqa: E402


def rows(scores):
    return [{"vacancy_id": str(i), "title": f"V{i}", "company": "C", "url": f"u{i}",
             "description": "Текст вакансии", "score": s} for i, s in enumerate(scores)]


def test_sample_takes_every_score_bucket():
    data = rows([10] * 30 + [50] * 30 + [70] * 30 + [90] * 30)
    sample = golden.build_sample(data, n=40, seed=1)
    scores = {r["vacancy_id"]: r["score"] for r in data}
    taken = [scores[i["vacancy_id"]] for i in sample]
    assert len(sample) == 40
    assert taken.count(10) == taken.count(50) == taken.count(70) == taken.count(90) == 10
    assert all(i["label"] is None and "score" not in i for i in sample)   # оценку модели не показываем


def test_sample_fills_from_unscored_and_skips_empty_texts():
    data = rows([90] * 3 + [None] * 20)
    data[0]["description"] = "  "
    sample = golden.build_sample(data, n=10)
    assert len(sample) == 10 and "0" not in {i["vacancy_id"] for i in sample}


def test_sample_is_reproducible():
    data = rows(list(range(100)))
    assert golden.build_sample(data, n=20, seed=7) == golden.build_sample(data, n=20, seed=7)


def test_label_progress_and_roundtrip(tmp_path):
    path = str(tmp_path / "golden.json")
    items = golden.build_sample(rows([10, 50, 70, 90]), n=4)
    golden.set_label(items, items[0]["vacancy_id"], "fit", "нравится стек")
    golden.set_label(items, items[1]["vacancy_id"], "not_fit")
    golden.save(items, path)
    loaded = golden.load(path)
    assert golden.progress(loaded) == {"total": 4, "labeled": 2, "fit": 1, "not_fit": 1, "unsure": 0}
    assert loaded[0]["note"] == "нравится стек"
    with pytest.raises(ValueError):
        golden.set_label(loaded, loaded[0]["vacancy_id"], "maybe")
    with pytest.raises(KeyError):
        golden.set_label(loaded, "nope", "fit")


def test_api_labeling(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from api import main as api

    async def no_init():
        return None

    monkeypatch.setattr(api.database, "init_db", no_init)
    path = str(tmp_path / "golden.json")
    monkeypatch.setattr(golden, "GOLDEN_PATH", path)

    with TestClient(api.app) as client:
        assert client.get("/eval/golden").status_code == 404          # набор ещё не собран
        golden.save(golden.build_sample(rows([10, 90]), n=2), path)
        body = client.get("/eval/golden").json()
        assert body["total"] == 2 and body["labeled"] == 0
        vid = body["items"][0]["vacancy_id"]
        assert client.put(f"/eval/golden/{vid}", json={"label": "fit"}).status_code == 200
        assert client.get("/eval/golden").json()["fit"] == 1
        assert client.put(f"/eval/golden/{vid}", json={"label": "wrong"}).status_code == 422
        assert client.put("/eval/golden/nope", json={"label": "fit"}).status_code == 404
        assert client.put(f"/eval/golden/{vid}", json={"label": None}).json()["label"] is None
