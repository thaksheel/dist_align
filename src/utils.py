import hashlib
import json
import os
import re
import threading

import numpy as np


def clean_text(text) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", str(text or ""))).strip()


def norm_name(name) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name or "").lower())


def to_json(value):
    return json.loads(value) if isinstance(value, str) else value


def iter_questions(obj, block=None):
    if isinstance(obj, dict):
        block = obj.get("BlockName", block)
        if "QuestionText" in obj:
            yield block, obj
        for v in obj.values():
            if isinstance(v, (dict, list)):
                yield from iter_questions(v, block)
    elif isinstance(obj, list):
        for v in obj:
            yield from iter_questions(v, block)


def strip_reasoning(text: str) -> str:
    return (
        re.sub(r"<think>.*?</think>", "", text or "", flags=re.S)
        .split("</think>")[-1]
        .strip()
    )


def extract_json(text: str):
    t = re.sub(r"```(?:json)?", "", strip_reasoning(text))
    m = re.search(r"\{.*\}", t, re.S)
    try:
        return json.loads(m.group(0)) if m else None
    except json.JSONDecodeError:
        return None


def stable_hash(obj, n: int | None = None) -> str:
    h = hashlib.sha256(
        json.dumps(obj, sort_keys=True, default=str).encode()
    ).hexdigest()
    return h[:n] if n else h


def task_rng(seed: int, *parts) -> np.random.Generator:
    h = hashlib.md5("|".join(map(str, (seed,) + parts)).encode()).hexdigest()
    return np.random.default_rng(int(h[:8], 16))


class ResponseCache:
    def __init__(self, path: str | None):
        self.path = path
        self._data = {}
        self._lock = threading.Lock()
        if path:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            if os.path.exists(path):
                with open(path, encoding="utf-8") as f:
                    for line in f:
                        rec = json.loads(line)
                        self._data[rec["key"]] = rec

    def get(self, key: str) -> dict | None:
        return self._data.get(key)

    def set(self, key: str, record: dict) -> None:
        rec = {"key": key, **record}
        with self._lock:
            self._data[key] = rec
            if self.path:
                with open(self.path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec) + "\n")
