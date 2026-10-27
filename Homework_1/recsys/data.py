"""Воспроизводимая загрузка, очистка и отбор только по прошлому."""

from pathlib import Path
import gzip
import hashlib
import html
import json
import re
import urllib.request
import numpy as np
import pandas as pd

SOURCE = "https://nijianmo.github.io/amazon/index.html"
BASE = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_v2/"
FILES = {
    "Digital_Music.csv": BASE + "categoryFilesSmall/Digital_Music.csv",
    "meta_Digital_Music.json.gz": BASE + "metaFiles2/meta_Digital_Music.json.gz",
}
EXPECTED_SHA256 = {
    "Digital_Music.csv": "a73f1504ed58fdc3533a6817b4c89fc4f722eb46dc18f94a4566a4b171439c9c",
    "meta_Digital_Music.json.gz": "c044fdd1804a4ff619764f1db47efdc28585d186392e9daf7fd42e3b6bb41559",
}
TRAIN_END = int(pd.Timestamp("2015-01-01", tz="UTC").timestamp())
VALID_END = int(pd.Timestamp("2016-01-01", tz="UTC").timestamp())
N_ITEMS = 7000
MIN_TRAIN_EVENTS = 2


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_data(root="data/raw"):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "manifest.json"
    saved = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    manifest = {}
    for name, url in FILES.items():
        target = root / name
        if not target.exists():
            temporary = target.with_suffix(target.suffix + ".part")
            try:
                with urllib.request.urlopen(url, timeout=120) as response, open(temporary, "wb") as output:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        output.write(chunk)
                temporary.replace(target)
            finally:
                temporary.unlink(missing_ok=True)
        checksum = sha256(target)
        if checksum != EXPECTED_SHA256[name]:
            raise ValueError(f"Контрольная сумма не соответствует зафиксированной версии: {name}")
        if saved and name in saved and saved[name]["sha256"] != checksum:
            raise ValueError(f"Изменился исходный файл: {name}")
        manifest[name] = {"url": url, "bytes": target.stat().st_size, "sha256": checksum}
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def _text(value):
    return re.sub(r"<[^>]*(?:>|$)", "", html.unescape(str(value or ""))).strip()


def load_dataset(root="data/raw"):
    root = Path(root)
    raw = pd.read_csv(root / "Digital_Music.csv", header=None,
                      names=["item_id", "user_id", "rating", "timestamp"],
                      dtype={"item_id": str, "user_id": str})
    audit = {"raw_rows": len(raw), "raw_users": raw.user_id.nunique(),
             "raw_items": raw.item_id.nunique(), "missing": raw.isna().sum().to_dict(),
             "exact_duplicates": int(raw.duplicated().sum())}
    clean = raw.dropna().copy()
    valid = (clean.rating.isin([1, 2, 3, 4, 5]) & clean.timestamp.gt(0)
             & clean.user_id.ne("") & clean.item_id.ne(""))
    audit["invalid_rows"] = int((~valid).sum())
    clean = clean[valid].sort_values(["timestamp", "user_id", "item_id", "rating"], kind="stable")
    # Первое наблюдение пары: ни отбор, ни train не используют последующую переоценку.
    audit["repeated_pairs"] = int(clean.duplicated(["user_id", "item_id"]).sum())
    clean = clean.drop_duplicates(["user_id", "item_id"], keep="first")
    audit["clean_rows_before_sampling"] = len(clean)
    past = clean[clean.timestamp < TRAIN_END]
    popularity = past.groupby("item_id").size().rename("n").reset_index()
    popularity = popularity.sort_values(["n", "item_id"], ascending=[False, True], kind="stable")
    selected_items = set(popularity.head(N_ITEMS).item_id)
    counts = past[past.item_id.isin(selected_items)].groupby("user_id").size()
    selected_users = set(counts[counts >= MIN_TRAIN_EVENTS].index)
    data = clean[clean.item_id.isin(selected_items) & clean.user_id.isin(selected_users)].copy()
    # Пустые после отбора пользователей товары исключаются тоже по train.
    retained_items = set(data[data.timestamp < TRAIN_END].item_id)
    data = data[data.item_id.isin(retained_items)].copy()
    users, items = sorted(selected_users), sorted(retained_items)
    data["user_idx"] = data.user_id.map({value: idx for idx, value in enumerate(users)}).astype(int)
    data["item_idx"] = data.item_id.map({value: idx for idx, value in enumerate(items)}).astype(int)
    train = data[data.timestamp < TRAIN_END].reset_index(drop=True)
    validation = data[(data.timestamp >= TRAIN_END) & (data.timestamp < VALID_END)].reset_index(drop=True)
    test = data[data.timestamp >= VALID_END].reset_index(drop=True)
    train_validation = data[data.timestamp < VALID_END].reset_index(drop=True)
    shape = (len(users), len(items))
    assert 100_000 <= len(data) <= 2_000_000, len(data)
    assert min(shape) >= 1000
    assert train.timestamp.max() < TRAIN_END <= validation.timestamp.min()
    assert validation.timestamp.max() < VALID_END <= test.timestamp.min()
    assert not data.duplicated(["user_idx", "item_idx"]).any()

    metadata, raw_metadata, duplicate_metadata = {}, 0, 0
    with gzip.open(root / "meta_Digital_Music.json.gz", "rt", encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            raw_metadata += 1
            asin = record.get("asin")
            if asin in retained_items:
                if asin in metadata:
                    duplicate_metadata += 1
                    continue
                metadata[asin] = {"title": _text(record.get("title")),
                                  "artist": _text(record.get("brand")),
                                  "categories": " | ".join(_text(x) for x in record.get("category", []))}
    item_table = pd.DataFrame([{"item_id": item, "item_idx": idx,
                                **metadata.get(item, {"title": "", "artist": "", "categories": ""})}
                               for idx, item in enumerate(items)])
    audit.update({"rows": len(data), "users": shape[0], "items": shape[1],
                  "train_rows": len(train), "validation_rows": len(validation), "test_rows": len(test),
                  "sparsity": 1 - len(data) / np.prod(shape),
                  "raw_metadata_rows": raw_metadata, "duplicate_selected_metadata": duplicate_metadata,
                  "metadata_missing": {c: int(item_table[c].eq("").sum()) for c in ["title", "artist", "categories"]}})
    # Диагностика исключений в будущих периодах по полному очищенному источнику.
    for name, mask in [("validation", (clean.timestamp >= TRAIN_END) & (clean.timestamp < VALID_END)),
                       ("test", clean.timestamp >= VALID_END)]:
        future = clean[mask]
        audit[name + "_source_coverage"] = {
            "source_events": len(future),
            "unknown_selected_user_events": int((~future.user_id.isin(selected_users)).sum()),
            "outside_catalog_events": int((~future.item_id.isin(retained_items)).sum()),
            "retained_events": int((future.user_id.isin(selected_users) & future.item_id.isin(retained_items)).sum()),
        }
    return {"data": data, "train": train, "validation": validation, "test": test,
            "train_validation": train_validation, "shape": shape, "items": item_table,
            "audit": audit, "raw_years": pd.to_datetime(raw.timestamp, unit="s", utc=True).dt.year.value_counts().sort_index()}
