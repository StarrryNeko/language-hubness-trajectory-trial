import argparse
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

from common import ensure_dirs, load_config, set_seed, write_jsonl


def pick_sentence(row):
    for key in ["sentence", "text", "translation"]:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    raise KeyError(f"Cannot find sentence field in row keys: {list(row.keys())}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 42))
    paths = ensure_dirs(cfg)

    dataset_cfg = cfg["dataset"]
    split = dataset_cfg.get("split", "dev")
    sample_size = int(dataset_cfg.get("sample_size_per_language", 200))
    languages = dataset_cfg["languages"]

    rows = []
    for short_lang, flores_lang in tqdm(languages.items(), desc="Loading FLORES languages"):
        ds = load_dataset("facebook/flores", flores_lang, split=split)
        limit = min(sample_size, len(ds))
        for idx in range(limit):
            row = ds[idx]
            rows.append(
                {
                    "id": f"{idx:05d}",
                    "lang": short_lang,
                    "flores_lang": flores_lang,
                    "text": pick_sentence(row),
                }
            )

    out_path = Path(paths["data"]) / "parallel_samples.jsonl"
    write_jsonl(str(out_path), rows)
    print(f"Wrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()

