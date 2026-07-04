import argparse
from pathlib import Path

from common import ensure_dirs, load_config, write_jsonl


TOY_PARALLEL = [
    {
        "en": "The committee discussed the new education policy yesterday.",
        "zh": "委员会昨天讨论了新的教育政策。",
        "de": "Der Ausschuss diskutierte gestern die neue Bildungspolitik.",
        "ha": "Kwamitin ya tattauna sabon tsarin ilimi jiya.",
    },
    {
        "en": "Researchers found that clean water improves public health.",
        "zh": "研究人员发现清洁饮水可以改善公共健康。",
        "de": "Forscher fanden heraus, dass sauberes Wasser die öffentliche Gesundheit verbessert.",
        "ha": "Masu bincike sun gano cewa ruwa mai tsabta yana inganta lafiyar jama'a.",
    },
    {
        "en": "The city plans to build more affordable housing next year.",
        "zh": "这座城市计划明年建设更多经济适用房。",
        "de": "Die Stadt plant, im nächsten Jahr mehr bezahlbaren Wohnraum zu bauen.",
        "ha": "Birnin yana shirin gina gidaje masu araha da yawa a shekara mai zuwa.",
    },
    {
        "en": "Farmers are adapting to climate change with new irrigation methods.",
        "zh": "农民正在通过新的灌溉方法适应气候变化。",
        "de": "Landwirte passen sich mit neuen Bewässerungsmethoden an den Klimawandel an.",
        "ha": "Manoma suna daidaitawa da sauyin yanayi ta sabbin hanyoyin ban ruwa.",
    },
    {
        "en": "The hospital introduced a digital system for patient records.",
        "zh": "医院引入了用于病人记录的数字系统。",
        "de": "Das Krankenhaus führte ein digitales System für Patientenakten ein.",
        "ha": "Asibitin ya gabatar da tsarin dijital don bayanan marasa lafiya.",
    },
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--repeat", type=int, default=20)
    args = parser.parse_args()

    cfg = load_config(args.config)
    paths = ensure_dirs(cfg)

    rows = []
    for rep in range(args.repeat):
        for idx, item in enumerate(TOY_PARALLEL):
            for lang, text in item.items():
                rows.append(
                    {
                        "id": f"toy_{rep:03d}_{idx:03d}",
                        "lang": lang,
                        "flores_lang": "",
                        "text": text,
                    }
                )

    out_path = Path(paths["data"]) / "parallel_samples.jsonl"
    write_jsonl(str(out_path), rows)
    print(f"Wrote {len(rows)} toy rows to {out_path}")


if __name__ == "__main__":
    main()
