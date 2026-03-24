import argparse
import json
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple


def _preprocess(text: str) -> str:
    text = str(text or "")
    text = text.replace("。", ".").replace("’", "'")
    text = text.replace(" ,", ",").replace(" .", ".")
    text = text.replace(" ?", "?").replace(" !", "!").replace(" ;", ";")
    return text.strip()


def _iter_dialog_turns(raw_dialog: Any) -> List[str]:
    if raw_dialog is None:
        return []
    if isinstance(raw_dialog, (list, tuple)):
        return [_preprocess(x) for x in raw_dialog if _preprocess(x)]
    if isinstance(raw_dialog, str):
        t = _preprocess(raw_dialog)
        return [t] if t else []
    return []


def _pick_qa_indices(n_turns: int, n_qa: int, rng: random.Random) -> List[int]:
    if n_turns < 2:
        return []
    candidates = list(range(0, n_turns - 1))
    rng.shuffle(candidates)
    return sorted(candidates[: min(n_qa, len(candidates))])


def _build_locomo_style_item(
    dialog_turns: Sequence[str],
    n_qa_per_dialog: int,
    rng: random.Random,
    max_chars_per_turn: Optional[int],
) -> Dict[str, Any]:
    turns = [t[:max_chars_per_turn] if max_chars_per_turn else t for t in dialog_turns]
    session = []
    for i, t in enumerate(turns):
        speaker = "A" if (i % 2 == 0) else "B"
        session.append({"speaker": speaker, "dia_id": f"D1:{i + 1}", "text": t})

    qa = []
    for i in _pick_qa_indices(len(turns), n_qa_per_dialog, rng):
        q_turn = turns[i]
        a_turn = turns[i + 1]
        a_speaker = "A" if ((i + 1) % 2 == 0) else "B"
        qa.append(
            {
                "question": f'What did {a_speaker} reply to: "{q_turn}"',
                "answer": a_turn,
                "evidence": [f"D1:{i + 2}"],
                "category": 1,
            }
        )

    return {
        "qa": qa,
        "conversation": {
            "speaker_a": "A",
            "speaker_b": "B",
            "session_1_date_time": "unknown",
            "session_1": session,
        },
    }


def build_dataset(
    split: str,
    max_conversations: Optional[int],
    min_turns: int,
    n_qa_per_dialog: int,
    seed: int,
    max_chars_per_turn: Optional[int],
) -> List[Dict[str, Any]]:
    try:
        import datasets  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "Missing dependency: datasets. Install with `pip install datasets`."
        ) from e

    ds = datasets.load_dataset("allenai/soda")
    if split not in ds:
        raise ValueError(f"Split not found in allenai/soda: {split}. Available: {list(ds.keys())}")

    rng = random.Random(seed)
    items: List[Dict[str, Any]] = []

    for row in ds[split]:
        dialog_turns = _iter_dialog_turns(row.get("dialogue"))
        if len(dialog_turns) < min_turns:
            continue
        item = _build_locomo_style_item(
            dialog_turns=dialog_turns,
            n_qa_per_dialog=n_qa_per_dialog,
            rng=rng,
            max_chars_per_turn=max_chars_per_turn,
        )
        if not item["qa"]:
            continue
        items.append(item)
        if max_conversations and len(items) >= max_conversations:
            break

    return items


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--split", type=str, default="validation")
    parser.add_argument("--max_conversations", type=int, default=999999)
    parser.add_argument("--min_turns", type=int, default=4)
    parser.add_argument("--n_qa_per_dialog", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_chars_per_turn", type=int, default=999999)
    args = parser.parse_args()

    items = build_dataset(
        split=args.split,
        max_conversations=args.max_conversations if args.max_conversations > 0 else None,
        min_turns=max(2, args.min_turns),
        n_qa_per_dialog=max(1, args.n_qa_per_dialog),
        seed=args.seed,
        max_chars_per_turn=args.max_chars_per_turn if args.max_chars_per_turn > 0 else None,
    )

    with open(args.output, "w") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)

    print(f"Wrote {len(items)} conversations to {args.output}")


if __name__ == "__main__":
    main()
