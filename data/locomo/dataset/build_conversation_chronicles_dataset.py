import argparse
import json
import os
import random
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import requests
from huggingface_hub import hf_hub_url


_SESSION_SPECS: List[Tuple[str, int]] = [
    ("first", 1),
    ("second", 2),
    ("third", 3),
    ("fourth", 4),
    ("fifth", 5),
]


def _safe_str(x: Any) -> str:
    return str(x or "").strip()


def _iter_jsonl_rows(url: str) -> Iterable[Dict[str, Any]]:
    with requests.get(url, stream=True, timeout=600) as r:
        r.raise_for_status()
        for raw in r.iter_lines(decode_unicode=True):
            if not raw:
                continue
            try:
                yield json.loads(raw)
            except Exception:
                continue


def _extract_session(
    row: Dict[str, Any],
    *,
    prefix: str,
    session_no: int,
    timestamp: str,
) -> List[Dict[str, Any]]:
    dialog = row.get(f"{prefix}_session_dialogue")
    speakers = row.get(f"{prefix}_session_speakers")
    if not isinstance(dialog, list) or not isinstance(speakers, list):
        return []
    if len(dialog) != len(speakers) or not dialog:
        return []

    turns: List[Dict[str, Any]] = []
    turn_no = 0
    for t, s in zip(dialog, speakers):
        text = _safe_str(t)
        speaker = _safe_str(s)
        if not text or not speaker:
            continue
        turn_no += 1
        turns.append(
            {
                "speaker": speaker,
                "dia_id": f"S{session_no}:{turn_no}",
                "text": text,
            }
        )
    return turns


def _pick_speaker_pair(speakers: Sequence[str]) -> Optional[Tuple[str, str]]:
    uniq = []
    for s in speakers:
        if s not in uniq:
            uniq.append(s)
        if len(uniq) > 2:
            return None
    if len(uniq) != 2:
        return None
    return uniq[0], uniq[1]


def _build_quote_qa(
    *,
    sessions: Sequence[Sequence[Dict[str, Any]]],
    speaker_a: str,
    speaker_b: str,
    n_qa: int,
    rng: random.Random,
) -> List[Dict[str, Any]]:
    candidates: List[Tuple[str, str]] = []
    for sess in sessions:
        for t in sess:
            dia_id = _safe_str(t.get("dia_id"))
            speaker = _safe_str(t.get("speaker"))
            text = _safe_str(t.get("text"))
            if not dia_id or not speaker or not text:
                continue
            excerpt = text.replace('"', "'").replace("\n", " ").strip()
            excerpt = " ".join(excerpt.split())
            if len(excerpt) > 96:
                excerpt = excerpt[:96].rstrip() + "…"
            candidates.append((dia_id, f'Who said: "{excerpt}"'))

    if not candidates:
        return []

    rng.shuffle(candidates)
    picked = candidates[: min(n_qa, len(candidates))]

    qas: List[Dict[str, Any]] = []
    for dia_id, question in picked:
        answer = speaker_a if dia_id.startswith("S") else speaker_a
        qas.append(
            {
                "question": question,
                "answer": None,
                "evidence": [dia_id],
                "category": 1,
                "adversarial_answer": None,
            }
        )

    evidence_to_speaker: Dict[str, str] = {}
    for sess in sessions:
        for t in sess:
            evidence_to_speaker[_safe_str(t.get("dia_id"))] = _safe_str(t.get("speaker"))

    for qa in qas:
        ev = (qa.get("evidence") or [None])[0]
        spk = evidence_to_speaker.get(_safe_str(ev)) or speaker_a
        qa["answer"] = spk
        qa["adversarial_answer"] = speaker_b if spk == speaker_a else speaker_a

    return qas


def build_dataset(
    *,
    split: str,
    max_conversations: int,
    n_qa_per_conversation: int,
    seed: int,
    max_rows_to_scan: int,
) -> List[Dict[str, Any]]:
    if split not in {"train", "valid", "test"}:
        raise ValueError("split must be one of: train, valid, test")

    url = hf_hub_url(
        repo_id="jihyoung/ConversationChronicles",
        repo_type="dataset",
        filename=f"{split}.jsonl",
    )

    rng = random.Random(seed)
    items: List[Dict[str, Any]] = []

    scanned = 0
    for row in _iter_jsonl_rows(url):
        scanned += 1
        if max_rows_to_scan > 0 and scanned > max_rows_to_scan:
            break

        time_intervals = row.get("time_interval")
        if not isinstance(time_intervals, list) or not time_intervals:
            continue

        sessions: List[List[Dict[str, Any]]] = []
        all_speakers: List[str] = []
        conversation: Dict[str, Any] = {}

        for prefix, session_no in _SESSION_SPECS:
            timestamp = _safe_str(time_intervals[session_no - 1]) if (session_no - 1) < len(time_intervals) else ""
            turns = _extract_session(
                row,
                prefix=prefix,
                session_no=session_no,
                timestamp=timestamp,
            )
            if not turns:
                continue
            sessions.append(turns)
            conversation[f"session_{session_no}_date_time"] = timestamp or "unknown"
            conversation[f"session_{session_no}"] = turns
            all_speakers.extend([_safe_str(t.get("speaker")) for t in turns if _safe_str(t.get("speaker"))])

        if not sessions:
            continue

        pair = _pick_speaker_pair(all_speakers)
        if pair is None:
            continue
        speaker_a, speaker_b = pair

        speaker_set_ok = True
        for s in all_speakers:
            if s not in (speaker_a, speaker_b):
                speaker_set_ok = False
                break
        if not speaker_set_ok:
            continue

        conversation["speaker_a"] = speaker_a
        conversation["speaker_b"] = speaker_b

        qa = _build_quote_qa(
            sessions=sessions,
            speaker_a=speaker_a,
            speaker_b=speaker_b,
            n_qa=n_qa_per_conversation,
            rng=rng,
        )
        if not qa:
            continue

        items.append({"qa": qa, "conversation": conversation})
        if len(items) >= max_conversations:
            break

    return items


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--max_conversations", type=int, default=10)
    parser.add_argument("--n_qa_per_conversation", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_rows_to_scan", type=int, default=5000)
    args = parser.parse_args()

    items = build_dataset(
        split=args.split,
        max_conversations=max(1, args.max_conversations),
        n_qa_per_conversation=max(1, args.n_qa_per_conversation),
        seed=args.seed,
        max_rows_to_scan=max(0, args.max_rows_to_scan),
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)

    print(f"Wrote {len(items)} conversations to {args.output}")


if __name__ == "__main__":
    main()
