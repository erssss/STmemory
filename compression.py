import re
from dataclasses import dataclass
from typing import List, Optional, Tuple


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    words = text.split()
    english_tokens = int(len(words) * 0.75)
    chinese_chars = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    punct = len(re.findall(r"[\.!?。！？]", text))
    return max(1, int(english_tokens + chinese_chars + punct * 0.5))


def _sentences(text: str) -> List[str]:
    if not text:
        return []
    parts = re.split(r"(?<=[\.!?。！？])\s+", text.strip())
    out: List[str] = []
    for p in parts:
        p = re.sub(r"\s+", " ", p).strip()
        if p:
            out.append(p)
    return out


def _extract_entities_heuristic(text: str, limit: int = 16) -> List[str]:
    if not text:
        return []
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9_\-]{2,}|[\u4e00-\u9fff]{2,}", text)
    seen = set()
    out: List[str] = []
    for t in tokens:
        k = t.lower()
        if k in seen:
            continue
        seen.add(k)
        out.append(t)
        if len(out) >= limit:
            break
    return out


@dataclass
class CompressionResult:
    raw_text: str
    compressed_text: str
    raw_tokens: int
    compressed_tokens: int
    ratio: float


class MemoryCompressor:
    def __init__(
        self,
        max_chars: int = 200,
        min_ratio: float = 0.5,
        target_ratio: float = 0.5,
        min_chars: int = 60,
    ):
        self.max_chars = int(max(40, max_chars))
        self.min_ratio = float(min(max(min_ratio, 0.1), 1.0))
        self.target_ratio = float(min(max(target_ratio, 0.1), 1.0))
        self.min_chars = int(max(40, min_chars))

    def compress_text(self, text: str) -> CompressionResult:
        return self._compress_with_limit(text, max_chars=self.max_chars)

    def _compress_with_limit(self, text: str, max_chars: int) -> CompressionResult:
        raw = (text or "").strip()
        raw_tokens = estimate_tokens(raw)
        if not raw:
            return CompressionResult("", "", 0, 0, 1.0)

        normalized = re.sub(r"\s+", " ", raw).strip()
        if len(normalized) <= max_chars:
            return CompressionResult(raw, normalized, raw_tokens, estimate_tokens(normalized), 1.0)

        sents = _sentences(normalized)
        if not sents:
            clipped = normalized[: max_chars].rstrip() + "..."
            return CompressionResult(raw, clipped, raw_tokens, estimate_tokens(clipped), estimate_tokens(clipped) / max(1, raw_tokens))

        entities = set(x.lower() for x in _extract_entities_heuristic(normalized))
        scored: List[Tuple[float, str]] = []
        for s in sents:
            s_norm = s.strip()
            if not s_norm:
                continue
            e = set(x.lower() for x in _extract_entities_heuristic(s_norm))
            ent_score = len(e.intersection(entities))
            digit_score = 2.0 if re.search(r"\d", s_norm) else 0.0
            date_score = 2.0 if re.search(r"\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b", s_norm) else 0.0
            len_pen = min(1.0, len(s_norm) / 120.0)
            score = ent_score + digit_score + date_score + len_pen
            scored.append((score, s_norm))

        scored.sort(key=lambda x: x[0], reverse=True)

        chosen: List[str] = []
        used = set()
        total_len = 0
        for _score, s in scored:
            key = s.lower()
            if key in used:
                continue
            extra = len(s) + (1 if chosen else 0)
            if total_len + extra > max_chars:
                continue
            chosen.append(s)
            used.add(key)
            total_len += extra
            if total_len >= max_chars:
                break

        if not chosen:
            clipped = normalized[: max_chars].rstrip() + "..."
            comp = clipped
        else:
            comp = " ".join(chosen).strip()
            if len(comp) < int(len(normalized) * self.min_ratio):
                comp = normalized[: max_chars].rstrip() + "..."

        comp_tokens = estimate_tokens(comp)
        ratio = comp_tokens / max(1, raw_tokens)
        return CompressionResult(raw, comp, raw_tokens, comp_tokens, ratio)

    def compress_to_target(self, text: str) -> CompressionResult:
        raw = (text or "").strip()
        if not raw:
            return CompressionResult("", "", 0, 0, 1.0)
        max_chars = self.max_chars
        best = self._compress_with_limit(raw, max_chars=max_chars)
        while best.raw_tokens > 0 and best.ratio > self.target_ratio and max_chars > self.min_chars:
            max_chars = max(self.min_chars, int(max_chars * 0.8))
            cand = self._compress_with_limit(raw, max_chars=max_chars)
            if cand.compressed_tokens <= best.compressed_tokens:
                best = cand
            else:
                break
        return best
