"""
Borrowed from https://github.com/WujiangXu/AgenticMemory/blob/main/utils.py

@article{xu2025mem,
    title={A-mem: Agentic memory for llm agents},
    author={Xu, Wujiang and Liang, Zujie and Mei, Kai and Gao, Hang and Tan, Juntao
           and Zhang, Yongfeng},
    journal={arXiv preprint arXiv:2502.12110},
    year={2025}
}
"""

from typing import Dict, List

try:
    import nltk
    from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
except Exception:
    nltk = None
    SmoothingFunction = None
    sentence_bleu = None

if nltk is not None:
    try:
        nltk.data.find("tokenizers/punkt")
    except Exception:
        pass


def simple_tokenize(text):
    text = str(text)
    return text.lower().replace(".", " ").replace(",", " ").replace("!", " ").replace("?", " ").split()


def _ngram_counts(tokens: List[str], n: int) -> Dict[str, int]:
    if n <= 0:
        return {}
    counts: Dict[str, int] = {}
    for i in range(0, max(0, len(tokens) - n + 1)):
        g = " ".join(tokens[i : i + n])
        counts[g] = counts.get(g, 0) + 1
    return counts


def _clipped_precision(pred_tokens: List[str], ref_tokens: List[str], n: int) -> float:
    pred_counts = _ngram_counts(pred_tokens, n)
    ref_counts = _ngram_counts(ref_tokens, n)
    if not pred_counts:
        return 0.0
    clipped = 0
    total = 0
    for g, c in pred_counts.items():
        total += c
        clipped += min(c, ref_counts.get(g, 0))
    return (clipped / total) if total else 0.0


def calculate_bleu_scores(prediction: str, reference: str) -> Dict[str, float]:
    if nltk is not None and sentence_bleu is not None and SmoothingFunction is not None:
        try:
            pred_tokens = nltk.word_tokenize(prediction.lower())
            ref_tokens = [nltk.word_tokenize(reference.lower())]
        except Exception:
            pred_tokens = simple_tokenize(prediction)
            ref_tokens = [simple_tokenize(reference)]

        weights_list = [(1, 0, 0, 0), (0.5, 0.5, 0, 0), (0.33, 0.33, 0.33, 0), (0.25, 0.25, 0.25, 0.25)]
        smooth = SmoothingFunction().method1

        scores = {}
        for n, weights in enumerate(weights_list, start=1):
            try:
                score = sentence_bleu(ref_tokens, pred_tokens, weights=weights, smoothing_function=smooth)
            except Exception as e:
                print(f"Error calculating BLEU score: {e}")
                score = 0.0
            scores[f"bleu{n}"] = score

        return scores

    pred_tokens = simple_tokenize(prediction)
    ref_tokens = simple_tokenize(reference)
    return {
        "bleu1": _clipped_precision(pred_tokens, ref_tokens, 1),
        "bleu2": _clipped_precision(pred_tokens, ref_tokens, 2),
        "bleu3": _clipped_precision(pred_tokens, ref_tokens, 3),
        "bleu4": _clipped_precision(pred_tokens, ref_tokens, 4),
    }


def calculate_metrics(prediction: str, reference: str) -> Dict[str, float]:
    if not prediction or not reference:
        return {
            "exact_match": 0,
            "f1": 0.0,
            "bleu1": 0.0,
            "bleu2": 0.0,
            "bleu3": 0.0,
            "bleu4": 0.0,
        }

    prediction = str(prediction).strip()
    reference = str(reference).strip()

    exact_match = int(prediction.lower() == reference.lower())

    pred_tokens = set(simple_tokenize(prediction))
    ref_tokens = set(simple_tokenize(reference))
    common_tokens = pred_tokens & ref_tokens

    if not pred_tokens or not ref_tokens:
        f1 = 0.0
    else:
        precision = len(common_tokens) / len(pred_tokens)
        recall = len(common_tokens) / len(ref_tokens)
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    bleu_scores = calculate_bleu_scores(prediction, reference)

    metrics = {
        "exact_match": exact_match,
        "f1": f1,
        **bleu_scores,
    }

    return metrics
