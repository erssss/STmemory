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

import statistics
from collections import defaultdict
from typing import Dict, List, Union

try:
    import nltk
    from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
    from nltk.translate.meteor_score import meteor_score
except Exception:
    nltk = None
    SmoothingFunction = None
    sentence_bleu = None
    meteor_score = None

try:
    from bert_score import score as bert_score
except Exception:
    bert_score = None

try:
    from rouge_score import rouge_scorer
except Exception:
    rouge_scorer = None

try:
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.util import pytorch_cos_sim
except Exception:
    SentenceTransformer = None
    pytorch_cos_sim = None

if nltk is not None:
    try:
        nltk.data.find("tokenizers/punkt")
    except Exception:
        pass

if SentenceTransformer is None:
    sentence_model = None
else:
    try:
        sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
    except Exception as e:
        print(f"Warning: Could not load SentenceTransformer model: {e}")
        sentence_model = None


def simple_tokenize(text):
    text = str(text)
    return text.lower().replace(".", " ").replace(",", " ").replace("!", " ").replace("?", " ").split()


def calculate_rouge_scores(prediction: str, reference: str) -> Dict[str, float]:
    if rouge_scorer is None:
        return {"rouge1_f": 0.0, "rouge2_f": 0.0, "rougeL_f": 0.0}
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = scorer.score(reference, prediction)
    return {
        "rouge1_f": scores["rouge1"].fmeasure,
        "rouge2_f": scores["rouge2"].fmeasure,
        "rougeL_f": scores["rougeL"].fmeasure,
    }


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


def calculate_bert_scores(prediction: str, reference: str) -> Dict[str, float]:
    if bert_score is None:
        return {"bert_precision": 0.0, "bert_recall": 0.0, "bert_f1": 0.0}
    try:
        P, R, F1 = bert_score([prediction], [reference], lang="en", verbose=False)
        return {"bert_precision": P.item(), "bert_recall": R.item(), "bert_f1": F1.item()}
    except Exception as e:
        print(f"Error calculating BERTScore: {e}")
        return {"bert_precision": 0.0, "bert_recall": 0.0, "bert_f1": 0.0}


def calculate_meteor_score(prediction: str, reference: str) -> float:
    if meteor_score is None:
        return 0.0
    try:
        return meteor_score([reference.split()], prediction.split())
    except Exception as e:
        print(f"Error calculating METEOR score: {e}")
        return 0.0


def calculate_sentence_similarity(prediction: str, reference: str) -> float:
    if sentence_model is None or pytorch_cos_sim is None:
        return 0.0
    try:
        embedding1 = sentence_model.encode([prediction], convert_to_tensor=True)
        embedding2 = sentence_model.encode([reference], convert_to_tensor=True)
        similarity = pytorch_cos_sim(embedding1, embedding2).item()
        return float(similarity)
    except Exception as e:
        print(f"Error calculating sentence similarity: {e}")
        return 0.0


def calculate_metrics(prediction: str, reference: str) -> Dict[str, float]:
    if not prediction or not reference:
        return {
            "exact_match": 0,
            "f1": 0.0,
            "rouge1_f": 0.0,
            "rouge2_f": 0.0,
            "rougeL_f": 0.0,
            "bleu1": 0.0,
            "bleu2": 0.0,
            "bleu3": 0.0,
            "bleu4": 0.0,
            "bert_f1": 0.0,
            "meteor": 0.0,
            "sbert_similarity": 0.0,
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


def aggregate_metrics(all_metrics: List[Dict[str, float]], all_categories: List[int]) -> Dict[str, Dict[str, Union[float, Dict[str, float]]]]:
    if not all_metrics:
        return {}

    aggregates = defaultdict(list)
    category_aggregates = defaultdict(lambda: defaultdict(list))

    for metrics, category in zip(all_metrics, all_categories):
        for metric_name, value in metrics.items():
            aggregates[metric_name].append(value)
            category_aggregates[category][metric_name].append(value)

    results = {"overall": {}}

    for metric_name, values in aggregates.items():
        results["overall"][metric_name] = {
            "mean": statistics.mean(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0.0,
            "median": statistics.median(values),
            "min": min(values),
            "max": max(values),
            "count": len(values),
        }

    for category in sorted(category_aggregates.keys()):
        results[f"category_{category}"] = {}
        for metric_name, values in category_aggregates[category].items():
            if values:
                results[f"category_{category}"][metric_name] = {
                    "mean": statistics.mean(values),
                    "std": statistics.stdev(values) if len(values) > 1 else 0.0,
                    "median": statistics.median(values),
                    "min": min(values),
                    "max": max(values),
                    "count": len(values),
                }

    return results
