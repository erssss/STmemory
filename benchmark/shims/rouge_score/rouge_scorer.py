from dataclasses import dataclass
from typing import Dict, List


@dataclass
class _Score:
    fmeasure: float = 0.0


class RougeScorer:
    def __init__(self, rouge_types: List[str], use_stemmer: bool = True):
        self.rouge_types = rouge_types
        self.use_stemmer = use_stemmer

    def score(self, target: str, prediction: str) -> Dict[str, _Score]:
        return {t: _Score(0.0) for t in self.rouge_types}

