from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class _Scalar:
    value: float

    def item(self) -> float:
        return float(self.value)


def score(cands: List[str], refs: List[str], lang: str = "en", verbose: bool = False) -> Tuple[_Scalar, _Scalar, _Scalar]:
    if not cands or not refs:
        return _Scalar(0.0), _Scalar(0.0), _Scalar(0.0)
    c = str(cands[0]).strip().lower()
    r = str(refs[0]).strip().lower()
    v = 1.0 if c == r else 0.0
    return _Scalar(v), _Scalar(v), _Scalar(v)

