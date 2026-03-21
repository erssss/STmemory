import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional, Tuple


@dataclass(frozen=True)
class TimeRange:
    start: Optional[datetime]
    end: Optional[datetime]


def _start_of_day(dt: datetime) -> datetime:
    return dt.replace(hour=0, minute=0, second=0, microsecond=0)


def _end_of_day(dt: datetime) -> datetime:
    return dt.replace(hour=23, minute=59, second=59, microsecond=999999)


def parse_time_range_from_query(query: str, now: Optional[datetime] = None) -> TimeRange:
    q = (query or "").strip()
    if not q:
        return TimeRange(None, None)
    now = now or datetime.now()

    if "刚才" in q or "刚刚" in q:
        return TimeRange(now - timedelta(minutes=10), now)
    if "最近" in q or "近期" in q:
        return TimeRange(now - timedelta(days=7), now)
    if "今天" in q:
        d = _start_of_day(now)
        return TimeRange(d, _end_of_day(d))
    if "昨天" in q:
        d = _start_of_day(now - timedelta(days=1))
        return TimeRange(d, _end_of_day(d))
    if "前天" in q:
        d = _start_of_day(now - timedelta(days=2))
        return TimeRange(d, _end_of_day(d))
    if "上周" in q:
        start = _start_of_day(now - timedelta(days=7))
        return TimeRange(start, now)
    if "上个月" in q:
        start = _start_of_day(now - timedelta(days=30))
        return TimeRange(start, now)
    if "去年" in q:
        start = now.replace(year=now.year - 1, month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
        end = now.replace(year=now.year - 1, month=12, day=31, hour=23, minute=59, second=59, microsecond=999999)
        return TimeRange(start, end)

    m = re.search(r"\b(\d{4})[-/](\d{1,2})[-/](\d{1,2})\b", q)
    if m:
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        try:
            day = datetime(y, mo, d)
            return TimeRange(_start_of_day(day), _end_of_day(day))
        except Exception:
            pass

    m2 = re.search(r"\b(\d{4})\b", q)
    if m2 and any(k in q for k in ["年", "in ", "during "]):
        y = int(m2.group(1))
        try:
            start = datetime(y, 1, 1)
            end = datetime(y, 12, 31, 23, 59, 59, 999999)
            return TimeRange(start, end)
        except Exception:
            pass

    month_map = {
        "jan": 1,
        "january": 1,
        "feb": 2,
        "february": 2,
        "mar": 3,
        "march": 3,
        "apr": 4,
        "april": 4,
        "may": 5,
        "jun": 6,
        "june": 6,
        "jul": 7,
        "july": 7,
        "aug": 8,
        "august": 8,
        "sep": 9,
        "sept": 9,
        "september": 9,
        "oct": 10,
        "october": 10,
        "nov": 11,
        "november": 11,
        "dec": 12,
        "december": 12,
    }

    m3 = re.search(r"\b([A-Za-z]{3,9})\s+(\d{1,2})\s*,?\s*(\d{4})\b", q)
    if m3:
        mon = month_map.get(m3.group(1).lower())
        if mon:
            try:
                day = datetime(int(m3.group(3)), int(mon), int(m3.group(2)))
                return TimeRange(_start_of_day(day), _end_of_day(day))
            except Exception:
                pass

    m4 = re.search(r"\b(\d{1,2})\s+([A-Za-z]{3,9})\s*,?\s*(\d{4})\b", q)
    if m4:
        mon = month_map.get(m4.group(2).lower())
        if mon:
            try:
                day = datetime(int(m4.group(3)), int(mon), int(m4.group(1)))
                return TimeRange(_start_of_day(day), _end_of_day(day))
            except Exception:
                pass

    return TimeRange(None, None)


def temporal_match_score(entry_time: datetime, time_range: TimeRange, now: Optional[datetime] = None) -> float:
    now = now or datetime.now()
    if time_range.start is None and time_range.end is None:
        return 0.0
    if time_range.start is not None and entry_time < time_range.start:
        return 0.0
    if time_range.end is not None and entry_time > time_range.end:
        return 0.0

    start = time_range.start or (now - timedelta(days=3650))
    end = time_range.end or now
    span = max(1.0, (end - start).total_seconds())
    mid = start + timedelta(seconds=span / 2.0)
    dist = abs((entry_time - mid).total_seconds())
    return max(0.0, 1.0 - (dist / (span / 2.0)))
