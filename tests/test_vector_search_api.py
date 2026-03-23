import time
from datetime import datetime

import asyncio

import numpy as np

from memory_layers import DeepMemoryLayer, MemoryConfig, MemoryEntry
from plugin import SpatioTemporalMemoryPlugin


def _make_encoder() -> callable:
    mapping = {
        "query": np.array([1.0, 0.0], dtype=np.float32),
        "alpha": np.array([1.0, 0.0], dtype=np.float32),
        "beta": np.array([0.0, 1.0], dtype=np.float32),
    }

    def encode(texts):
        out = []
        for t in texts:
            key = str(t or "").strip()
            out.append(mapping.get(key, np.array([0.70710677, 0.70710677], dtype=np.float32)))
        return out

    return encode


def test_deep_vector_search_supports_multiple_distances():
    cfg = MemoryConfig(deep_persist_path=":memory:", deep_enable_vector_index=True, deep_vector_store_provider="numpy")
    layer = DeepMemoryLayer(cfg)
    layer.set_encoder(_make_encoder())

    now = datetime.now()
    layer.add(MemoryEntry(id="a", query="alpha", response="", timestamp=now, layer="deep", token_count=1))
    layer.add(MemoryEntry(id="b", query="beta", response="", timestamp=now, layer="deep", token_count=1))

    cos = layer.vector_search("query", top_k=2, distance="cosine")
    assert [r["id"] for r in cos[:2]] == ["a", "b"]

    euc = layer.vector_search("query", top_k=2, distance="euclid")
    assert [r["id"] for r in euc[:2]] == ["a", "b"]


def test_plugin_vector_search_api_shape():
    cfg = MemoryConfig(deep_persist_path=":memory:", deep_enable_vector_index=True, deep_vector_store_provider="numpy")
    plugin = SpatioTemporalMemoryPlugin(model_name="test", memory_config=cfg, enable_logging=False)
    plugin.layers["deep"].set_encoder(_make_encoder())

    now = datetime.now()
    plugin.layers["deep"].add(MemoryEntry(id="a", query="alpha", response="", timestamp=now, layer="deep", token_count=1))
    plugin.layers["deep"].add(MemoryEntry(id="b", query="beta", response="", timestamp=now, layer="deep", token_count=1))

    out = asyncio.run(plugin.vector_search("query", top_k=2, distance="cosine"))
    assert isinstance(out, dict)
    assert "results" in out
    assert [r["id"] for r in out["results"][:2]] == ["a", "b"]

    st = plugin.get_vector_store_stats()
    assert isinstance(st, dict)
    assert st.get("provider") == "numpy"


def test_vector_search_basic_latency_bound():
    cfg = MemoryConfig(deep_persist_path=":memory:", deep_enable_vector_index=True, deep_vector_store_provider="numpy")
    layer = DeepMemoryLayer(cfg)
    layer.set_encoder(_make_encoder())
    now = datetime.now()

    for i in range(200):
        q = "alpha" if i % 2 == 0 else "beta"
        layer.add(MemoryEntry(id=f"m{i}", query=q, response="", timestamp=now, layer="deep", token_count=1))

    t0 = time.perf_counter()
    out = layer.vector_search("query", top_k=10, distance="cosine")
    dt = time.perf_counter() - t0
    assert len(out) == 10
    assert dt < 3.0
