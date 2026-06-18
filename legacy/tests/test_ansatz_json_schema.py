"""ARCH-02 schema regression: results/ansatz_comparison.json.

Validates the extended long-form ``rows[]`` + ``models[]`` schema (D-10-16)
extended with ``ansatz``/``depth``/``topology`` dims (Phase 13). Skips cleanly
if the aggregator output is absent so the suite stays green before the Wave-2
sweep+aggregation has run.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_JSON_PATH = _REPO_ROOT / "results" / "ansatz_comparison.json"

# The 9 required fields on every long-form row (D-10-16 + Phase-13 dims).
_REQUIRED_ROW_FIELDS = {
    "model_kind",
    "variant",
    "depth",
    "topology",
    "pipeline",
    "seed",
    "metric_name",
    "scale",
    "value",
}

# Expected (depth, topology, parameter_count) per variant.
_EXPECTED_VARIANT_SPEC = {
    "V1": (4, "range", 75),
    "V2": (8, "range", 135),
    "V3": (4, "linear", 75),
}


@pytest.fixture(scope="module")
def comparison_doc() -> dict:
    if not _JSON_PATH.exists():
        pytest.skip(
            f"{_JSON_PATH} not present yet — run "
            "`python -m run_ansatz_comparison` after the sweep."
        )
    return json.loads(_JSON_PATH.read_text())


def test_top_level_keys_present(comparison_doc: dict) -> None:
    for key in (
        "schema",
        "model_kinds",
        "ansatz_variants",
        "seeds",
        "rows",
        "data_equivalence",
    ):
        assert key in comparison_doc, f"missing top-level key: {key}"
    assert comparison_doc["model_kinds"] == ["quantum"]
    assert comparison_doc["seeds"] == [42, 43, 44, 45, 46]


def test_ansatz_variants_v1_v2_v3(comparison_doc: dict) -> None:
    variants = {v["variant"]: v for v in comparison_doc["ansatz_variants"]}
    assert set(variants) == {"V1", "V2", "V3"}, set(variants)
    for name, (depth, topo, pcount) in _EXPECTED_VARIANT_SPEC.items():
        spec = variants[name]
        assert spec["depth"] == depth, (name, spec)
        assert spec["topology"] == topo, (name, spec)
        assert spec["parameter_count"] == pcount, (name, spec)


def test_v1_source_is_reuse_no_recompute(comparison_doc: dict) -> None:
    variants = {v["variant"]: v for v in comparison_doc["ansatz_variants"]}
    v1_src = variants["V1"]["source"].lower()
    assert "no recompute" in v1_src, variants["V1"]["source"]
    assert "d-13-01" in v1_src, variants["V1"]["source"]
    # The by-construction data-equivalence note must mention no V1 recompute.
    eq = comparison_doc["data_equivalence"].lower()
    assert "no recompute" in eq and "d-13-01" in eq, eq


def test_rows_non_empty_and_well_formed(comparison_doc: dict) -> None:
    rows = comparison_doc["rows"]
    assert rows, "rows must be non-empty"
    for r in rows:
        missing = _REQUIRED_ROW_FIELDS - set(r)
        assert not missing, f"row missing fields {missing}: {r}"
        assert r["model_kind"] == "quantum"
        assert r["pipeline"] == "B"


def test_variant_set_is_v1_v2_v3(comparison_doc: dict) -> None:
    assert {r["variant"] for r in comparison_doc["rows"]} == {
        "V1",
        "V2",
        "V3",
    }


def test_scale_set_subset(comparison_doc: dict) -> None:
    assert {r["scale"] for r in comparison_doc["rows"]} <= {
        "log_return",
        "OD",
    }


def test_row_dims_match_variant_registry(comparison_doc: dict) -> None:
    """Every row's (depth, topology) matches its variant's registry spec."""
    for r in comparison_doc["rows"]:
        depth, topo, _ = _EXPECTED_VARIANT_SPEC[r["variant"]]
        assert r["depth"] == depth, r
        assert r["topology"] == topo, r
