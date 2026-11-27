"""Exercise default runtime resources from either a source or wheel installation."""

from pathlib import Path

from fastapi.testclient import TestClient

from mrnax.cloud.cas_store import ExecutionCASStore
from mrnax.cloud.gateway import create_control_plane_app
from mrnax.constraints.registry import ScopedConstraintEngine
from mrnax.engine.profiles import resolve_default_cards_dir, resolve_default_seed_dir
from mrnax.engine.stage0_baseline import _load_bicodon_atlas
from mrnax.engine.surrogates import CommercialCleanSurrogateV1
from mrnax.rules.models import load_replacement_card_yaml


def test_default_runtime_resources_outside_checkout(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("MRNAX_SEED_DIR", raising=False)
    store = ExecutionCASStore(tmp_path / "control.sqlite")
    with TestClient(create_control_plane_app(store)) as client:
        presets = client.get("/v1/presets")
    assert presets.status_code == 200
    assert presets.json()["preset_reference_r"]["segments"]
    codons, pairs = _load_bicodon_atlas()
    assert len(codons) == 61
    assert len(pairs) == 3721

    seeds = resolve_default_seed_dir()
    ScopedConstraintEngine.from_seed_dir(seeds)
    heads = CommercialCleanSurrogateV1.from_seed_dir(seeds).loaded_heads
    assert len(set(heads)) == 4
    cards = [load_replacement_card_yaml(p) for p in resolve_default_cards_dir().glob("*.yaml")]
    assert {card.card_id for card in cards} == {
        "RC-001-VIENNARNA-2.8.0",
        "RC-002-STRUCTRFM-0.0.9-MIT",
        "RC-003-ORTHRUS-1.0-MIT",
        "RC-004-ETERNAFOLD-1.3.1-BSD3",
        "RC-005-CODONTRANSFORMER-1.0-APACHE2",
    }
