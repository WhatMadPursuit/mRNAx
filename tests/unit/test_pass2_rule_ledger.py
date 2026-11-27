import json
import shutil
from pathlib import Path

import pytest

from mrnax.rules.tribunal import RuleRegistry


SEED = Path(__file__).resolve().parents[2] / "registry_seed"
RULES = "rules_commercial_v1.json"
LEDGER = "rule_observation_ledger_v1.json"


@pytest.fixture
def registry(tmp_path):
    for file in (RULES, LEDGER):
        shutil.copyfile(SEED / file, tmp_path / file)
    return RuleRegistry(tmp_path / RULES, track="commercial_clean")


def test_verified_rules_require_ledger(registry, tmp_path):
    assert registry.snapshot_version().startswith("rules_commercial_clean@")
    (tmp_path / LEDGER).unlink()
    with pytest.raises(ValueError, match="observation ledger"):
        registry.snapshot_version()
    with pytest.raises(ValueError, match="observation ledger"):
        registry.get_active_rules(chemistry="unmodified_u")


@pytest.mark.parametrize(("field", "value"), [
    ("measured_effect", 0), ("measured_effect", False),
    ("measured_effect", float("nan")), ("measured_effect", float("inf")),
    ("preregistered_threshold", -1), ("rule_id", "another_rule"),
    ("claim_id", "another_claim"), ("underlying_dataset_id", "different_dataset"),
    ("observed_direction", "positive"), ("author_lineage_group", None),
])
def test_unsubstantiated_observation_cannot_verify(registry, tmp_path, field, value):
    path = tmp_path / LEDGER
    ledger = json.loads(path.read_bytes())
    claim = next(c for c in ledger["claims"].values() if c["preregistered_direction"] == "negative")
    claim[field] = value
    path.write_text(json.dumps(ledger))
    with pytest.raises(ValueError):
        registry.get_active_rules(chemistry="unmodified_u")


def test_snapshot_binds_validated_ledger_and_isolated_fork(registry, tmp_path):
    path = tmp_path / LEDGER
    fork = registry.fork_isolated_copy(tmp_path / "sandbox" / "rules.json")
    original = registry.snapshot_version()
    assert RuleRegistry(fork.backing_path, track=fork.track).snapshot_version() == original
    with pytest.raises(FileExistsError):
        registry.fork_isolated_copy(registry.backing_path)
    ledger = json.loads(path.read_bytes())
    next(iter(ledger["claims"].values()))["measured_effect"] *= 1.1
    path.write_text(json.dumps(ledger))
    assert registry.snapshot_version() != original
    assert fork.snapshot_version() == original
    for claim in ledger["claims"].values():
        claim["measured_effect"] = 0
    fork.backing_path.with_suffix(".observations.json").write_text(json.dumps(ledger))
    with pytest.raises(ValueError, match="quantitative threshold"):
        fork.snapshot_version()
    assert registry.snapshot_version() != original
