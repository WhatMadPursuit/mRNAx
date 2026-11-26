# mRNAx v2

Industrial-grade, Clean-Room commercial-ready Agentic RNA Sequence Design Platform (rebuilt from scratch).

## Quick Verification & Replay

```bash
# Run unit & contract test suite
uv run pytest tests/unit -v

# Configure BYO-LLM keys safely (hidden terminal input into ~/.mrnax/secrets.env, chmod 600)
uv run mrnax auth set-key --provider openai
uv run mrnax auth set-key --provider deepseek
uv run mrnax auth set-key --provider kimi
uv run mrnax auth set-key --provider gemini

# Deterministic CLI replay from a versioned DesignIntent
uv run mrnax replay --intent-file artifacts/m1_egfp/protein_only_intent_v1.json --out-dir artifacts/m1_egfp/replay_protein_only
```
