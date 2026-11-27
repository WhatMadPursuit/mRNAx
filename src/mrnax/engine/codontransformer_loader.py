"""Native Apache-2.0 HuggingFace `CodonTransformer` (`adibvafa/CodonTransformer`, RC-005) Loader, Generator & Scorer.

Clean-Room & Safety Invariants (RC-005-codontransformer-v1.0.yaml):
- 3-Layer Commercial-Clean: Code Apache-2.0 (loaded natively via HuggingFace `transformers.BigBirdForMaskedLM` +
  `AutoTokenizer`, zero `CodonTransformer` monolithic pip bloat, zero GPL/AGPL) / Weights Apache-2.0 pinned to
  exact HF git revision SHA `9744dcc920d813066391fc828d7a590207f148e8` / Training Data Zenodo 12509224 CC-BY-4.0.
- Homo sapiens conditioning (`organism_id = 59`, `token_type_ids = 59`) with zero `[UNK]` assertion and `<= 2046` codon guard.
- Mandatory synonymous-only logit masking (`match_protein = True`) preserving 100% exact target amino-acid sequence.
- Mandatory deterministic `m1Ψ` `UUUC` and `U{4,}` slippery-run remediation (`remediate_m1psi_slippery_runs`) on `m1psi` /
  `pseudouridine` constructs (and must always be paired with a cleared structure/stability head at pipeline level).
"""

from __future__ import annotations

import re
import shutil
import tempfile
from pathlib import Path
from typing import Any
from weakref import WeakSet

import numpy as np

from mrnax.constraints.frameshift_m1psi import remediate_m1psi_slippery_runs
from mrnax.engine.checkpoint_identity import verify_release_directory_artifacts

APPROVED_CODONTRANSFORMER_HF_REPO = "adibvafa/CodonTransformer"
APPROVED_CODONTRANSFORMER_HF_REV = "9744dcc920d813066391fc828d7a590207f148e8"
HOMO_SAPIENS_ORG_ID = 59
MAX_CODONTRANSFORMER_CODONS = 2046
MAX_CODONTRANSFORMER_SENSE_CODONS = 2045

_LOADER_AUTHENTICATED_INSTANCES: WeakSet[CodonTransformerNativeModel] = WeakSet()


def is_loader_authenticated_codontransformer(obj: Any) -> bool:
    """True ONLY if `obj` was constructed by `CodonTransformerNativeModel.from_local_checkpoint(..., allow_sandbox_override=False)`."""
    return isinstance(obj, CodonTransformerNativeModel) and (obj in _LOADER_AUTHENTICATED_INSTANCES)


def _set_bigbird_attention_mode(model: Any, attn: str) -> None:
    """Switch BigBird attention mode across both live layer state (`self.bert.attention_type`) and static `config`."""
    bert = getattr(model, "bert", None)
    if bert is not None:
        if hasattr(bert, "set_attention_type") and getattr(bert, "attention_type", None) != attn:
            bert.set_attention_type(attn)
        if getattr(bert, "config", None) is not None:
            bert.config.attention_type = attn
    if getattr(model, "config", None) is not None:
        model.config.attention_type = attn

_DNA_CODON_TO_AA: dict[str, str] = {
    "TTT": "F", "TTC": "F", "TTA": "L", "TTG": "L", "CTT": "L", "CTC": "L", "CTA": "L", "CTG": "L",
    "ATT": "I", "ATC": "I", "ATA": "I", "ATG": "M", "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V",
    "TCT": "S", "TCC": "S", "TCA": "S", "TCG": "S", "AGT": "S", "AGC": "S",
    "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P", "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T",
    "GCT": "A", "GCC": "A", "GCA": "A", "GCG": "A", "TAT": "Y", "TAC": "Y",
    "CAT": "H", "CAC": "H", "CAA": "Q", "CAG": "Q", "AAT": "N", "AAC": "N",
    "AAA": "K", "AAG": "K", "GAT": "D", "GAC": "D", "GAA": "E", "GAG": "E",
    "TGT": "C", "TGC": "C", "TGG": "W",
    "CGT": "R", "CGC": "R", "CGA": "R", "CGG": "R", "AGA": "R", "AGG": "R",
    "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G",
    "TAA": "_", "TAG": "_", "TGA": "_",
}


def _clean_dna(seq: str) -> str:
    return "".join(seq.split()).upper().replace("U", "T")


def _translate_sense_rna(cds_rna: str) -> str:
    dna = _clean_dna(cds_rna)
    if len(dna) % 3 != 0:
        raise ValueError(f"CDS length {len(dna)} is not divisible by 3")
    out: list[str] = []
    for i in range(0, len(dna), 3):
        aa = _DNA_CODON_TO_AA.get(dna[i : i + 3])
        if aa is None or aa == "_":
            raise ValueError(f"Invalid or premature stop codon {dna[i : i + 3]!r} at codon {i // 3}")
        out.append(aa)
    return "".join(out)


class CodonTransformerNativeModel:
    """Native HuggingFace `BigBirdForMaskedLM` wrapper for `adibvafa/CodonTransformer` v1.0.0 (`RC-005`)."""

    def __init__(
        self,
        *,
        tokenizer: Any,
        model: Any,
        device: str = "cpu",
        verified_release_digests: bool = False,
    ) -> None:
        import torch  # type: ignore[import-not-found]

        self.tokenizer = tokenizer
        self.model = model
        self.device = device
        self.verified_release_digests = verified_release_digests

        vocab: dict[str, int] = dict(tokenizer.get_vocab())
        self._vocab = vocab
        self._stop_unk_tok = "__unk" if "__unk" in vocab else "__UNK"
        aa_to_ids: dict[str, list[int]] = {}
        id_to_codon: dict[int, str] = {}
        tok_to_id: dict[str, int] = {}
        for tok, tid in vocab.items():
            tl = tok.lower()
            if len(tl) == 5 and tl[1] == "_" and not tl.endswith("unk"):
                aa = "_" if tl.startswith("__") else tl[0].upper()
                cod = tl[2:].upper()
                if set(cod) <= set("ACGT"):
                    aa_to_ids.setdefault(aa, []).append(int(tid))
                    id_to_codon[int(tid)] = cod
                    tok_to_id[f"__{cod}" if aa == "_" else f"{aa}_{cod}"] = int(tid)

        if len(aa_to_ids) != 21 or len(tok_to_id) != 64:
            raise ValueError(
                f"CodonTransformer vocabulary failed canonical 21-AA / 64-codon validation "
                f"(got {len(aa_to_ids)} classes, {len(tok_to_id)} codons)"
            )
        self._aa_to_ids = aa_to_ids
        self._id_to_codon = id_to_codon
        self._tok_to_id = tok_to_id

        vocab_size = int(model.config.vocab_size)
        self._aa_mask_tensors: dict[str, Any] = {}
        for aa, tids in aa_to_ids.items():
            m = torch.full((vocab_size,), float("-inf"), device=device)
            m[sorted(tids)] = 0.0
            self._aa_mask_tensors[aa] = m

    @classmethod
    def from_local_checkpoint(
        cls,
        checkpoint_dir: Path,
        *,
        expected_safetensors_sha256: str | None = None,
        allow_sandbox_override: bool = False,
        device: str | None = None,
    ) -> CodonTransformerNativeModel:
        ckpt = Path(checkpoint_dir)
        if not (ckpt / "config.json").is_file() or not (ckpt / "model.safetensors").is_file():
            raise FileNotFoundError(
                f"CodonTransformer checkpoint directory '{ckpt}' missing required config.json or model.safetensors"
            )
        verify_release_directory_artifacts(
            ckpt,
            "codontransformer_cds_baseline",
            allow_sandbox_override=allow_sandbox_override,
            weights_sha256_override=expected_safetensors_sha256,
            weights_env_var="MRNAX_CODONTRANSFORMER_WEIGHTS_SHA256",
        )
        import torch  # type: ignore[import-not-found]
        from transformers import AutoTokenizer, BigBirdForMaskedLM  # type: ignore[import-not-found]

        resolved_dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        tok = AutoTokenizer.from_pretrained(str(ckpt), local_files_only=True)
        mdl = BigBirdForMaskedLM.from_pretrained(
            str(ckpt), local_files_only=True, use_safetensors=True
        )
        mdl.eval().to(resolved_dev)
        inst = cls(
            tokenizer=tok,
            model=mdl,
            device=resolved_dev,
            verified_release_digests=not allow_sandbox_override,
        )
        if not allow_sandbox_override:
            _LOADER_AUTHENTICATED_INSTANCES.add(inst)
        return inst

    @classmethod
    def from_huggingface(
        cls,
        repo_id: str = APPROVED_CODONTRANSFORMER_HF_REPO,
        *,
        revision: str = APPROVED_CODONTRANSFORMER_HF_REV,
        expected_safetensors_sha256: str | None = None,
        allow_sandbox_override: bool = False,
        device: str | None = None,
    ) -> CodonTransformerNativeModel:
        rev_clean = str(revision or "").strip().lower()
        if not re.fullmatch(r"[0-9a-f]{40}", rev_clean) or rev_clean != APPROVED_CODONTRANSFORMER_HF_REV:
            raise ValueError(
                f"CodonTransformer HuggingFace download requires exact pinned 40-hex revision "
                f"'{APPROVED_CODONTRANSFORMER_HF_REV}', got {revision!r}"
            )
        if allow_sandbox_override:
            raise ValueError(
                "CodonTransformerNativeModel.from_huggingface strictly enforces RC-005 release digests; "
                "sandbox overrides are forbidden on network downloads"
            )
        if repo_id != APPROVED_CODONTRANSFORMER_HF_REPO:
            raise ValueError(
                f"Unapproved CodonTransformer repository '{repo_id}'; expected '{APPROVED_CODONTRANSFORMER_HF_REPO}'"
            )
        from huggingface_hub import hf_hub_download  # type: ignore[import-not-found]

        files = (
            "config.json",
            "model.safetensors",
            "special_tokens_map.json",
            "tokenizer.json",
            "tokenizer_config.json",
        )
        downloaded = [Path(hf_hub_download(repo_id=repo_id, filename=fn, revision=rev_clean)) for fn in files]
        parents = {p.parent for p in downloaded}
        if len(parents) == 1 and all((downloaded[0].parent / fn).is_file() for fn in files):
            return cls.from_local_checkpoint(
                downloaded[0].parent,
                expected_safetensors_sha256=expected_safetensors_sha256,
                allow_sandbox_override=False,
                device=device,
            )
        with tempfile.TemporaryDirectory(prefix="mrnax_ct_hf_") as td:
            stage = Path(td)
            for fn, src in zip(files, downloaded, strict=True):
                shutil.copy2(src, stage / fn)
            return cls.from_local_checkpoint(
                stage,
                expected_safetensors_sha256=expected_safetensors_sha256,
                allow_sandbox_override=False,
                device=device,
            )

    def _encode_protein_unk_tokens(self, protein_seq: str) -> tuple[str, str]:
        raw_prot = str(protein_seq).strip().upper()
        sense_prot = raw_prot[:-1] if raw_prot.endswith("_") else raw_prot
        if not sense_prot:
            raise ValueError("Cannot generate CDS for empty protein sequence")
        if "_" in sense_prot:
            raise ValueError("Internal stop symbol '_' is not permitted inside target protein sequence")
        framed_prot = f"{sense_prot}_"
        if len(sense_prot) > MAX_CODONTRANSFORMER_SENSE_CODONS:
            raise ValueError(
                f"Protein length {len(sense_prot)} (+ stop token = {len(sense_prot) + 1}) exceeds "
                f"CodonTransformer 2,048-position window ({MAX_CODONTRANSFORMER_SENSE_CODONS} sense codons + 1 stop = {MAX_CODONTRANSFORMER_CODONS})"
            )
        toks: list[str] = []
        for aa in framed_prot:
            tok = self._stop_unk_tok if aa == "_" else f"{aa.lower()}_unk"
            if tok not in self._vocab:
                raise ValueError(f"Unsupported amino-acid symbol {aa!r} for CodonTransformer")
            toks.append(tok)
        return sense_prot, " ".join(toks)

    def score_cds_synonymous_loglik(self, cds_seqs: list[str]) -> np.ndarray:
        """Compute zero-shot Homo sapiens synonymous log-likelihood per codon (`RC-005`)."""
        import torch  # type: ignore[import-not-found]

        if not cds_seqs:
            return np.empty(0, dtype=np.float64)
        out = np.full(len(cds_seqs), np.nan, dtype=np.float64)
        for idx, raw_cds in enumerate(cds_seqs):
            dna = _clean_dna(raw_cds)
            if len(dna) < 3 or len(dna) % 3 != 0:
                continue
            cod_list = [dna[3 * i : 3 * i + 3] for i in range(len(dna) // 3)]
            # Normalize trailing in-frame tandem stop cassettes (e.g. UGAUAA / UGAUAAUAG) to a single stop codon
            trailing_stops = 0
            while cod_list and _DNA_CODON_TO_AA.get(cod_list[-1]) == "_":
                trailing_stops += 1
                cod_list.pop()
            if not cod_list:
                continue
            # Reject internal premature stop codons before sense end
            if any(_DNA_CODON_TO_AA.get(c) in {None, "_"} for c in cod_list):
                continue
            if len(cod_list) > MAX_CODONTRANSFORMER_SENSE_CODONS:
                raise ValueError(
                    f"CDS sense codon count {len(cod_list)} (+ mandatory stop token = {len(cod_list) + 1}) exceeds "
                    f"CodonTransformer safe limit ({MAX_CODONTRANSFORMER_SENSE_CODONS} sense codons / {MAX_CODONTRANSFORMER_CODONS} total)"
                )
            first_stop = dna[3 * len(cod_list) : 3 * (len(cod_list) + 1)] if trailing_stops > 0 else None
            eval_codons = [*cod_list, first_stop] if first_stop else cod_list
            n_cod = len(eval_codons)
            unk_toks: list[str] = []
            tgt_ids: list[int] = []
            aas: list[str] = []
            valid = True
            for cod in eval_codons:
                assert cod is not None
                aa = _DNA_CODON_TO_AA.get(cod)
                if aa is None:
                    valid = False
                    break
                tok_key = f"__{cod}" if aa == "_" else f"{aa}_{cod}"
                tok_unk = self._stop_unk_tok if aa == "_" else f"{aa.lower()}_unk"
                if tok_key not in self._tok_to_id or tok_unk not in self._vocab:
                    valid = False
                    break
                unk_toks.append(tok_unk)
                tgt_ids.append(self._tok_to_id[tok_key])
                aas.append(aa)
            if not valid:
                continue

            # Unified framing: always end with exactly one trailing stop `__unk` before `[SEP]`
            framed_unk_toks = unk_toks if aas[-1] == "_" else [*unk_toks, self._stop_unk_tok]
            n_framed = len(framed_unk_toks)
            attn = "original_full" if n_framed <= 1020 else "block_sparse"
            _set_bigbird_attention_mode(self.model, attn)
            enc = self.tokenizer(
                " ".join(framed_unk_toks),
                return_tensors="pt",
                pad_to_multiple_of=64 if attn == "block_sparse" else None,
            )
            unk_id = getattr(self.tokenizer, "unk_token_id", 0)
            if unk_id is not None and bool((enc["input_ids"] == unk_id).any().item()):
                raise RuntimeError("CodonTransformer tokenizer produced forbidden [UNK] token")
            enc["token_type_ids"] = torch.full_like(enc["input_ids"], HOMO_SAPIENS_ORG_ID)
            enc = {k: v.to(self.device) for k, v in enc.items()}
            with torch.no_grad():
                logits = self.model(**enc).logits[0, 1 : n_cod + 1, :].float()
            syn_masks = torch.stack([self._aa_mask_tensors[a] for a in aas], dim=0)
            syn_lp = torch.log_softmax(logits + syn_masks, dim=-1)
            tgt_t = torch.tensor(tgt_ids, device=self.device, dtype=torch.long)
            s_ll = syn_lp.gather(1, tgt_t[:, None]).squeeze(1)
            out[idx] = float(s_ll.mean().item())
        return out

    def generate_and_score_cds(
        self,
        protein_seq: str,
        *,
        uridine_chemistry: str = "m1psi",
        deterministic: bool = True,
        temperature: float = 0.2,
        seed: int = 42,
    ) -> tuple[str, float]:
        """Generate Homo sapiens synonymous CDS (`match_protein=True`) + mandatory m1Psi UUUC/U{4,} slippery post-filter."""
        import torch  # type: ignore[import-not-found]

        prot, unk_str = self._encode_protein_unk_tokens(protein_seq)
        n_cod = len(prot)
        n_framed = n_cod + 1
        attn = "original_full" if n_framed <= 1020 else "block_sparse"
        _set_bigbird_attention_mode(self.model, attn)
        enc = self.tokenizer(
            unk_str,
            return_tensors="pt",
            pad_to_multiple_of=64 if attn == "block_sparse" else None,
        )
        unk_id = getattr(self.tokenizer, "unk_token_id", 0)
        if unk_id is not None and bool((enc["input_ids"] == unk_id).any().item()):
            raise RuntimeError("CodonTransformer tokenizer produced forbidden [UNK] token")
        enc["token_type_ids"] = torch.full_like(enc["input_ids"], HOMO_SAPIENS_ORG_ID)
        enc = {k: v.to(self.device) for k, v in enc.items()}
        with torch.no_grad():
            logits = self.model(**enc).logits[0, 1 : n_cod + 1, :].float()

        # Mandatory synonymous-only logit mask (`match_protein=True`)
        syn_masks = torch.stack([self._aa_mask_tensors[aa] for aa in prot], dim=0)
        masked_logits = logits + syn_masks
        if deterministic:
            pred_ids = masked_logits.argmax(dim=-1).tolist()
        else:
            gen = torch.Generator(device=self.device).manual_seed(int(seed))
            probs = torch.softmax(masked_logits / max(float(temperature), 1e-4), dim=-1)
            pred_ids = torch.multinomial(probs, num_samples=1, generator=gen).squeeze(-1).tolist()

        cds_rna = "".join(self._id_to_codon[int(tid)] for tid in pred_ids).replace("T", "U")
        chem_norm = str(uridine_chemistry).strip().lower()
        if chem_norm in {"m1psi", "pseudouridine", "mpu"}:
            cds_rna, _ = remediate_m1psi_slippery_runs(cds_rna, include_uuuc=True)

        translated = _translate_sense_rna(cds_rna)
        if translated != prot:
            raise RuntimeError("CodonTransformer output violated exact target amino-acid translation lock")

        # Score the final post-filtered CDS sequence
        syn_ll = float(self.score_cds_synonymous_loglik([cds_rna])[0])
        return cds_rna, syn_ll
