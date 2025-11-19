"""
Local React-driven feature probe UI server.

Usage:
    python viz_analysis/feature_probe_server.py --host 127.0.0.1 --port 8765
Then open http://127.0.0.1:8765/ to view the UI.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import numpy as np
from transformer_lens import HookedTransformer

from utils.run_picker import get_latest_run

STATIC_DIR = Path(__file__).resolve().parent


def _load_prompt_entries(run_path: Path) -> list[dict[str, Any]]:
    prompts_jsonl = run_path / "prompts.jsonl"
    prompt_txt = run_path / "prompt.txt"

    if prompts_jsonl.exists():
        entries: list[dict[str, Any]] = []
        with open(prompts_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entries.append(json.loads(line))
        if not entries:
            raise ValueError("prompts.jsonl was empty.")
        return entries

    if prompt_txt.exists():
        prompt_str = prompt_txt.read_text(encoding="utf-8").strip()
        if not prompt_str:
            raise ValueError("prompt.txt was empty.")
        return [
            {
                "row_id": -1,
                "seq_len": None,
                "prompt": prompt_str,
            }
        ]

    raise FileNotFoundError(
        f"Expected prompts.jsonl or prompt.txt in {run_path}, but neither was found."
    )


def _flatten_latents(
    latents: np.ndarray, seq_lens: list[int]
) -> tuple[np.ndarray, list[int], list[int]]:
    """Flatten [num_prompts, max_seq, hidden] -> [T, hidden] while tracking metadata."""
    if latents.ndim == 2:
        total_tokens = latents.shape[0]
        prompt_indices: list[int] = []
        token_positions: list[int] = []
        idx = 0
        for prompt_idx, seq_len in enumerate(seq_lens):
            for pos in range(seq_len):
                prompt_indices.append(prompt_idx)
                token_positions.append(pos)
                idx += 1
                if idx >= total_tokens:
                    break
            if idx >= total_tokens:
                break
        return latents, prompt_indices, token_positions

    if latents.ndim != 3:
        raise ValueError(f"Unsupported latent tensor shape: {latents.shape}")

    hidden_dim = latents.shape[-1]
    total_tokens = int(sum(seq_lens))
    flat = np.zeros((total_tokens, hidden_dim), dtype=latents.dtype)
    prompt_indices: list[int] = []
    token_positions: list[int] = []
    cursor = 0
    for prompt_idx, seq_len in enumerate(seq_lens):
        if seq_len == 0:
            continue
        flat[cursor : cursor + seq_len] = latents[prompt_idx, :seq_len]
        prompt_indices.extend([prompt_idx] * seq_len)
        token_positions.extend(list(range(seq_len)))
        cursor += seq_len

    return flat, prompt_indices, token_positions


def _flatten_tokens(tokens: np.ndarray, seq_lens: list[int]) -> np.ndarray:
    if tokens.ndim == 1:
        return tokens[: sum(seq_lens)]

    if tokens.ndim != 2:
        raise ValueError(f"Unsupported token tensor shape: {tokens.shape}")

    total_tokens = int(sum(seq_lens))
    flat = np.zeros(total_tokens, dtype=tokens.dtype)
    cursor = 0
    for prompt_idx, seq_len in enumerate(seq_lens):
        if seq_len == 0:
            continue
        flat[cursor : cursor + seq_len] = tokens[prompt_idx, :seq_len]
        cursor += seq_len
    return flat


@dataclass
class FeatureProbeData:
    run_path: Path
    latents: np.ndarray
    token_ids: np.ndarray
    prompt_entries: list[dict[str, Any]]
    seq_lens: list[int]
    prompt_indices: list[int]
    token_positions: list[int]
    token_strings: list[str]
    feature_means: np.ndarray
    feature_max: np.ndarray
    feature_frac_active: np.ndarray
    feature_stats_summary: dict[str, Any] | None

    @classmethod
    def load(cls, repo_root: Path) -> "FeatureProbeData":
        analysis_dir = repo_root / "analysis_data"
        run_path = get_latest_run(analysis_dir)
        print(f"[feature_probe_server] Using run: {run_path}")

        feature_stats_summary: dict[str, Any] | None = None
        summary_path = run_path / "feature_stats.json"
        if summary_path.exists():
            with open(summary_path, "r", encoding="utf-8") as f:
                feature_stats_summary = json.load(f)
            print(
                "[feature_probe_server] Loaded feature summary with metrics:",
                ", ".join(feature_stats_summary.get("metrics", {}).keys()),
            )

        latents = np.load(run_path / "latent_activations.npy")
        tokens = np.load(run_path / "tokens.npy")
        prompt_entries = _load_prompt_entries(run_path)
        seq_lens = [
            int(entry["seq_len"])
            if entry.get("seq_len") is not None
            else int(latents.shape[0] if latents.ndim == 2 else tokens.shape[1])
            for entry in prompt_entries
        ]

        lat_flat, prompt_indices, token_positions = _flatten_latents(latents, seq_lens)
        tok_flat = _flatten_tokens(tokens, seq_lens)

        model = HookedTransformer.from_pretrained("gpt2", center_writing_weights=False)
        tokenizer = model.tokenizer
        token_strings = [
            tokenizer.decode([int(token_id)], clean_up_tokenization_spaces=False)
            for token_id in tok_flat.tolist()
        ]

        feature_means = lat_flat.mean(axis=0)
        feature_max = lat_flat.max(axis=0)
        frac_active = (lat_flat > 0).mean(axis=0)

        return cls(
            run_path=run_path,
            latents=lat_flat,
            token_ids=tok_flat,
            prompt_entries=prompt_entries,
            seq_lens=seq_lens,
            prompt_indices=prompt_indices,
            token_positions=token_positions,
            token_strings=token_strings,
            feature_means=feature_means,
            feature_max=feature_max,
            feature_frac_active=frac_active,
            feature_stats_summary=feature_stats_summary,
        )

    @property
    def num_tokens(self) -> int:
        return int(self.latents.shape[0])

    @property
    def num_features(self) -> int:
        return int(self.latents.shape[1])

    @property
    def metrics_available(self) -> list[str]:
        if self.feature_stats_summary:
            return list(self.feature_stats_summary.get("metrics", {}).keys())
        return ["mean_activation", "max_activation", "fraction_active"]

    @property
    def metric_descriptions(self) -> dict[str, str]:
        if self.feature_stats_summary:
            return {
                name: data.get("description", "")
                for name, data in self.feature_stats_summary.get("metrics", {}).items()
            }
        return {
            "mean_activation": "Average activation across all tokens.",
            "max_activation": "Maximum activation observed for each feature.",
            "fraction_active": "Fraction of tokens where activation > 0.",
        }

    def get_top_features(
        self, limit: int = 50, metric_name: str = "mean_activation"
    ) -> list[dict[str, Any]]:
        metric_name = metric_name or "mean_activation"
        if self.feature_stats_summary:
            metrics_block = self.feature_stats_summary.get("metrics", {}).get(metric_name)
            if metrics_block is None:
                raise ValueError(
                    f"Metric '{metric_name}' is not present in feature_stats.json. "
                    f"Available metrics: {', '.join(self.metrics_available)}"
                )
            top_features = metrics_block.get("top_features", [])
            limit = max(1, min(limit, len(top_features)))
            return top_features[:limit]

        allowed_metrics = {"mean_activation", "max_activation", "fraction_active"}
        if metric_name not in allowed_metrics:
            raise ValueError(
                f"Metric '{metric_name}' is not supported without feature_stats.json. "
                f"Available metrics: {', '.join(sorted(allowed_metrics))}"
            )

        limit = max(1, min(limit, self.num_features))
        metric_arrays = {
            "mean_activation": self.feature_means,
            "max_activation": self.feature_max,
            "fraction_active": self.feature_frac_active,
        }
        values = metric_arrays[metric_name]
        top_idx = np.argsort(values)[-limit:][::-1]
        return [
            {
                "feature_id": int(idx),
                "value": float(values[idx]),
                "metrics": {
                    "mean_activation": float(self.feature_means[idx]),
                    "max_activation": float(self.feature_max[idx]),
                    "fraction_active": float(self.feature_frac_active[idx]),
                },
            }
            for idx in top_idx
        ]

    def get_feature_tokens(self, feature_id: int, top_k: int = 10) -> dict[str, Any]:
        if feature_id < 0 or feature_id >= self.num_features:
            raise ValueError(f"feature_id must be in [0, {self.num_features - 1}]")
        top_k = max(1, min(top_k, self.num_tokens))

        values = self.latents[:, feature_id]
        top_indices = np.argpartition(values, -top_k)[-top_k:]
        ordered = top_indices[np.argsort(values[top_indices])[::-1]]

        tokens_info: list[dict[str, Any]] = []
        for idx in ordered:
            prompt_idx = self.prompt_indices[idx] if idx < len(self.prompt_indices) else None
            prompt_meta = (
                self.prompt_entries[prompt_idx] if prompt_idx is not None else {"prompt": ""}
            )
            tokens_info.append(
                {
                    "token_index": int(idx),
                    "activation": float(values[idx]),
                    "token_str": self.token_strings[idx] if idx < len(self.token_strings) else "",
                    "prompt_index": int(prompt_idx) if prompt_idx is not None else None,
                    "prompt_row_id": prompt_meta.get("row_id"),
                    "prompt_snippet": (prompt_meta.get("prompt") or "")[:160],
                }
            )

        return {
            "feature_id": int(feature_id),
            "top_k": len(tokens_info),
            "tokens": tokens_info,
        }


class FeatureProbeRequestHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, data_store: FeatureProbeData, **kwargs):
        self.data_store = data_store
        super().__init__(*args, directory=str(STATIC_DIR), **kwargs)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
        print("[feature_probe_server]", format % args)

    def _send_json(self, payload: dict[str, Any], status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/api/features":
            params = parse_qs(parsed.query)
            limit = int(params.get("limit", ["50"])[0])
            metric_options = self.data_store.metrics_available or ["mean_activation"]
            metric = params.get("metric", [metric_options[0]])[0]
            try:
                features = self.data_store.get_top_features(limit, metric_name=metric)
                self._send_json(
                    {
                        "features": features,
                        "metric": metric,
                        "metrics_available": metric_options,
                        "metric_descriptions": self.data_store.metric_descriptions,
                        "num_features": self.data_store.num_features,
                        "num_tokens": self.data_store.num_tokens,
                    }
                )
            except Exception as exc:  # noqa: BLE001
                self._send_json({"error": str(exc)}, status=400)
            return

        if parsed.path == "/api/feature":
            params = parse_qs(parsed.query)
            feature_id = params.get("id")
            if feature_id is None:
                self._send_json({"error": "Missing 'id' query parameter"}, status=400)
                return
            top_k = int(params.get("top_k", ["10"])[0])
            try:
                data = self.data_store.get_feature_tokens(int(feature_id[0]), top_k=top_k)
                self._send_json(data)
            except Exception as exc:  # noqa: BLE001
                self._send_json({"error": str(exc)}, status=400)
            return

        if parsed.path in ("", "/"):
            self.path = "/feature_probe_frontend.html"
        return super().do_GET()


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the feature probe React UI.")
    parser.add_argument("--host", default="127.0.0.1", help="Hostname to bind (default 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8765, help="Port to bind (default 8765)")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    data_store = FeatureProbeData.load(repo_root)

    handler = lambda *handler_args, **handler_kwargs: FeatureProbeRequestHandler(  # noqa: E731
        *handler_args, data_store=data_store, **handler_kwargs
    )

    server = ThreadingHTTPServer((args.host, args.port), handler)
    url = f"http://{args.host}:{args.port}/"
    print(f"[feature_probe_server] Serving UI at {url}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[feature_probe_server] Shutting down...")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()

