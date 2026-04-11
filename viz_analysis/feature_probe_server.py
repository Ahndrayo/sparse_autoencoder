"""
Local React-driven feature probe UI server.

Usage:
    python viz_analysis/feature_probe_server.py --host 127.0.0.1 --port 8765
    
    # Load a specific run:
    python viz_analysis/feature_probe_server.py --run-id 12
    
Then open http://127.0.0.1:8765/ to view the UI.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

# Add project root to sys.path so we can import from top-level utils
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from utils.run_picker import get_latest_run, get_run_by_id

STATIC_DIR = Path(__file__).resolve().parent
HIST_BIN_WIDTH = 0.2


@dataclass
class FeatureProbeData:
    run_path: Path
    run_metadata: dict[str, Any]
    feature_stats: dict[str, Any]
    feature_stats_baseline: dict[str, Any] | None
    feature_tokens: dict[str, list[dict[str, Any]]]
    feature_tokens_baseline: dict[str, list[dict[str, Any]]]
    mean_act_squared: list[float]
    mean_act_squared_baseline: list[float]
    feature_metrics_map: dict[int, dict[str, float]]
    feature_metrics_map_baseline: dict[int, dict[str, float]]
    headline_features: list[dict[str, Any]]
    baseline_headline_features: list[dict[str, Any]] | None
    avg_cns_by_feature: dict[int, float]
    interpretability_results: list[dict[str, Any]]
    interpretability_by_feature: dict[str, dict[str, Any]]

    @classmethod
    def load(cls, repo_root: Path, run_id: int | None = None) -> "FeatureProbeData":
        analysis_dir = repo_root / "analysis_data"
        if run_id is not None:
            run_path = get_run_by_id(analysis_dir, run_id)
        else:
            run_path = get_latest_run(analysis_dir)
        print(f"[feature_probe_server] Using run: {run_path}")

        metadata_path = run_path / "metadata.json"
        run_metadata: dict[str, Any] = {}
        if metadata_path.exists():
            with open(metadata_path, "r", encoding="utf-8") as f:
                run_metadata = json.load(f)

        summary_path = run_path / "feature_stats.json"
        if not summary_path.exists():
            raise FileNotFoundError(
                f"Expected feature_stats.json in {run_path}, please rerun the inference cell in Sentiment_Classification.ipynb."
            )
        with open(summary_path, "r", encoding="utf-8") as f:
            feature_stats = json.load(f)

        tokens_path = run_path / "feature_tokens.json"
        feature_tokens: dict[str, list[dict[str, Any]]] = {}
        if tokens_path.exists():
            with open(tokens_path, "r", encoding="utf-8") as f:
                token_payload = json.load(f)
            feature_tokens = token_payload.get("features", {})
        else:
            print(
                "[feature_probe_server] Warning: feature_tokens.json not found; "
                "token lookups will be unavailable."
            )

        tokens_baseline_path = run_path / "feature_tokens_baseline.json"
        feature_tokens_baseline: dict[str, list[dict[str, Any]]] = {}
        if tokens_baseline_path.exists():
            with open(tokens_baseline_path, "r", encoding="utf-8") as f:
                token_payload_b = json.load(f)
            feature_tokens_baseline = token_payload_b.get("features", {})
        else:
            print(
                "[feature_probe_server] Info: feature_tokens_baseline.json not found; "
                "baseline token examples unavailable (re-run notebook save cell)."
            )

        stats_baseline_path = run_path / "feature_stats_baseline.json"
        feature_stats_baseline: dict[str, Any] | None = None
        if stats_baseline_path.exists():
            with open(stats_baseline_path, "r", encoding="utf-8") as f:
                feature_stats_baseline = json.load(f)

        def build_metrics_map(fs: dict[str, Any]) -> dict[int, dict[str, float]]:
            m: dict[int, dict[str, float]] = {}
            num_f = int(fs.get("num_features", 0))
            ma = fs.get("mean_activation")
            xa = fs.get("max_activation")
            fa = fs.get("fraction_active")
            if (
                num_f > 0
                and isinstance(ma, list)
                and isinstance(xa, list)
                and isinstance(fa, list)
                and len(ma) == num_f
                and len(xa) == num_f
                and len(fa) == num_f
            ):
                return {
                    i: {
                        "mean_activation": float(ma[i]),
                        "max_activation": float(xa[i]),
                        "fraction_active": float(fa[i]),
                    }
                    for i in range(num_f)
                }
            for metric_block in fs.get("metrics", {}).values():
                for entry in metric_block.get("top_features", []):
                    fid = int(entry["feature_id"])
                    m.setdefault(fid, entry.get("metrics", {}))
            return m

        feature_metrics_map = build_metrics_map(feature_stats)
        num_features = int(feature_stats.get("num_features", 0))
        mean_activation = feature_stats.get("mean_activation")
        max_activation = feature_stats.get("max_activation")
        fraction_active = feature_stats.get("fraction_active")
        if not (
            num_features > 0
            and isinstance(mean_activation, list)
            and isinstance(max_activation, list)
            and isinstance(fraction_active, list)
            and len(mean_activation) == num_features
            and len(max_activation) == num_features
            and len(fraction_active) == num_features
        ):
            print(
                "[feature_probe_server] Warning: feature_stats.json does not include full "
                "mean_activation/max_activation/fraction_active arrays. "
                "Only features present in metrics.*.top_features will have metric snapshots. "
                "Re-run the notebook Save Results cell to regenerate this run."
            )

        feature_metrics_map_baseline: dict[int, dict[str, float]] = {}
        if feature_stats_baseline is not None:
            feature_metrics_map_baseline = build_metrics_map(feature_stats_baseline)
        mean_act_squared_baseline: list[float] = []
        if feature_stats_baseline is not None:
            mas = feature_stats_baseline.get("mean_act_squared", [])
            mean_act_squared_baseline = mas if isinstance(mas, list) else []
        metric_names = feature_stats.get("metrics", {}).keys()
        print(
            "[feature_probe_server] Loaded metrics:",
            ", ".join(metric_names) if metric_names else "(none)",
        )

        headlines_path = run_path / "headline_features.json"
        headline_features: list[dict[str, Any]] = []
        if headlines_path.exists():
            with open(headlines_path, "r", encoding="utf-8") as f:
                headline_features = json.load(f)
        else:
            print("[feature_probe_server] Warning: headline_features.json not found; headline view will be empty.")

        baseline_headlines_path = run_path / "headline_features_baseline.json"
        baseline_headline_features: list[dict[str, Any]] | None = None
        if baseline_headlines_path.exists():
            with open(baseline_headlines_path, "r", encoding="utf-8") as f:
                baseline_headline_features = json.load(f)

        # Per-feature CNS over headlines where the feature was actually ablated.
        cns_sums: dict[int, float] = {}
        cns_counts: dict[int, int] = {}
        for row in headline_features:
            cns_val = row.get("cns")
            ablated_features = row.get("ablated_features")
            if not isinstance(cns_val, (int, float)) or not isinstance(ablated_features, list):
                continue
            for fid in ablated_features:
                try:
                    fid_int = int(fid)
                except Exception:  # noqa: BLE001
                    continue
                cns_sums[fid_int] = cns_sums.get(fid_int, 0.0) + float(cns_val)
                cns_counts[fid_int] = cns_counts.get(fid_int, 0) + 1
        avg_cns_by_feature = {
            fid: (cns_sums[fid] / cns_counts[fid])
            for fid in cns_sums
            if cns_counts.get(fid, 0) > 0
        }

        interpretability_path = run_path / "interpretability_llm_results.json"
        interpretability_results: list[dict[str, Any]] = []
        interpretability_by_feature: dict[str, dict[str, Any]] = {}
        if interpretability_path.exists():
            try:
                with open(interpretability_path, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                if isinstance(loaded, list):
                    interpretability_results = [
                        d for d in loaded if isinstance(d, dict)
                    ]
                elif isinstance(loaded, dict):
                    # Be tolerant: some runs may serialize results as a dict.
                    interpretability_results = [
                        d for d in loaded.values() if isinstance(d, dict)
                    ]
            except Exception as exc:  # noqa: BLE001
                print(
                    f"[feature_probe_server] Warning: failed to load "
                    f"{interpretability_path.name}: {exc}"
                )

        for entry in interpretability_results:
            fid = entry.get("feature_id")
            if fid is None:
                continue
            interpretability_by_feature[str(fid)] = entry

        return cls(
            run_path=run_path,
            run_metadata=run_metadata,
            feature_stats=feature_stats,
            feature_stats_baseline=feature_stats_baseline,
            feature_tokens=feature_tokens,
            feature_tokens_baseline=feature_tokens_baseline,
            mean_act_squared=feature_stats.get("mean_act_squared", []),
            mean_act_squared_baseline=mean_act_squared_baseline,
            feature_metrics_map=feature_metrics_map,
            feature_metrics_map_baseline=feature_metrics_map_baseline,
            headline_features=headline_features,
            baseline_headline_features=baseline_headline_features,
            avg_cns_by_feature=avg_cns_by_feature,
            interpretability_results=interpretability_results,
            interpretability_by_feature=interpretability_by_feature,
        )

    @property
    def metrics_available(self) -> list[str]:
        return list(self.feature_stats.get("metrics", {}).keys())

    @property
    def metric_descriptions(self) -> dict[str, str]:
        return {
            name: data.get("description", "")
            for name, data in self.feature_stats.get("metrics", {}).items()
        }

    @property
    def num_features(self) -> int:
        return int(self.feature_stats.get("num_features", 0))

    @property
    def num_tokens(self) -> int:
        return int(self.feature_stats.get("total_tokens", 0))

    @property
    def accuracy(self) -> float:
        return float(self.feature_stats.get("accuracy", 0.0))

    @property
    def num_samples(self) -> int:
        return int(self.feature_stats.get("num_samples", 0))

    @property
    def run_name(self) -> str:
        return self.run_path.name

    @property
    def run_id(self) -> int | None:
        match = re.search(r"run-(\d+)", self.run_path.name)
        if match is None:
            return None
        try:
            return int(match.group(1))
        except ValueError:
            return None

    @property
    def is_ablation_run(self) -> bool:
        return self.run_metadata.get("ablation_mode") is not None

    @property
    def has_baseline_headlines(self) -> bool:
        return self.baseline_headline_features is not None

    def get_headlines(self, limit: int = 100, variant: str = "ablated") -> list[dict[str, Any]]:
        variant = (variant or "ablated").lower()
        if variant not in ("baseline", "ablated"):
            raise ValueError("variant must be either 'baseline' or 'ablated'")

        if variant == "baseline":
            if self.is_ablation_run:
                source = self.baseline_headline_features or []
            else:
                # Non-ablation runs only have one headline file; treat it as baseline.
                source = self.headline_features
        else:
            source = self.headline_features

        if not source:
            return []
        limit = max(1, min(limit, len(source)))
        return source[:limit]

    def get_top_features(
        self, limit: int = 50, metric_name: str = "mean_activation"
    ) -> list[dict[str, Any]]:
        metrics_block = self.feature_stats.get("metrics", {})
        metric_name = metric_name or next(iter(metrics_block.keys()), None)
        if metric_name not in metrics_block:
            raise ValueError(
                f"Metric '{metric_name}' is not present. "
                f"Available metrics: {', '.join(metrics_block.keys())}"
            )
        top_features = metrics_block[metric_name].get("top_features", [])
        limit = max(1, min(limit, len(top_features)))
        enriched: list[dict[str, Any]] = []
        for entry in top_features[:limit]:
            e = dict(entry)
            try:
                fid = int(e.get("feature_id"))
            except Exception:  # noqa: BLE001
                fid = None
            e["avg_cns"] = self.avg_cns_by_feature.get(fid) if fid is not None else None
            enriched.append(e)
        return enriched

    def get_feature_tokens(
        self, feature_id: int, top_k: int = 10, variant: str = "ablated"
    ) -> dict[str, Any]:
        variant = (variant or "ablated").lower()
        tokens_dict = (
            self.feature_tokens_baseline if variant == "baseline" else self.feature_tokens
        )
        feature_key = str(feature_id)
        tokens = tokens_dict.get(feature_key, [])
        if not tokens:
            raise ValueError(
                f"No token information saved for feature {feature_id} (variant={variant}). "
                "Re-run inference with full per-feature token tracking, or try the other variant."
            )
        return {
            "feature_id": feature_id,
            "variant": variant,
            "top_k": min(top_k, len(tokens)),
            "tokens": tokens[:top_k],
        }

    def _build_hist(
        self,
        feature_id: int,
        tokens_dict: dict[str, list[dict[str, Any]]],
    ) -> dict[str, int]:
        hist: dict[str, int] = {}
        for entry in tokens_dict.get(str(feature_id), []):
            activation = entry.get("activation", 0.0)
            bin_index = int(activation // HIST_BIN_WIDTH)
            hist[str(bin_index)] = hist.get(str(bin_index), 0) + 1
        if not hist:
            hist["0"] = 0
        return hist

    def _make_sequence(self, entry: dict[str, Any], density: float) -> dict[str, Any]:
        token_id = entry.get("token_id")
        row_id = entry.get("row_id")
        return {
            "density": density,
            "doc_id": int(row_id) if isinstance(row_id, int) else -1,
            "idx": int(entry.get("token_position", 0)),
            "acts": [entry.get("activation", 0.0)],
            "act": entry.get("activation", 0.0),
            "tokens": [entry.get("token_str", "")],
            "token_ints": [int(token_id) if token_id is not None else -1],
            "prompt_snippet": entry.get("prompt_snippet", ""),
            "prompt": entry.get("prompt", ""),
            "prompt_tokens": entry.get("prompt_tokens", []),
            "predicted_label": entry.get("predicted_label"),
            "true_label": entry.get("true_label"),
        }

    def get_feature_info(self, feature_id: int, top_k: int = 10, variant: str = "ablated") -> dict[str, Any]:
        variant = (variant or "ablated").lower()
        if variant not in ("baseline", "ablated"):
            variant = "ablated"

        tokens_dict = (
            self.feature_tokens_baseline if variant == "baseline" else self.feature_tokens
        )
        if variant == "baseline" and not tokens_dict:
            raise ValueError(
                "No baseline token data in this run (missing feature_tokens_baseline.json). "
                "Re-run the notebook save cell after baseline tracking is enabled."
            )

        metrics_map = (
            self.feature_metrics_map_baseline
            if variant == "baseline" and self.feature_stats_baseline is not None
            else self.feature_metrics_map
        )
        mean_sq_vec = (
            self.mean_act_squared_baseline
            if variant == "baseline"
            and self.feature_stats_baseline is not None
            and bool(self.mean_act_squared_baseline)
            else self.mean_act_squared
        )

        if feature_id < 0 or feature_id >= self.num_features:
            raise ValueError(
                f"feature_id {feature_id} out of range (num_features={self.num_features})"
            )

        metrics = metrics_map.get(feature_id, {})
        mean_act = metrics.get("mean_activation", 0.0)
        density = metrics.get("fraction_active", 0.0)
        mean_sq = (
            mean_sq_vec[feature_id]
            if mean_sq_vec and 0 <= feature_id < len(mean_sq_vec)
            else mean_act**2
        )

        tokens = tokens_dict.get(str(feature_id), [])
        is_inactive = density <= 0 and len(tokens) == 0
        sequences = [self._make_sequence(entry, density) for entry in tokens]
        top_sequences = sequences[: min(top_k, len(sequences))]
        random_sequences = sequences[::-1][: min(top_k, len(sequences))]

        return {
            "variant": variant,
            "density": density,
            "mean_act": mean_act,
            "mean_act_squared": mean_sq,
            "avg_cns": self.avg_cns_by_feature.get(feature_id),
            "is_inactive": is_inactive,
            "hist": self._build_hist(feature_id, tokens_dict),
            "top": top_sequences,
            "random": random_sequences,
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
        self._set_cors_headers()
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _set_cors_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(200)
        self._set_cors_headers()
        self.end_headers()

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
                        "accuracy": self.data_store.accuracy,
                        "num_samples": self.data_store.num_samples,
                    }
                )
            except Exception as exc:  # noqa: BLE001
                self._send_json({"error": str(exc)}, status=400)
            return

        if parsed.path == "/api/headlines":
            params = parse_qs(parsed.query)
            limit = int(params.get("limit", ["100"])[0])
            variant = params.get("variant", ["ablated"])[0]
            try:
                headlines = self.data_store.get_headlines(limit, variant=variant)
                self._send_json({"headlines": headlines})
            except Exception as exc:  # noqa: BLE001
                self._send_json({"error": str(exc)}, status=400)
            return

        if parsed.path == "/api/metadata":
            try:
                metadata_path = self.data_store.run_path / "metadata.json"
                if metadata_path.exists():
                    with open(metadata_path, "r", encoding="utf-8") as f:
                        metadata = json.load(f)
                    metadata.setdefault("run_name", self.data_store.run_name)
                    if self.data_store.run_id is not None:
                        metadata.setdefault("run_id", self.data_store.run_id)
                    metadata.setdefault("has_baseline_headlines", self.data_store.has_baseline_headlines)
                    metadata.setdefault(
                        "has_feature_tokens_baseline",
                        bool(self.data_store.feature_tokens_baseline),
                    )
                    metadata.setdefault(
                        "has_feature_stats_baseline",
                        self.data_store.feature_stats_baseline is not None,
                    )
                    if self.data_store.is_ablation_run:
                        metadata.setdefault(
                            "ablated_accuracy",
                            metadata.get("accuracy", self.data_store.accuracy),
                        )
                        if "baseline_accuracy" in metadata:
                            metadata["baseline_accuracy"] = float(metadata["baseline_accuracy"])
                    self._send_json({"metadata": metadata})
                else:
                    metadata = {"run_name": self.data_store.run_name}
                    if self.data_store.run_id is not None:
                        metadata["run_id"] = self.data_store.run_id
                    metadata["has_baseline_headlines"] = self.data_store.has_baseline_headlines
                    metadata["has_feature_tokens_baseline"] = bool(
                        self.data_store.feature_tokens_baseline
                    )
                    metadata["has_feature_stats_baseline"] = (
                        self.data_store.feature_stats_baseline is not None
                    )
                    self._send_json({"metadata": metadata})
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
            variant = params.get("variant", ["ablated"])[0]
            try:
                data = self.data_store.get_feature_tokens(
                    int(feature_id[0]), top_k=top_k, variant=variant
                )
                self._send_json(data)
            except Exception as exc:  # noqa: BLE001
                self._send_json({"error": str(exc)}, status=400)
            return

        if parsed.path == "/api/feature_info":
            params = parse_qs(parsed.query)
            feature_id = params.get("id")
            if feature_id is None:
                self._send_json({"error": "Missing 'id' query parameter"}, status=400)
                return
            top_k = int(params.get("top_k", ["10"])[0])
            variant = params.get("variant", ["ablated"])[0]
            try:
                data = self.data_store.get_feature_info(
                    int(feature_id[0]), top_k=top_k, variant=variant
                )
                self._send_json(data)
            except Exception as exc:  # noqa: BLE001
                self._send_json({"error": str(exc)}, status=400)
            return

        if parsed.path == "/api/feature_cns":
            params = parse_qs(parsed.query)
            feature_id = params.get("id")
            if feature_id is None:
                self._send_json({"error": "Missing 'id' query parameter"}, status=400)
                return
            try:
                fid = int(feature_id[0])
                avg_cns = self.data_store.avg_cns_by_feature.get(fid)
                self._send_json({"feature_id": fid, "avg_cns": avg_cns})
            except Exception as exc:  # noqa: BLE001
                self._send_json({"error": str(exc)}, status=400)
            return

        if parsed.path == "/api/interpretability/features":
            features_out: list[dict[str, Any]] = []
            for entry in self.data_store.interpretability_results:
                fid = entry.get("feature_id")
                try:
                    fid_int = int(fid)
                except Exception:  # noqa: BLE001
                    fid_int = fid
                features_out.append(
                    {
                        "feature_id": fid_int,
                        "correlation": entry.get("correlation"),
                        "n_eval": entry.get("n_eval"),
                        "skipped": entry.get("skipped", False),
                        "error": entry.get("error"),
                    }
                )
            self._send_json(
                {
                    "features": features_out,
                    "has_results": bool(features_out),
                }
            )
            return

        if parsed.path == "/api/interpretability/feature":
            params = parse_qs(parsed.query)
            feature_id = params.get("id")
            if feature_id is None:
                self._send_json({"error": "Missing 'id' query parameter"}, status=400)
                return
            key = str(feature_id[0])
            entry = self.data_store.interpretability_by_feature.get(key)
            if entry is None:
                self._send_json(
                    {"error": f"No interpretability results for feature {key}"},
                    status=404,
                )
                return
            self._send_json(entry)
            return

        if parsed.path in ("", "/"):
            self.path = "/feature_probe_frontend.html"
        return super().do_GET()


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the feature probe React UI.")
    parser.add_argument("--host", default="127.0.0.1", help="Hostname to bind (default 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8765, help="Port to bind (default 8765)")
    parser.add_argument("--run-id", type=int, default=None, help="Specific run ID to load (e.g., 12 for run-012). If not provided, loads the latest run.")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    data_store = FeatureProbeData.load(repo_root, run_id=args.run_id)

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

