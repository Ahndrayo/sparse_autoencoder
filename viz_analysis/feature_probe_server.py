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

from utils.run_picker import get_latest_run

STATIC_DIR = Path(__file__).resolve().parent


@dataclass
class FeatureProbeData:
    run_path: Path
    feature_stats: dict[str, Any]
    feature_tokens: dict[str, list[dict[str, Any]]]

    @classmethod
    def load(cls, repo_root: Path) -> "FeatureProbeData":
        analysis_dir = repo_root / "analysis_data"
        run_path = get_latest_run(analysis_dir)
        print(f"[feature_probe_server] Using run: {run_path}")

        summary_path = run_path / "feature_stats.json"
        if not summary_path.exists():
            raise FileNotFoundError(
                f"Expected feature_stats.json in {run_path}, please rerun main.py."
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

        metric_names = feature_stats.get("metrics", {}).keys()
        print(
            "[feature_probe_server] Loaded metrics:",
            ", ".join(metric_names) if metric_names else "(none)",
        )

        return cls(run_path=run_path, feature_stats=feature_stats, feature_tokens=feature_tokens)

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
        return top_features[:limit]

    def get_feature_tokens(self, feature_id: int, top_k: int = 10) -> dict[str, Any]:
        feature_key = str(feature_id)
        tokens = self.feature_tokens.get(feature_key, [])
        if not tokens:
            raise ValueError(
                f"No token information saved for feature {feature_id}. "
                "Only preselected features are available."
            )
        return {
            "feature_id": feature_id,
            "top_k": min(top_k, len(tokens)),
            "tokens": tokens[:top_k],
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

