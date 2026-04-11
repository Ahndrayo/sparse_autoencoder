import React, { useEffect, useState, useRef } from "react";
import FeatureInfo from "./featureInfo";
import { fetchFeatureCns, fetchFeatures, fetchMetadata } from "../interpAPI";

type Props = {
  initialFeatureId?: number | null;
  onClearSearch?: () => void;
};

const LOCAL_AUTOENCODER = {
  subject: "local",
  family: "local",
  num_features: 0,
  H: {},
  path: "",
};

export default function FeaturesView({ initialFeatureId, onClearSearch }: Props) {
  const [limit, setLimit] = useState(100);
  const [metric, setMetric] = useState("mean_activation");
  const [payload, setPayload] = useState<{
    features: any[];
    metrics_available: string[];
    metric_descriptions: Record<string, string>;
    metric: string;
    accuracy?: number;
    num_samples?: number;
    num_features?: number;
    num_tokens?: number;
  } | null>(null);
  const [selectedFeatureId, setSelectedFeatureId] = useState<number | null>(initialFeatureId ?? null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [searchInput, setSearchInput] = useState("");
  const initialFeatureRef = useRef<number | null>(initialFeatureId ?? null);
  const [hasBaselineTokens, setHasBaselineTokens] = useState(false);
  const [tokenVariant, setTokenVariant] = useState<"baseline" | "ablated">("ablated");
  const [selectedAvgCns, setSelectedAvgCns] = useState<number | null>(null);

  useEffect(() => {
    fetchMetadata()
      .then((d) => {
        setHasBaselineTokens(d.metadata?.has_feature_tokens_baseline === true);
      })
      .catch(() => setHasBaselineTokens(false));
  }, []);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);
    fetchFeatures(limit, metric)
      .then((data) => {
        if (cancelled) return;
        setPayload(data);
        setSelectedFeatureId((prev) => {
          const desired = initialFeatureRef.current;

          if (desired !== null) {
            // If the user/search has already moved away from the initial selection,
            // don't allow future refetches to hijack the UI.
            if (prev !== desired) {
              initialFeatureRef.current = null;
              return prev;
            }

            // Only consume the initial selection when it is present in the loaded top table.
            if (data.features.some((f) => f.feature_id === desired)) {
              initialFeatureRef.current = null;
              return desired;
            }

            // Keep the currently selected feature even if it is outside the top-N table.
            return prev;
          }

          // Initial load (no selection yet): default to the first row.
          if (prev === null) {
            return data.features[0]?.feature_id ?? null;
          }

          return prev;
        });
      })
      .catch((err) => {
        if (cancelled) return;
        setError(err.message || "Failed to load features.");
      })
      .finally(() => {
        if (!cancelled) {
          setLoading(false);
        }
      });
    return () => {
      cancelled = true;
    };
  }, [limit, metric]);

  // Update selection if parent passes a new initialFeatureId
  useEffect(() => {
    if (initialFeatureId !== undefined && initialFeatureId !== null) {
      setSelectedFeatureId(initialFeatureId);
      setSearchInput(String(initialFeatureId));
    }
  }, [initialFeatureId]);

  const handleSearch = () => {
    const id = parseInt(searchInput);
    if (!isNaN(id)) {
      setSelectedFeatureId(id);
    }
  };

  const selectedFeature =
    selectedFeatureId !== null
      ? { atom: selectedFeatureId, autoencoder: LOCAL_AUTOENCODER }
      : null;

  const featureRows = payload?.features || [];
  const selectedFeatureRow =
    selectedFeatureId !== null
      ? featureRows.find((f) => f.feature_id === selectedFeatureId)
      : null;
  const visibleRows = featureRows.filter((feat) => {
    const metrics = feat.metrics || {};
    const frac = metrics.fraction_active ?? 0;
    const maxAct = metrics.max_activation ?? 0;
    // "Dead" latent features never fired (no positive activations) in this run.
    return !(frac === 0 && maxAct === 0);
  });

  useEffect(() => {
    let cancelled = false;
    if (selectedFeatureId === null) {
      setSelectedAvgCns(null);
      return;
    }

    // Fast path when selected feature is in current top-N payload.
    if (selectedFeatureRow && typeof selectedFeatureRow.avg_cns === "number") {
      setSelectedAvgCns(selectedFeatureRow.avg_cns);
    } else {
      setSelectedAvgCns(null);
    }

    // Always fetch independently so searched features outside top-N still show Avg CNS.
    fetchFeatureCns(selectedFeatureId)
      .then((d) => {
        if (cancelled) return;
        setSelectedAvgCns(typeof d.avg_cns === "number" ? d.avg_cns : null);
      })
      .catch(() => {
        if (!cancelled) setSelectedAvgCns(null);
      });

    return () => {
      cancelled = true;
    };
  }, [selectedFeatureId, selectedFeatureRow]);

  return (
    <div className="app-shell">
      <div className="column features-column">
        <header>
          <h1>Sparse AE features</h1>
          <p>
            Viewing the latest run&apos;s top {limit} features sorted by{" "}
            <strong>{payload?.metric || metric}</strong>.
          </p>
          {payload?.accuracy !== undefined && (
            <p className="accuracy-display">
              <strong>Model Accuracy:</strong>{" "}
              <span className="accuracy-value">
                {(payload.accuracy * 100).toFixed(2)}%
              </span>
              {payload.num_samples && (
                <span className="sample-count">
                  {" "}
                  ({payload.num_samples} samples)
                </span>
              )}
            </p>
          )}
        </header>

        <div className="control-row">
          <label>
            Top features
            <input
              type="number"
              min={10}
              max={500}
              value={limit}
              onChange={(event) => setLimit(Number(event.target.value))}
            />
          </label>
          <label>
            Metric
            <select
              value={metric}
              onChange={(event) => setMetric(event.target.value)}
            >
              {payload?.metrics_available.map((item) => (
                <option key={item} value={item}>
                  {item}
                </option>
              ))}
            </select>
          </label>
          {hasBaselineTokens && (
            <label>
              Token examples
              <select
                value={tokenVariant}
                onChange={(e) =>
                  setTokenVariant(e.target.value as "baseline" | "ablated")
                }
              >
                <option value="ablated">Ablated run</option>
                <option value="baseline">Baseline (no ablation)</option>
              </select>
            </label>
          )}
        </div>

        <div className="control-row search-row">
          <input
            type="number"
            placeholder="Search feature ID..."
            value={searchInput}
            onChange={(e) => setSearchInput(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && handleSearch()}
          />
          <button onClick={handleSearch}>Search</button>
          {onClearSearch && (
            <button className="ghost" onClick={onClearSearch}>
              Clear
            </button>
          )}
        </div>

        {error && <div className="error-message">{error}</div>}
        <div className="feature-table-wrapper">
          {loading ? (
            <div>Loading features…</div>
          ) : visibleRows.length === 0 ? (
            <div style={{ padding: "8px 0" }}>
              No active features in the current top list. Try a different metric or increase{" "}
              <code>Top features</code>.
            </div>
          ) : (
            <table className="feature-table">
              <thead>
                <tr>
                  <th>Feature</th>
                  <th>Value</th>
                  <th>Mean</th>
                  <th>Max</th>
                  <th>Frac&gt;0</th>
                </tr>
              </thead>
              <tbody>
                {visibleRows.map((feat) => {
                  const metrics = feat.metrics || {};
                  return (
                    <tr
                      key={feat.feature_id}
                      className={
                        feat.feature_id === selectedFeatureId ? "active" : ""
                      }
                      onClick={() => setSelectedFeatureId(feat.feature_id)}
                    >
                      <td>{feat.feature_id}</td>
                      <td>{feat.value?.toFixed(4)}</td>
                      <td>
                        {metrics.mean_activation?.toFixed(4) ?? "—"}
                      </td>
                      <td>{metrics.max_activation?.toFixed(4) ?? "—"}</td>
                      <td>
                        {metrics.fraction_active?.toFixed(3) ?? "—"}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          )}
        </div>
      </div>

      <div className="column info-column">
        <header>
          <h2>
            {selectedFeatureId !== null
              ? `Feature ${selectedFeatureId}`
              : "Select a feature"}
          </h2>
          {selectedFeatureId !== null && (
            <p className="metric-description">
              Avg CNS:{" "}
              {selectedAvgCns !== null
                ? `${selectedAvgCns >= 0 ? "+" : ""}${selectedAvgCns.toFixed(3)}`
                : "—"}
            </p>
          )}
          {payload?.metric && payload.metric_descriptions[payload.metric] && (
            <p className="metric-description">
              {payload.metric_descriptions[payload.metric]}
            </p>
          )}
        </header>
        {selectedFeature ? (
          <FeatureInfo feature={selectedFeature} tokenVariant={tokenVariant} />
        ) : (
          <div>Select a feature from the left column.</div>
        )}
      </div>
    </div>
  );
}





