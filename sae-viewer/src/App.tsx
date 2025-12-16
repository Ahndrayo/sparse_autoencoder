import "./App.css";
import React, { useEffect, useState } from "react";
import FeatureInfo from "./components/featureInfo";
import { fetchFeatures } from "./interpAPI";

const LOCAL_AUTOENCODER = {
  subject: "local",
  family: "local",
  num_features: 0,
  H: {},
  path: "",
};

function App() {
  const [limit, setLimit] = useState(100);
  const [metric, setMetric] = useState("mean_activation");
  const [payload, setPayload] = useState<{
    features: any[];
    metrics_available: string[];
    metric_descriptions: Record<string, string>;
    metric: string;
  } | null>(null);
  const [selectedFeatureId, setSelectedFeatureId] = useState<number | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);
    fetchFeatures(limit, metric)
      .then((data) => {
        if (cancelled) return;
        setPayload(data);
        if (!data.features.some((f) => f.feature_id === selectedFeatureId)) {
          setSelectedFeatureId(data.features[0]?.feature_id ?? null);
        }
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

  const selectedFeature =
    selectedFeatureId !== null
      ? { atom: selectedFeatureId, autoencoder: LOCAL_AUTOENCODER }
      : null;

  const featureRows = payload?.features || [];

  return (
    <div className="app-shell">
      <div className="column features-column">
        <header>
          <h1>Sparse AE features</h1>
          <p>
            Viewing the latest run&apos;s top {limit} features sorted by{" "}
            <strong>{payload?.metric || metric}</strong>.
          </p>
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
        </div>
        {error && <div className="error-message">{error}</div>}
        <div className="feature-table-wrapper">
          {loading ? (
            <div>Loading features…</div>
          ) : (
            <table className="feature-table">
              <thead>
                <tr>
                  <th>Feature</th>
                  <th>Value</th>
                  <th>Mean</th>
                  <th>Max</th>
                  <th>Frac>0</th>
                </tr>
              </thead>
              <tbody>
                {featureRows.map((feat) => {
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
          {payload?.metric && payload.metric_descriptions[payload.metric] && (
            <p className="metric-description">
              {payload.metric_descriptions[payload.metric]}
            </p>
          )}
        </header>
        {selectedFeature ? (
          <FeatureInfo feature={selectedFeature} />
        ) : (
          <div>Select a feature from the left column.</div>
        )}
      </div>
    </div>
  );
}

export default App;
