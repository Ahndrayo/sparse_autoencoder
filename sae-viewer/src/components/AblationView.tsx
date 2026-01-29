import React, { useEffect, useState } from "react";
import { fetchHeadlines } from "../interpAPI";
import { HeadlineInfo } from "../types";

type Props = {
  onFeatureClick?: (featureId: number) => void;
};

export default function AblationView({ onFeatureClick }: Props) {
  const [headlines, setHeadlines] = useState<HeadlineInfo[]>([]);
  const [limit, setLimit] = useState(100);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [expanded, setExpanded] = useState<Record<number, boolean>>({});

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);
    fetchHeadlines(limit)
      .then((data) => {
        if (cancelled) return;
        const rawHeadlines = data.headlines || [];
        const ablationHeadlines = rawHeadlines.filter(
          (headline: HeadlineInfo) => headline.num_ablated_features !== undefined
        );
        setHeadlines(ablationHeadlines);
      })
      .catch((err) => {
        if (cancelled) return;
        setError(err.message || "Failed to load ablation headlines.");
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [limit]);

  const toggleExpand = (rowId: number) => {
    setExpanded((prev) => ({ ...prev, [rowId]: !prev[rowId] }));
  };

  const getImpactColor = (fraction?: number) => {
    if (fraction === undefined) return "#6b7280";
    if (fraction < 0.2) return "#22c55e";
    if (fraction < 0.5) return "#eab308";
    return "#ef4444";
  };

  return (
    <div className="headlines-view">
      <header className="headlines-header">
        <h1>Ablation</h1>
        <div className="control-row">
          <label>
            Max headlines
            <input
              type="number"
              min={10}
              max={500}
              value={limit}
              onChange={(e) => setLimit(Number(e.target.value))}
            />
          </label>
        </div>
      </header>

      {loading && <div>Loading ablation headlines…</div>}
      {error && <div className="error-message">{error}</div>}

      {!loading && !error && headlines.length === 0 && (
        <div className="no-ablation-data">
          No ablation data found for this run. Run the ablation cell to see
          ablation metrics.
        </div>
      )}

      {!loading && !error && headlines.length > 0 && (
        <div className="headlines-table-wrapper">
          <table className="feature-table">
            <thead>
              <tr>
                <th>ID</th>
                <th>Prediction</th>
                <th>Confidence</th>
                <th>Ablation Impact</th>
                <th>True Label</th>
                <th>Headline</th>
                <th>Top Features</th>
              </tr>
            </thead>
            <tbody>
              {headlines.map((h) => {
                const topFeatures = h.features || [];
                const showAll = expanded[h.row_id] || false;
                const displayFeatures = showAll ? topFeatures.slice(0, 10) : topFeatures.slice(0, 3);
                return (
                  <tr key={h.row_id}>
                    <td className="center">{h.row_id}</td>
                    <td
                      className="center"
                      style={{
                        color: h.correct ? "#22c55e" : "#ef4444",
                        fontWeight: "bold",
                      }}
                    >
                      {h.predicted_label}
                    </td>
                    <td className="center">
                      {h.confidence !== undefined
                        ? `${(h.confidence * 100).toFixed(1)}%`
                        : "—"}
                    </td>
                    <td className="center ablation-metrics">
                      <div className="ablation-count">
                        {h.num_ablated_features}/{h.total_baseline_features} ablated
                      </div>
                      <div
                        className="ablation-fraction"
                        style={{ color: getImpactColor(h.ablation_fraction) }}
                      >
                        {(h.ablation_fraction! * 100).toFixed(1)}% removed
                      </div>
                    </td>
                    <td className="center">{h.true_label}</td>
                    <td style={{ maxWidth: "360px" }}>
                      <span>{h.prompt}</span>
                    </td>
                    <td>
                      <div className="feature-chip-row">
                        {displayFeatures.map((f) => (
                          <button
                            key={f.feature_id}
                            className="feature-chip"
                            onClick={() => onFeatureClick?.(f.feature_id)}
                            title={`Activation: ${f.max_activation?.toFixed(3) ?? "—"}`}
                          >
                            {f.feature_id}: {f.token_str}
                          </button>
                        ))}
                        {topFeatures.length > 3 && (
                          <button
                            className="feature-chip ghost"
                            onClick={() => toggleExpand(h.row_id)}
                          >
                            {showAll ? "Hide" : `+${Math.min(topFeatures.length, 10) - 3} more`}
                          </button>
                        )}
                      </div>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
