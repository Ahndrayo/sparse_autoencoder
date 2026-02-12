import React, { useEffect, useMemo, useState } from "react";
import { fetchHeadlines } from "../interpAPI";
import { HeadlineInfo } from "../types";

type Props = {
  onFeatureClick?: (featureId: number) => void;
  accuracy?: number | null;
  numSamples?: number | null;
};

type SortKey =
  | "row_id"
  | "baseline_prediction"
  | "predicted_label"
  | "confidence_delta"
  | "transition"
  | "ablation_energy_fraction_global"
  | "true_label"
  | "prompt";

export default function AblationView({ onFeatureClick, accuracy, numSamples }: Props) {
  const [headlines, setHeadlines] = useState<HeadlineInfo[]>([]);
  const [limit, setLimit] = useState(100);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [expanded, setExpanded] = useState<Record<number, boolean>>({});
  const [sortKey, setSortKey] = useState<SortKey>("row_id");
  const [sortDir, setSortDir] = useState<"asc" | "desc">("asc");

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

  const sortedHeadlines = useMemo(() => {
    const sorted = [...headlines];
    const dir = sortDir === "asc" ? 1 : -1;
    sorted.sort((a, b) => {
      const valueA =
        sortKey === "ablation_energy_fraction_global"
          ? a.ablation_energy_fraction_global
          : a[sortKey];
      const valueB =
        sortKey === "ablation_energy_fraction_global"
          ? b.ablation_energy_fraction_global
          : b[sortKey];
      if (valueA === undefined || valueA === null) return 1;
      if (valueB === undefined || valueB === null) return -1;
      if (typeof valueA === "number" && typeof valueB === "number") {
        return (valueA - valueB) * dir;
      }
      return String(valueA).localeCompare(String(valueB)) * dir;
    });
    return sorted;
  }, [headlines, sortDir, sortKey]);

  const toggleSort = (key: SortKey) => {
    if (sortKey === key) {
      setSortDir((prev) => (prev === "asc" ? "desc" : "asc"));
    } else {
      setSortKey(key);
      setSortDir("asc");
    }
  };

  const renderSortLabel = (label: string, key: SortKey) => {
    const isActive = sortKey === key;
    const indicator = isActive ? (sortDir === "asc" ? "^" : "v") : "";
    return (
      <button
        className={`sortable-button ${isActive ? "active" : ""}`}
        onClick={() => toggleSort(key)}
        type="button"
      >
        {label}
        {indicator && <span className="sort-indicator">{indicator}</span>}
      </button>
    );
  };

  return (
    <div className="headlines-view">
      <header className="headlines-header">
        <h1>Ablation</h1>
        {accuracy !== null && accuracy !== undefined && (
          <p className="accuracy-display">
            <strong>Model Accuracy:</strong>{" "}
            <span className="accuracy-value">{(accuracy * 100).toFixed(2)}%</span>
            {numSamples ? (
              <span className="sample-count"> ({numSamples} samples)</span>
            ) : null}
          </p>
        )}
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
                <th>{renderSortLabel("ID", "row_id")}</th>
                <th>{renderSortLabel("Base Prediction", "baseline_prediction")}</th>
                <th>{renderSortLabel("Ablated Prediction", "predicted_label")}</th>
                <th>{renderSortLabel("Delta Confidence", "confidence_delta")}</th>
                <th>{renderSortLabel("Flip Outcome", "transition")}</th>
                <th>{renderSortLabel("Ablation Impact", "ablation_energy_fraction_global")}</th>
                <th>{renderSortLabel("True Label", "true_label")}</th>
                <th>{renderSortLabel("Headline", "prompt")}</th>
                <th>Top Features</th>
              </tr>
            </thead>
            <tbody>
              {sortedHeadlines.map((h) => {
                const topFeatures = h.features || [];
                const showAll = expanded[h.row_id] || false;
                const displayFeatures = showAll ? topFeatures.slice(0, 10) : topFeatures.slice(0, 3);
                const globalFraction = h.ablation_fraction_global;
                const globalEnergyFraction = h.ablation_energy_fraction_global;
                const impactFraction = globalFraction ?? h.ablation_fraction;
                const impactPercent =
                  impactFraction !== undefined
                    ? `${(impactFraction * 100).toFixed(1)}%`
                    : "â€”";
                const energyPercent =
                  globalEnergyFraction !== undefined
                    ? `${(globalEnergyFraction * 100).toFixed(1)}%`
                    : "â€”";
                const hasGlobalMetrics =
                  globalFraction !== undefined || globalEnergyFraction !== undefined;
                return (
                  <tr key={h.row_id}>
                    <td className="center">{h.row_id}</td>
                    <td className="center">
                      {h.baseline_prediction || "—"}
                      {h.baseline_confidence !== undefined && (
                        <div className="confidence-badge">
                          {(h.baseline_confidence * 100).toFixed(1)}%
                        </div>
                      )}
                    </td>
                    <td
                      className="center"
                      style={{
                        color: h.correct ? "#22c55e" : "#ef4444",
                        fontWeight: "bold",
                      }}
                    >
                      {h.predicted_label}
                      {h.confidence !== undefined && (
                        <div className="confidence-badge">
                          {(h.confidence * 100).toFixed(1)}%
                        </div>
                      )}
                    </td>
                    <td
                      className="center"
                      style={{
                        color:
                          h.confidence_delta === undefined
                            ? "#6b7280"
                            : Math.abs(h.confidence_delta) < 0.05
                            ? "#6b7280"
                            : h.confidence_delta > 0
                            ? "#22c55e"
                            : "#ef4444",
                        fontWeight: "bold",
                      }}
                    >
                      {h.confidence_delta !== undefined
                        ? `${h.confidence_delta >= 0 ? "+" : ""}${(
                            h.confidence_delta * 100
                          ).toFixed(1)}%`
                        : "—"}
                    </td>
                    <td
                      className="center transition-cell"
                      style={{
                        color:
                          h.transition === "C -> W"
                            ? "#ef4444"
                            : h.transition === "W -> C"
                            ? "#22c55e"
                            : "#6b7280",
                        fontWeight: h.transition ? "bold" : "normal",
                      }}
                    >
                      {h.transition || "—"}
                    </td>
                    <td className="center ablation-metrics">
                      {hasGlobalMetrics ? (
                        <>
                          <div className="ablation-count">Global mass removed</div>
                          <div
                            className="ablation-fraction"
                            style={{ color: getImpactColor(impactFraction) }}
                          >
                            {impactPercent}
                          </div>
                          <div className="ablation-count">Global energy removed</div>
                          <div className="ablation-fraction">{energyPercent}</div>
                        </>
                      ) : (
                        <>
                          <div className="ablation-count">
                            {h.num_ablated_features}/{h.total_baseline_features} ablated
                          </div>
                          <div
                            className="ablation-fraction"
                            style={{ color: getImpactColor(h.ablation_fraction) }}
                          >
                            {(h.ablation_fraction! * 100).toFixed(1)}% removed
                          </div>
                        </>
                      )}
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
