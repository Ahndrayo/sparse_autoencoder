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
  | "predicted_label"
  | "confidence"
  | "true_label"
  | "prompt";

export default function HeadlinesView({ onFeatureClick, accuracy, numSamples }: Props) {
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
        setHeadlines(data.headlines || []);
      })
      .catch((err) => {
        if (cancelled) return;
        setError(err.message || "Failed to load headlines.");
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

  const sortedHeadlines = useMemo(() => {
    const sorted = [...headlines];
    const dir = sortDir === "asc" ? 1 : -1;
    sorted.sort((a, b) => {
      const valueA = a[sortKey];
      const valueB = b[sortKey];
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
        <h1>Headlines</h1>
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

      {loading && <div>Loading headlines…</div>}
      {error && <div className="error-message">{error}</div>}

      {!loading && !error && (
        <div className="headlines-table-wrapper">
          <table className="feature-table">
            <thead>
              <tr>
                <th>{renderSortLabel("ID", "row_id")}</th>
                <th>{renderSortLabel("Prediction", "predicted_label")}</th>
                <th>{renderSortLabel("Confidence", "confidence")}</th>
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

