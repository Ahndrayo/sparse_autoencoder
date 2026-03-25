import React, { useEffect, useMemo, useRef, useState } from "react";
import {
  fetchInterpretabilityFeatures,
  fetchInterpretabilityFeature,
} from "../interpAPI";

type InterpFeatureSummary = {
  feature_id: number | string;
  correlation?: number | null;
  n_eval?: number | null;
  skipped?: boolean;
  error?: string;
};

export default function InterpretedView() {
  const [hasResults, setHasResults] = useState<boolean | null>(null);
  const [featureSummaries, setFeatureSummaries] = useState<InterpFeatureSummary[]>([]);
  const [searchInput, setSearchInput] = useState("");
  const [selectedFeatureId, setSelectedFeatureId] = useState<number | null>(null);

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [detail, setDetail] = useState<any | null>(null);

  const requestId = useRef(0);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);
    fetchInterpretabilityFeatures()
      .then((data) => {
        if (cancelled) return;
        setHasResults(Boolean(data?.has_results));
        const feats: InterpFeatureSummary[] = Array.isArray(data?.features)
          ? data.features
          : [];
        setFeatureSummaries(feats);
        if (feats.length > 0) {
          const first = feats[0]?.feature_id;
          const firstId =
            typeof first === "string" ? parseInt(first, 10) : first;
          setSelectedFeatureId(Number.isFinite(firstId as number) ? (firstId as number) : null);
        }
      })
      .catch((err) => {
        if (cancelled) return;
        setHasResults(false);
        setFeatureSummaries([]);
        setError(err.message || "Failed to load interpretability results.");
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (selectedFeatureId === null) return;
    let cancelled = false;
    const myRequestId = ++requestId.current;

    setLoading(true);
    setError(null);

    fetchInterpretabilityFeature(selectedFeatureId)
      .then((d) => {
        if (cancelled) return;
        if (myRequestId !== requestId.current) return;
        setDetail(d);
      })
      .catch((err) => {
        if (cancelled) return;
        if (myRequestId !== requestId.current) return;
        setDetail(null);
        setError(err.message || "Failed to load interpretability feature details.");
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [selectedFeatureId]);

  const filteredSummaries = useMemo(() => {
    const q = searchInput.trim();
    if (!q) return featureSummaries;
    return featureSummaries.filter((f) => {
      const fid = typeof f.feature_id === "string" ? f.feature_id : String(f.feature_id);
      return fid.includes(q);
    });
  }, [featureSummaries, searchInput]);

  const correlation = detail?.correlation ?? null;
  const explanation = detail?.explanation ?? "";
  const explExamples = Array.isArray(detail?.explanation_examples)
    ? detail.explanation_examples
    : [];
  const evalExamples = Array.isArray(detail?.evaluation_examples)
    ? detail.evaluation_examples
    : [];

  const maxExplEvalRows = Math.max(explExamples.length, evalExamples.length);

  return (
    <div className="app-shell">
      <div className="column features-column">
        <header>
          <h1>Interpreted</h1>
          <p>
            LLM explanations and sampled examples from{" "}
            <code>interpretability_llm_results.json</code>.
          </p>
        </header>

        <div className="control-row search-row">
          <input
            type="number"
            placeholder="Search feature ID..."
            value={searchInput}
            onChange={(e) => setSearchInput(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && null}
          />
          <button
            onClick={() => {
              const q = searchInput.trim();
              if (!q) return;
              const parsed = parseInt(q, 10);
              if (!Number.isNaN(parsed)) setSelectedFeatureId(parsed);
            }}
          >
            Search
          </button>
        </div>

        {hasResults === false ? (
          <div style={{ marginTop: 12 }}>
            No interpretability results found for the selected run.
          </div>
        ) : null}

        {error ? <div className="error-message">{error}</div> : null}

        <div className="feature-table-wrapper">
          {loading ? (
            <div>Loading…</div>
          ) : (
            <table className="feature-table">
              <thead>
                <tr>
                  <th>Feature</th>
                  <th>ρ</th>
                  <th>n</th>
                </tr>
              </thead>
              <tbody>
                {filteredSummaries.map((f) => {
                  const fidNum =
                    typeof f.feature_id === "string"
                      ? parseInt(f.feature_id, 10)
                      : f.feature_id;
                  const fidSafe = Number.isFinite(fidNum as number)
                    ? (fidNum as number)
                    : null;
                  return (
                    <tr
                      key={String(f.feature_id)}
                      className={fidSafe === selectedFeatureId ? "active" : ""}
                      onClick={() => {
                        if (fidSafe !== null) setSelectedFeatureId(fidSafe);
                      }}
                    >
                      <td>{f.feature_id}</td>
                      <td>
                        {f.correlation === null || f.correlation === undefined
                          ? "—"
                          : (f.correlation as number).toFixed(3)}
                      </td>
                      <td>{f.n_eval ?? "—"}</td>
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
              : "Select an interpreted feature"}
          </h2>
        </header>

        {selectedFeatureId === null ? (
          <div>Select a feature from the left list.</div>
        ) : loading && !detail ? (
          <div style={{ padding: "12px" }}>Loading feature details…</div>
        ) : detail ? (
          <div>
            <div style={{ marginBottom: 12 }}>
              <b>LLM Explanation</b>
              <div style={{ marginTop: 6, whiteSpace: "pre-wrap" }}>
                {explanation || "—"}
              </div>
            </div>

            <div style={{ marginBottom: 12 }}>
              <b>Spearman ρ</b>{" "}
              {correlation === null || correlation === undefined
                ? "—"
                : (correlation as number).toFixed(3)}
            </div>

            <table className="activations-table" style={{ width: "100%" }}>
              <thead>
                <tr>
                  <th>Explanation Sample</th>
                  <th>Score</th>
                  <th>Evaluation Sample</th>
                  <th>True</th>
                  <th>Pred</th>
                </tr>
              </thead>
              <tbody>
                {maxExplEvalRows === 0 ? (
                  <tr>
                    <td colSpan={5}>No example snippets stored.</td>
                  </tr>
                ) : (
                  Array.from({ length: maxExplEvalRows }).map((_, i) => {
                    const expl = explExamples[i] ?? null;
                    const ev = evalExamples[i] ?? null;
                    return (
                      <tr key={i}>
                        <td style={{ verticalAlign: "top" }}>
                          <div style={{ whiteSpace: "pre-wrap" }}>
                            {expl?.snippet || "—"}
                          </div>
                        </td>
                        <td style={{ verticalAlign: "top" }}>
                          {expl?.quantized_score ?? "—"}
                        </td>
                        <td style={{ verticalAlign: "top" }}>
                          <div style={{ whiteSpace: "pre-wrap" }}>
                            {ev?.snippet || "—"}
                          </div>
                        </td>
                        <td style={{ verticalAlign: "top" }}>
                          {ev?.quantized_true_score ?? "—"}
                        </td>
                        <td style={{ verticalAlign: "top" }}>
                          {ev?.predicted_score ?? "—"}
                        </td>
                      </tr>
                    );
                  })
                )}
              </tbody>
            </table>
          </div>
        ) : (
          <div style={{ padding: "12px" }}>No details available.</div>
        )}
      </div>
    </div>
  );
}

