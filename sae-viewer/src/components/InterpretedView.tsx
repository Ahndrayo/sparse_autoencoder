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

  const [explIndex, setExplIndex] = useState(0);
  const [evalIndex, setEvalIndex] = useState(0);

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
        setExplIndex(0);
        setEvalIndex(0);
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

  const curExpl = explExamples[explIndex] ?? null;
  const curEval = evalExamples[evalIndex] ?? null;

  const hasExpl = explExamples.length > 0 && curExpl !== null;
  const hasEval = evalExamples.length > 0 && curEval !== null;

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
                <tr>
                  <td style={{ verticalAlign: "top" }}>
                    {hasExpl ? (
                      <div>
                        <div style={{ whiteSpace: "pre-wrap" }}>
                          {curExpl.snippet || "—"}
                        </div>
                        <div className="control-row" style={{ marginTop: 8 }}>
                          <button
                            onClick={() =>
                              setExplIndex((i) => Math.max(0, i - 1))
                            }
                            disabled={explIndex <= 0}
                          >
                            Prev
                          </button>
                          <span>
                            {explIndex + 1}/{explExamples.length}
                          </span>
                          <button
                            onClick={() =>
                              setExplIndex((i) =>
                                Math.min(explExamples.length - 1, i + 1)
                              )
                            }
                            disabled={explIndex >= explExamples.length - 1}
                          >
                            Next
                          </button>
                        </div>
                      </div>
                    ) : (
                      <div>No example snippets stored.</div>
                    )}
                  </td>
                  <td style={{ verticalAlign: "top" }}>
                    {hasExpl ? (
                      <div>
                        {curExpl.quantized_score ?? "—"}
                      </div>
                    ) : (
                      "—"
                    )}
                  </td>
                  <td style={{ verticalAlign: "top" }}>
                    {hasEval ? (
                      <div>
                        <div style={{ whiteSpace: "pre-wrap" }}>
                          {curEval.snippet || "—"}
                        </div>
                        <div className="control-row" style={{ marginTop: 8 }}>
                          <button
                            onClick={() =>
                              setEvalIndex((i) => Math.max(0, i - 1))
                            }
                            disabled={evalIndex <= 0}
                          >
                            Prev
                          </button>
                          <span>
                            {evalIndex + 1}/{evalExamples.length}
                          </span>
                          <button
                            onClick={() =>
                              setEvalIndex((i) =>
                                Math.min(evalExamples.length - 1, i + 1)
                              )
                            }
                            disabled={evalIndex >= evalExamples.length - 1}
                          >
                            Next
                          </button>
                        </div>
                      </div>
                    ) : (
                      <div>No example snippets stored.</div>
                    )}
                  </td>
                  <td style={{ verticalAlign: "top" }}>
                    {hasEval ? curEval.quantized_true_score ?? "—" : "—"}
                  </td>
                  <td style={{ verticalAlign: "top" }}>
                    {hasEval ? curEval.predicted_score ?? "—" : "—"}
                  </td>
                </tr>
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

