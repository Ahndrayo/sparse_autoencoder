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

  type SortKey =
    | "expl_snippet"
    | "expl_score"
    | "eval_snippet"
    | "true_score"
    | "predicted_score"
    | "difference";
  const [sortKey, setSortKey] = useState<SortKey>("difference");
  const [sortDir, setSortDir] = useState<"asc" | "desc">("desc");

  const sortedRows = useMemo(() => {
    const rows = Array.from({ length: maxExplEvalRows }).map((_, i) => {
      const expl = explExamples[i] ?? null;
      const ev = evalExamples[i] ?? null;
      const trueScore = ev?.quantized_true_score ?? null;
      const predScore = ev?.predicted_score ?? null;
      const difference =
        typeof trueScore === "number" && typeof predScore === "number"
          ? Math.abs(predScore - trueScore)
          : null;

      return {
        i,
        expl,
        ev,
        difference,
      };
    });

    const dirMul = sortDir === "asc" ? 1 : -1;

    const getNum = (x: unknown): number | null => {
      if (typeof x === "number") return x;
      if (x === null || x === undefined) return null;
      const n = Number(x);
      return Number.isFinite(n) ? n : null;
    };

    const compare = (a: typeof rows[number], b: typeof rows[number]) => {
      const aVal =
        sortKey === "expl_snippet"
          ? (a.expl?.snippet ?? "").toString()
          : sortKey === "eval_snippet"
          ? (a.ev?.snippet ?? "").toString()
          : sortKey === "expl_score"
          ? getNum(a.expl?.quantized_score)
          : sortKey === "true_score"
          ? getNum(a.ev?.quantized_true_score)
          : sortKey === "predicted_score"
          ? getNum(a.ev?.predicted_score)
          : getNum(a.difference);
      const bVal =
        sortKey === "expl_snippet"
          ? (b.expl?.snippet ?? "").toString()
          : sortKey === "eval_snippet"
          ? (b.ev?.snippet ?? "").toString()
          : sortKey === "expl_score"
          ? getNum(b.expl?.quantized_score)
          : sortKey === "true_score"
          ? getNum(b.ev?.quantized_true_score)
          : sortKey === "predicted_score"
          ? getNum(b.ev?.predicted_score)
          : getNum(b.difference);

      if (typeof aVal === "string" && typeof bVal === "string") {
        return dirMul * aVal.localeCompare(bVal);
      }

      const aNum = typeof aVal === "number" ? aVal : null;
      const bNum = typeof bVal === "number" ? bVal : null;

      if (aNum === null && bNum === null) return a.i - b.i;
      if (aNum === null) return 1;
      if (bNum === null) return -1;
      return dirMul * (aNum - bNum);
    };

    return rows.slice().sort(compare);
  }, [
    evalExamples,
    explExamples,
    maxExplEvalRows,
    sortDir,
    sortKey,
  ]);

  const sortIndicator = (k: SortKey) => {
    if (sortKey !== k) return "";
    return sortDir === "asc" ? " ▲" : " ▼";
  };

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
                  <th
                    style={{ cursor: "pointer" }}
                    onClick={() => {
                      if (sortKey === "expl_snippet") {
                        setSortDir((d) => (d === "asc" ? "desc" : "asc"));
                      } else {
                        setSortKey("expl_snippet");
                        setSortDir("asc");
                      }
                    }}
                  >
                    Explanation Sample{sortIndicator("expl_snippet")}
                  </th>
                  <th
                    style={{ cursor: "pointer" }}
                    onClick={() => {
                      if (sortKey === "expl_score") {
                        setSortDir((d) => (d === "asc" ? "desc" : "asc"));
                      } else {
                        setSortKey("expl_score");
                        setSortDir("desc");
                      }
                    }}
                  >
                    Score{sortIndicator("expl_score")}
                  </th>
                  <th
                    style={{ cursor: "pointer" }}
                    onClick={() => {
                      if (sortKey === "eval_snippet") {
                        setSortDir((d) => (d === "asc" ? "desc" : "asc"));
                      } else {
                        setSortKey("eval_snippet");
                        setSortDir("asc");
                      }
                    }}
                  >
                    Evaluation Sample{sortIndicator("eval_snippet")}
                  </th>
                  <th
                    style={{ cursor: "pointer" }}
                    onClick={() => {
                      if (sortKey === "true_score") {
                        setSortDir((d) => (d === "asc" ? "desc" : "asc"));
                      } else {
                        setSortKey("true_score");
                        setSortDir("desc");
                      }
                    }}
                  >
                    True{sortIndicator("true_score")}
                  </th>
                  <th
                    style={{ cursor: "pointer" }}
                    onClick={() => {
                      if (sortKey === "predicted_score") {
                        setSortDir((d) => (d === "asc" ? "desc" : "asc"));
                      } else {
                        setSortKey("predicted_score");
                        setSortDir("desc");
                      }
                    }}
                  >
                    Pred{sortIndicator("predicted_score")}
                  </th>
                  <th
                    style={{ cursor: "pointer" }}
                    onClick={() => {
                      if (sortKey === "difference") {
                        setSortDir((d) => (d === "asc" ? "desc" : "asc"));
                      } else {
                        setSortKey("difference");
                        setSortDir("desc");
                      }
                    }}
                  >
                    Difference{sortIndicator("difference")}
                  </th>
                </tr>
              </thead>
              <tbody>
                {maxExplEvalRows === 0 ? (
                  <tr>
                    <td colSpan={6}>No example snippets stored.</td>
                  </tr>
                ) : (
                  sortedRows.map((row) => {
                    const expl = row.expl;
                    const ev = row.ev;
                    const diff = row.difference;
                    return (
                      <tr key={row.i}>
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
                        <td style={{ verticalAlign: "top" }}>
                          {diff === null || diff === undefined ? "—" : diff}
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

