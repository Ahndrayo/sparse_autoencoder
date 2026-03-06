import React, { useEffect, useRef, useState } from "react";
import { normalizeSequences, Feature, FeatureInfo, SequenceInfo } from "../types";
import TokenHeatmap from "./tokenHeatmap";
import TokenAblationmap from "./tokenAblationmap";
import Histogram from "./histogram";
import Tooltip from "./tooltip";

import { get_feature_info } from "../interpAPI";

const SPECIAL_TOKENS = new Set(["[CLS]", "[SEP]", "[PAD]", "[UNK]"]);

type PromptMode = "original" | "tokenized";

export default ({ feature }: { feature: Feature }) => {
  const [data, setData] = useState(null as FeatureInfo | null);
  const [showingMore, setShowingMore] = useState<Record<string, boolean>>({});
  const [renderNewlines, setRenderNewlines] = useState(false);
  const [promptRenderMode, setPromptRenderMode] = useState<PromptMode>("original");
  const [isLoading, setIsLoading] = useState(true);
  const [got_error, setError] = useState<any>(null);
  const currentFeatureRef = useRef(feature);

  const sanitizeTokens = (tokens: string[]) =>
    tokens.filter((token) => !SPECIAL_TOKENS.has(token));

  const clampHighlightIndex = (idx: number, length: number) => {
    if (length <= 0) return 0;
    return Math.max(0, Math.min(length - 1, idx));
  };

  const activationTokenForRow = (sequence: SequenceInfo) =>
    sequence.tokens?.[sequence.idx] ?? sequence.tokens?.[0] ?? "-";

  const renderOriginalPrompt = (sequence: SequenceInfo) => {
    if ((sequence as any).prompt) {
      return <span>{(sequence as any).prompt}</span>;
    }
    return <span>{(sequence as any).prompt_snippet || "-"}</span>;
  };

  const renderTokenizedPrompt = (sequence: SequenceInfo) => {
    const allTokens: string[] = (sequence as any).prompt_tokens || [];
    if (!allTokens.length) {
      return renderOriginalPrompt(sequence);
    }

    const rawHighlightIdx = clampHighlightIndex(sequence.idx ?? 0, allTokens.length);
    const removedBefore = allTokens
      .slice(0, rawHighlightIdx)
      .reduce((count, tok) => count + (SPECIAL_TOKENS.has(tok) ? 1 : 0), 0);

    const tokens = sanitizeTokens(allTokens);
    if (!tokens.length) {
      return renderOriginalPrompt(sequence);
    }

    const highlightIdx = clampHighlightIndex(rawHighlightIdx - removedBefore, tokens.length);
    const highlightedToken = tokens[highlightIdx] || "";

    const activationToken = activationTokenForRow(sequence);
    if (
      process.env.NODE_ENV !== "production" &&
      activationToken !== "-" &&
      highlightedToken &&
      activationToken !== highlightedToken
    ) {
      console.warn("[featureInfo] Prompt highlight token mismatch", {
        feature_id: feature.atom,
        doc_id: sequence.doc_id,
        idx: sequence.idx,
        activationToken,
        highlightedToken,
      });
    }

    const before = tokens.slice(0, highlightIdx).join(" ");
    const after = tokens.slice(highlightIdx + 1).join(" ");
    return (
      <span>
        {before}
        {before && " "}
        <span className="prompt-token highlight">{highlightedToken}</span>
        {after && " "}
        {after}
      </span>
    );
  };

  useEffect(() => {
    async function fetchData() {
      setIsLoading(true);
      try {
        currentFeatureRef.current = feature;
        const result = await get_feature_info(feature);
        if (currentFeatureRef.current !== feature) {
          return;
        }
        normalizeSequences(result.top, result.random);
        result.top.sort((a, b) => b.act - a.act);
        setData(result);
        setIsLoading(false);
        setError(null);
      } catch (e) {
        setError(e);
      }
      try {
        const result = await get_feature_info(feature, true);
        if (currentFeatureRef.current !== feature) {
          return;
        }
        normalizeSequences(result.top, result.random);
        result.top.sort((a, b) => b.act - a.act);
        setData(result);
        setIsLoading(false);
        setError(null);
      } catch (e) {
        setError("Note: ablation effects data not available for this model");
      }
    }
    fetchData();
  }, [feature]);

  if (isLoading) {
    return (
      <div className="flex justify-center items-center h-64">
        <div className="w-8 h-8 border-4 border-gray-300 rounded-full animate-spin"></div>
        <div>loading top dataset examples</div>
        {got_error ? <span style={{ color: "red" }}>Error loading data: {got_error}</span> : null}
      </div>
    );
  }
  if (!data) {
    throw new Error("no data. this should not happen.");
  }

  const all_sequences = [
    {
      label: "Random positive activations",
      sequences: data.random,
      default_show: 5,
    },
    {
      label: "Top activations",
      sequences: data.top,
      default_show: 5,
    },
  ];

  return (
    <div>
      {got_error ? <span style={{ color: "#ee5555" }}>{got_error}</span> : null}
      <div style={{ flexDirection: "row", display: "flex" }}>
        <div style={{ width: "500px", height: "200px" }}>
          <Histogram data={data.hist} />
        </div>
        <table style={{ marginLeft: "20px" }}>
          <tbody>
            <tr>
              <td>
                <Tooltip content={"Density"} tooltip={"E[a > 0]"} />
              </td>
              <td>{data.density.toExponential(2)}</td>
            </tr>
            <tr>
              <td>
                <Tooltip content={"Mean"} tooltip={"E[a]"} />
              </td>
              <td>{data.mean_act ? data.mean_act.toExponential(2) : "data not available"}</td>
            </tr>
            <tr>
              <td>
                <Tooltip content={"Variance (0 centered)"} tooltip={<>E[a<sup>2</sup>]</>} />
              </td>
              <td>{data.mean_act_squared ? data.mean_act_squared.toExponential(2) : "data not available"}</td>
            </tr>
            <tr>
              <td>
                <Tooltip
                  content={"Skew (0 centered)"}
                  tooltip={
                    <>
                      E[a<sup>3</sup>]/(E[a<sup>2</sup>])<sup>1.5</sup>
                    </>
                  }
                />
              </td>
              <td>{(data as any).skew ? (data as any).skew.toExponential(2) : "data not available"}</td>
            </tr>
            <tr>
              <td>
                <Tooltip
                  content={"Kurtosis (0 centered)"}
                  tooltip={
                    <>
                      E[a<sup>4</sup>]/(E[a<sup>2</sup>])<sup>2</sup>
                    </>
                  }
                />
              </td>
              <td>{(data as any).kurtosis ? (data as any).kurtosis.toExponential(2) : "data not available"}</td>
            </tr>
          </tbody>
        </table>
      </div>

      <div className="control-row prompt-view-controls">
        <span className="prompt-view-label">Prompt view:</span>
        <button
          className={promptRenderMode === "original" ? "active" : ""}
          onClick={() => setPromptRenderMode("original")}
          type="button"
        >
          Original
        </button>
        <button
          className={promptRenderMode === "tokenized" ? "active" : ""}
          onClick={() => setPromptRenderMode("tokenized")}
          type="button"
        >
          Tokenized (Exact)
        </button>
        <span
          className="prompt-view-hint"
          title="Tokenized mode guarantees exact activation-token highlight."
        >
          Tokenized mode guarantees exact activation-token highlight.
        </span>
      </div>

      {all_sequences.map(({ label, sequences, default_show }, idx) => {
        const n_show = showingMore[label] ? sequences.length : default_show;
        return (
          <React.Fragment key={idx}>
            <h3 className="text-md font-bold">
              {label}
              <button
                className="ml-2 mb-2 mt-2 text-sm text-gray-500"
                onClick={() => setShowingMore({ ...showingMore, [label]: !showingMore[label] })}
              >
                {showingMore[label] ? "show less" : "show more"}
              </button>
              <button
                className="ml-2 mb-2 mt-2 text-sm text-gray-500"
                onClick={() => setRenderNewlines(!renderNewlines)}
              >
                {renderNewlines ? "collapse newlines" : "show newlines"}
              </button>
            </h3>
            <table style={{ fontSize: "12px" }} className="activations-table">
              <thead>
                <tr>
                  <th>Doc ID</th>
                  <th>Token</th>
                  <th>Activation</th>
                  <th>Prediction</th>
                  <th>Prompt</th>
                  <th>Activations</th>
                  {sequences.length && (sequences[0] as any).ablate_loss_diff && <th>Effects</th>}
                </tr>
              </thead>
              <tbody>
                {sequences.slice(0, n_show).map((sequence, i) => (
                  <tr key={i}>
                    <td className="center">{sequence.doc_id}</td>
                    <td className="center">{activationTokenForRow(sequence)}</td>
                    <td className="center">{sequence.act.toFixed(2)}</td>
                    <td
                      className="center"
                      style={{
                        color: (sequence as any).predicted_label === (sequence as any).true_label ? "#22c55e" : "#ef4444",
                        fontWeight: "bold",
                      }}
                    >
                      {(sequence as any).predicted_label || "-"}
                      {(sequence as any).true_label && (sequence as any).predicted_label !== (sequence as any).true_label && (
                        <span style={{ fontSize: "10px", color: "#888" }}>
                          {" "}(true: {(sequence as any).true_label})
                        </span>
                      )}
                    </td>
                    <td className="prompt-cell p-2">
                      <div className="prompt-inline">
                        {promptRenderMode === "tokenized"
                          ? renderTokenizedPrompt(sequence)
                          : renderOriginalPrompt(sequence)}
                      </div>
                    </td>
                    <td className="p-2">
                      <TokenHeatmap info={sequence} renderNewlines={renderNewlines} />
                    </td>
                    {(sequence as any).ablate_loss_diff && (
                      <td className="p-2">
                        <TokenAblationmap info={sequence} renderNewlines={renderNewlines} />
                      </td>
                    )}
                  </tr>
                ))}
              </tbody>
            </table>
          </React.Fragment>
        );
      })}
    </div>
  );
};
