import React from "react";
import { HeadlineRecord } from "../types";

type Props = {
  records: HeadlineRecord[];
  isLoading?: boolean;
  error?: string | null;
};

function FeatureChip({
  feature_id,
  activation,
  token_str,
}: {
  feature_id: number;
  activation: number;
  token_str?: string;
}) {
  const intensity = Math.min(1, activation / 5);
  const bg = `rgba(37, 99, 235, ${0.08 + 0.3 * intensity})`;
  return (
    <span className="feature-chip" style={{ background: bg }}>
      <span className="feature-chip-id">#{feature_id}</span>
      <span className="feature-chip-activation">{activation.toFixed(2)}</span>
      {token_str ? <span className="feature-chip-token">{token_str}</span> : null}
    </span>
  );
}

export default function HeadlineFeatureView({ records, isLoading, error }: Props) {
  if (isLoading) {
    return <div>Loading headlines…</div>;
  }
  if (error) {
    return <div className="error-message">{error}</div>;
  }
  if (!records.length) {
    return <div>No headline activations available.</div>;
  }

  return (
    <div className="headline-list">
      {records.map((rec) => (
        <div className="headline-card" key={rec.row_id ?? rec.prompt}>
          <div className="headline-header">
            <div className="headline-meta">
              <span className="headline-id">#{rec.row_id ?? "?"}</span>
              <span
                className={`headline-label ${
                  rec.correct ? "label-correct" : "label-incorrect"
                }`}
              >
                {rec.predicted_label ?? "—"}
                {rec.true_label && rec.true_label !== rec.predicted_label ? (
                  <span className="headline-true"> (true: {rec.true_label})</span>
                ) : null}
              </span>
            </div>
          </div>
          <div className="headline-text">{rec.prompt}</div>
          <div className="headline-features">
            {rec.features.map((feat) => (
              <FeatureChip
                key={`${rec.row_id}-${feat.feature_id}-${feat.token_position ?? 0}`}
                feature_id={feat.feature_id}
                activation={feat.activation}
                token_str={feat.token_str}
              />
            ))}
          </div>
        </div>
      ))}
    </div>
  );
}
import React from "react";
import { HeadlineRecord } from "../types";

type Props = {
  records: HeadlineRecord[];
  isLoading?: boolean;
  error?: string | null;
};

function FeatureChip({
  feature_id,
  activation,
  token_str,
}: {
  feature_id: number;
  activation: number;
  token_str?: string;
}) {
  const intensity = Math.min(1, activation / 5); // quick normalization for color strength
  const bg = `rgba(37, 99, 235, ${0.08 + 0.3 * intensity})`;
  return (
    <span className="feature-chip" style={{ background: bg }}>
      <span className="feature-chip-id">#{feature_id}</span>
      <span className="feature-chip-activation">{activation.toFixed(2)}</span>
      {token_str ? <span className="feature-chip-token">{token_str}</span> : null}
    </span>
  );
}

export default function HeadlineFeatureView({ records, isLoading, error }: Props) {
  if (isLoading) {
    return <div>Loading headlines…</div>;
  }
  if (error) {
    return <div className="error-message">{error}</div>;
  }
  if (!records.length) {
    return <div>No headline activations available.</div>;
  }

  return (
    <div className="headline-list">
      {records.map((rec) => (
        <div className="headline-card" key={rec.row_id ?? rec.prompt}>
          <div className="headline-header">
            <div className="headline-meta">
              <span className="headline-id">#{rec.row_id ?? "?"}</span>
              <span
                className={`headline-label ${
                  rec.correct ? "label-correct" : "label-incorrect"
                }`}
              >
                {rec.predicted_label ?? "—"}
                {rec.true_label && rec.true_label !== rec.predicted_label ? (
                  <span className="headline-true"> (true: {rec.true_label})</span>
                ) : null}
              </span>
            </div>
          </div>
          <div className="headline-text">{rec.prompt}</div>
          <div className="headline-features">
            {rec.features.map((feat) => (
              <FeatureChip
                key={`${rec.row_id}-${feat.feature_id}-${feat.token_position ?? 0}`}
                feature_id={feat.feature_id}
                activation={feat.activation}
                token_str={feat.token_str}
              />
            ))}
          </div>
        </div>
      ))}
    </div>
  );
}


