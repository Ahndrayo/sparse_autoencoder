(() => {
  const { useEffect, useMemo, useState } = React;

  function useFeatures(limit, metric) {
    const [data, setData] = useState({
      features: [],
      metrics: [],
      metricDescriptions: {},
      activeMetric: metric,
      loading: true,
      error: null,
    });

    useEffect(() => {
      let cancelled = false;
      setData((prev) => ({ ...prev, loading: true, error: null }));
      fetch(`/api/features?limit=${limit}&metric=${encodeURIComponent(metric)}`)
        .then(async (res) => {
          if (!res.ok) {
            throw new Error((await res.json()).error || "Failed to load features");
          }
          return res.json();
        })
        .then((payload) => {
          if (!cancelled) {
            setData({
              features: payload.features,
              metrics: payload.metrics_available,
              metricDescriptions: payload.metric_descriptions,
              activeMetric: payload.metric,
              loading: false,
              error: null,
            });
          }
        })
        .catch((err) => {
          if (!cancelled) {
            setData({
              features: [],
              metrics: [],
              metricDescriptions: {},
              activeMetric: metric,
              loading: false,
              error: err.message,
            });
          }
        });
      return () => {
        cancelled = true;
      };
    }, [limit, metric]);

    return data;
  }

  function App() {
    const [limit, setLimit] = useState(50);
    const [metric, setMetric] = useState("mean_activation");
    const [selectedFeature, setSelectedFeature] = useState(null);
    const [topTokens, setTopTokens] = useState([]);
    const [tokenState, setTokenState] = useState({ loading: false, error: null });
    const [customFeatureId, setCustomFeatureId] = useState("");
    const [topK, setTopK] = useState(10);

    const { features, metrics, metricDescriptions, activeMetric, loading, error } = useFeatures(
      limit,
      metric
    );

    const fetchTokens = (featureId, overrideTopK) => {
      if (featureId == null) return;
      setTokenState({ loading: true, error: null });
      fetch(`/api/feature?id=${featureId}&top_k=${overrideTopK ?? topK}`)
        .then(async (res) => {
          if (!res.ok) {
            throw new Error((await res.json()).error || "Failed to load feature tokens");
          }
          return res.json();
        })
        .then((payload) => {
          setSelectedFeature(featureId);
          setTopTokens(payload.tokens);
          setTokenState({ loading: false, error: null });
        })
        .catch((err) => {
          setTopTokens([]);
          setTokenState({ loading: false, error: err.message });
        });
    };

    useEffect(() => {
      if (features.length > 0) {
        fetchTokens(features[0].feature_id);
      }
    }, [features]);

    useEffect(() => {
      if (selectedFeature != null) {
        fetchTokens(selectedFeature, topK);
      }
    }, [topK]);

    const featureRows = useMemo(() => {
      return features.map((feat) => {
        const metricsSnapshot = feat.metrics || {};
        const activeValue =
          feat.value ??
          metricsSnapshot[activeMetric] ??
          metricsSnapshot.mean_activation ??
          0;
        return React.createElement(
          "tr",
          {
            key: feat.feature_id,
            onClick: () => fetchTokens(feat.feature_id),
            className: feat.feature_id === selectedFeature ? "selected" : "",
          },
          React.createElement("td", null, feat.feature_id),
          React.createElement("td", null, activeValue.toFixed(4)),
          React.createElement(
            "td",
            null,
            metricsSnapshot.mean_activation?.toFixed
              ? metricsSnapshot.mean_activation.toFixed(4)
              : "—"
          ),
          React.createElement(
            "td",
            null,
            metricsSnapshot.max_activation?.toFixed
              ? metricsSnapshot.max_activation.toFixed(4)
              : "—"
          ),
          React.createElement(
            "td",
            null,
            metricsSnapshot.fraction_active?.toFixed
              ? metricsSnapshot.fraction_active.toFixed(3)
              : "—"
          )
        );
      });
    }, [features, selectedFeature, activeMetric]);

    const tokenList = useMemo(
      () =>
        topTokens.map((tok) =>
          React.createElement(
            "li",
            { key: `${tok.token_index}-${tok.activation}` },
            React.createElement(
              "div",
              { className: "token-meta" },
              React.createElement(
                "span",
                null,
                `Token #${tok.token_index} · prompt ${tok.prompt_index ?? "-"}`
              ),
              React.createElement("span", null, `act=${tok.activation.toFixed(4)}`)
            ),
            React.createElement(
              "div",
              { className: "token-snippet" },
              `${tok.token_str} — ${tok.prompt_snippet || ""}`
            )
          )
        ),
      [topTokens]
    );

    return React.createElement(
      "div",
      { className: "layout" },
      React.createElement(
        "section",
        { className: "card" },
        React.createElement("h2", null, "Top features"),
        React.createElement(
          "div",
          { className: "controls" },
          React.createElement(
            "label",
            null,
            "Show top N",
            React.createElement("input", {
              type: "number",
              min: 5,
              max: 200,
              value: limit,
              onChange: (e) => setLimit(Number(e.target.value)),
            })
          ),
          React.createElement(
            "label",
            null,
            "Metric",
            React.createElement(
              "select",
              {
                value: activeMetric,
                onChange: (e) => setMetric(e.target.value),
                disabled: loading || metrics.length === 0,
              },
              (metrics.length ? metrics : [activeMetric]).map((name) =>
                React.createElement(
                  "option",
                  { key: name, value: name },
                  name
                )
              )
            )
          ),
          React.createElement(
            "label",
            null,
            "Top tokens per feature",
            React.createElement("input", {
              type: "number",
              min: 3,
              max: 50,
              value: topK,
              onChange: (e) => setTopK(Number(e.target.value)),
            })
          )
        ),
        error &&
          React.createElement("div", { className: "error" }, `Failed to load features: ${error}`),
        activeMetric &&
          metricDescriptions[activeMetric] &&
          React.createElement(
            "p",
            { style: { fontSize: "0.85rem", color: "#475569", marginTop: -8 } },
            metricDescriptions[activeMetric]
          ),
        React.createElement(
          "div",
          { style: { maxHeight: 520, overflow: "auto" } },
          loading
            ? React.createElement("p", null, "Loading features…")
            : React.createElement(
                "table",
                null,
                React.createElement(
                  "thead",
                  null,
                  React.createElement(
                    "tr",
                    null,
                    React.createElement("th", null, "Feature"),
                    React.createElement(
                      "th",
                      null,
                      activeMetric
                        ? `Value (${activeMetric})`
                        : "Value"
                    ),
                    React.createElement("th", null, "Mean"),
                    React.createElement("th", null, "Max"),
                    React.createElement("th", null, "Frac>0")
                  )
                ),
                React.createElement("tbody", null, featureRows)
              )
        )
      ),
      React.createElement(
        "section",
        { className: "card" },
        React.createElement("h2", null, "Token viewers"),
        React.createElement(
          "div",
          { className: "controls" },
          React.createElement(
            "label",
            null,
            "Feature ID",
            React.createElement("input", {
              type: "number",
              placeholder: "12345",
              value: customFeatureId,
              onChange: (e) => setCustomFeatureId(e.target.value),
            })
          ),
          React.createElement(
            "button",
            {
              onClick: () => {
                if (customFeatureId !== "") {
                  fetchTokens(Number(customFeatureId));
                }
              },
              disabled: tokenState.loading,
            },
            tokenState.loading ? "Loading…" : "Inspect"
          )
        ),
        tokenState.error &&
          React.createElement("div", { className: "error" }, `Failed to load tokens: ${tokenState.error}`),
        selectedFeature != null &&
          React.createElement(
            "p",
            null,
            `Top tokens for feature ${selectedFeature} (top ${topK})`
          ),
        React.createElement(
          "ul",
          { className: "token-list" },
          tokenList.length > 0 ? tokenList : React.createElement("li", null, "No tokens yet.")
        )
      )
    );
  }

  ReactDOM.createRoot(document.getElementById("root")).render(React.createElement(App));
})();

