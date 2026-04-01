import "./App.css";
import React, { useEffect, useState } from "react";
import FeaturesView from "./components/FeaturesView";
import HeadlinesView from "./components/HeadlinesView";
import AblationView from "./components/AblationView";
import InterpretedView from "./components/InterpretedView";
import { fetchMetadata } from "./interpAPI";

function App() {
  const [
    activeTab,
    setActiveTab,
  ] = useState<"headlines" | "ablation" | "features" | "interpreted">("headlines");
  const [searchFeatureId, setSearchFeatureId] = useState<number | null>(null);
  const [hasAblationData, setHasAblationData] = useState(false);
  const [hasBaselineHeadlines, setHasBaselineHeadlines] = useState(true);
  const [runLabel, setRunLabel] = useState<string | null>(null);
  const [baselineAccuracy, setBaselineAccuracy] = useState<number | null>(null);
  const [ablatedAccuracy, setAblatedAccuracy] = useState<number | null>(null);
  const [runSampleCount, setRunSampleCount] = useState<number | null>(null);

  // When a feature is selected from headlines, switch tab
  useEffect(() => {
    if (searchFeatureId !== null) {
      setActiveTab("features");
    }
  }, [searchFeatureId]);

  useEffect(() => {
    fetchMetadata()
      .then((data) => {
        const metadata = data.metadata || {};
        const isAblationRun = metadata.ablation_mode !== undefined;
        setHasAblationData(isAblationRun);
        setHasBaselineHeadlines(
          isAblationRun ? metadata.has_baseline_headlines === true : true
        );
        if (metadata.run_id !== undefined) {
          setRunLabel(`run-${String(metadata.run_id).padStart(3, "0")}`);
        } else if (metadata.run_name) {
          setRunLabel(metadata.run_name);
        } else {
          setRunLabel(null);
        }
        if (isAblationRun) {
          setBaselineAccuracy(
            typeof metadata.baseline_accuracy === "number"
              ? metadata.baseline_accuracy
              : null
          );
          setAblatedAccuracy(
            typeof metadata.ablated_accuracy === "number"
              ? metadata.ablated_accuracy
              : typeof metadata.accuracy === "number"
              ? metadata.accuracy
              : null
          );
        } else {
          const nonAblatedAccuracy =
            typeof metadata.accuracy === "number" ? metadata.accuracy : null;
          setBaselineAccuracy(nonAblatedAccuracy);
          setAblatedAccuracy(nonAblatedAccuracy);
        }
        setRunSampleCount(
          typeof metadata.num_samples === "number" ? metadata.num_samples : null
        );
      })
      .catch(() => {
        setHasAblationData(false);
        setHasBaselineHeadlines(true);
        setRunLabel(null);
        setBaselineAccuracy(null);
        setAblatedAccuracy(null);
        setRunSampleCount(null);
      });
  }, []);

  return (
    <div className="app-shell tabs-shell">
      <header className="app-header">
        <div>
          <h1>SAE Feature Explorer</h1>
          {runLabel && (
            <div className="run-label">
              Run: <strong>{runLabel}</strong>
            </div>
          )}
        </div>
        <div className="tab-buttons">
          <button
            className={activeTab === "headlines" ? "active" : ""}
            onClick={() => setActiveTab("headlines")}
          >
            Unablated
          </button>
          <button
            className={activeTab === "ablation" ? "active" : ""}
            onClick={() => setActiveTab("ablation")}
            disabled={!hasAblationData}
            title={!hasAblationData ? "No ablation data for this run" : ""}
            style={{
              opacity: hasAblationData ? 1 : 0.5,
              cursor: hasAblationData ? "pointer" : "not-allowed",
            }}
          >
            Ablated
          </button>
          <button
            className={activeTab === "features" ? "active" : ""}
            onClick={() => setActiveTab("features")}
          >
            Features
          </button>
          <button
            className={activeTab === "interpreted" ? "active" : ""}
            onClick={() => setActiveTab("interpreted")}
          >
            Interpreted
          </button>
        </div>
      </header>

      {activeTab === "headlines" ? (
        <HeadlinesView
          onFeatureClick={(fid) => setSearchFeatureId(fid)}
          accuracy={baselineAccuracy}
          numSamples={runSampleCount}
          isAblationRun={hasAblationData}
          hasBaselineHeadlines={hasBaselineHeadlines}
        />
      ) : activeTab === "ablation" ? (
        <AblationView
          onFeatureClick={(fid) => setSearchFeatureId(fid)}
          accuracy={ablatedAccuracy}
          numSamples={runSampleCount}
        />
      ) : activeTab === "interpreted" ? (
        <InterpretedView />
      ) : (
        <FeaturesView
          initialFeatureId={searchFeatureId}
          onClearSearch={() => setSearchFeatureId(null)}
        />
      )}
    </div>
  );
}

export default App;
