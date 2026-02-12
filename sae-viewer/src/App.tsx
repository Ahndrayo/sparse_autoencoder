import "./App.css";
import React, { useEffect, useState } from "react";
import FeaturesView from "./components/FeaturesView";
import HeadlinesView from "./components/HeadlinesView";
import AblationView from "./components/AblationView";
import { fetchMetadata } from "./interpAPI";

function App() {
  const [activeTab, setActiveTab] = useState<"headlines" | "ablation" | "features">("headlines");
  const [searchFeatureId, setSearchFeatureId] = useState<number | null>(null);
  const [hasAblationData, setHasAblationData] = useState(false);
  const [runLabel, setRunLabel] = useState<string | null>(null);
  const [runAccuracy, setRunAccuracy] = useState<number | null>(null);
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
        setHasAblationData(metadata.ablation_mode !== undefined);
        if (metadata.run_id !== undefined) {
          setRunLabel(`run-${String(metadata.run_id).padStart(3, "0")}`);
        } else if (metadata.run_name) {
          setRunLabel(metadata.run_name);
        } else {
          setRunLabel(null);
        }
        setRunAccuracy(
          typeof metadata.accuracy === "number" ? metadata.accuracy : null
        );
        setRunSampleCount(
          typeof metadata.num_samples === "number" ? metadata.num_samples : null
        );
      })
      .catch(() => {
        setHasAblationData(false);
        setRunLabel(null);
        setRunAccuracy(null);
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
            Headlines
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
            Ablation
          </button>
          <button
            className={activeTab === "features" ? "active" : ""}
            onClick={() => setActiveTab("features")}
          >
            Features
          </button>
        </div>
      </header>

      {activeTab === "headlines" ? (
        <HeadlinesView
          onFeatureClick={(fid) => setSearchFeatureId(fid)}
          accuracy={runAccuracy}
          numSamples={runSampleCount}
        />
      ) : activeTab === "ablation" ? (
        <AblationView
          onFeatureClick={(fid) => setSearchFeatureId(fid)}
          accuracy={runAccuracy}
          numSamples={runSampleCount}
        />
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
