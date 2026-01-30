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

  // When a feature is selected from headlines, switch tab
  useEffect(() => {
    if (searchFeatureId !== null) {
      setActiveTab("features");
    }
  }, [searchFeatureId]);

  useEffect(() => {
    fetchMetadata()
      .then((data) => {
        setHasAblationData(data.metadata.ablation_mode !== undefined);
      })
      .catch(() => setHasAblationData(false));
  }, []);

  return (
    <div className="app-shell tabs-shell">
      <header className="app-header">
        <h1>SAE Feature Explorer</h1>
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
        <HeadlinesView onFeatureClick={(fid) => setSearchFeatureId(fid)} />
      ) : activeTab === "ablation" ? (
        <AblationView onFeatureClick={(fid) => setSearchFeatureId(fid)} />
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
