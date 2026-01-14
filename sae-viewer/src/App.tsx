import "./App.css";
import React, { useEffect, useState } from "react";
import FeaturesView from "./components/FeaturesView";
import HeadlinesView from "./components/HeadlinesView";

function App() {
  const [activeTab, setActiveTab] = useState<"headlines" | "features">("headlines");
  const [searchFeatureId, setSearchFeatureId] = useState<number | null>(null);

  // When a feature is selected from headlines, switch tab
  useEffect(() => {
    if (searchFeatureId !== null) {
      setActiveTab("features");
    }
  }, [searchFeatureId]);

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
            className={activeTab === "features" ? "active" : ""}
            onClick={() => setActiveTab("features")}
          >
            Features
          </button>
        </div>
      </header>

      {activeTab === "headlines" ? (
        <HeadlinesView onFeatureClick={(fid) => setSearchFeatureId(fid)} />
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
