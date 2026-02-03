import { Feature, FeatureInfo } from "./types";

const API_BASE =
  (process.env.FEATURE_SERVER_URL as string | undefined) || "http://127.0.0.1:8765";

export async function fetchFeatures(
  limit: number,
  metric: string
): Promise<{
  features: any[];
  metric: string;
  metrics_available: string[];
  metric_descriptions: Record<string, string>;
  accuracy?: number;
  num_samples?: number;
  num_features?: number;
  num_tokens?: number;
}> {
  const url = new URL(`${API_BASE}/api/features`);
  url.searchParams.set("limit", String(limit));
  url.searchParams.set("metric", metric);
  const res = await fetch(url.toString());
  if (!res.ok) {
    throw new Error(`Failed to load features: ${res.statusText}`);
  }
  return res.json();
}

// feature_info endpoint
export async function get_feature_info(
  feature: Feature,
  ablated?: boolean
): Promise<FeatureInfo> {
  const url = new URL(`${API_BASE}/api/feature_info`);
  url.searchParams.set("id", String(feature.atom));
  url.searchParams.set("top_k", "20");
  if (ablated) {
    url.searchParams.set("ablated", "1");
  }
  const res = await fetch(url.toString(), {
    headers: {
      "Accept": "application/json",
    },
  });
  if (!res.ok) {
    throw new Error(`Failed to load feature info: ${res.statusText}`);
  }
  return res.json();
}

export async function fetchHeadlines(
  limit: number = 100
): Promise<{
  headlines: any[];
}> {
  const url = new URL(`${API_BASE}/api/headlines`);
  url.searchParams.set("limit", String(limit));
  const res = await fetch(url.toString(), {
    headers: {
      "Accept": "application/json",
    },
  });
  if (!res.ok) {
    throw new Error(`Failed to load headlines: ${res.statusText}`);
  }
  return res.json();
}

export async function fetchMetadata(): Promise<{
  metadata: {
    ablated_features?: number[];
    accuracy?: number;
    num_samples?: number;
    run_id?: number;
    run_name?: string;
    [key: string]: any;
  };
}> {
  const url = new URL(`${API_BASE}/api/metadata`);
  const res = await fetch(url.toString(), {
    headers: {
      "Accept": "application/json",
    },
  });
  if (!res.ok) {
    throw new Error(`Failed to load metadata: ${res.statusText}`);
  }
  return res.json();
}
