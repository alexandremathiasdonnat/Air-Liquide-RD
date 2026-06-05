import { vnorm } from "./moe";

const REGIME_KEYS = ["wind_norm","mom_24","mom_48","vol_24","vol_48","trend_24_gap","hour_sin","hour_cos"];

function solveLinear(A, b) {
  const n = b.length;
  const M = A.map((row, i) => [...row, b[i]]);
  for (let col = 0; col < n; col++) {
    let max = col;
    for (let row = col + 1; row < n; row++) if (Math.abs(M[row][col]) > Math.abs(M[max][col])) max = row;
    [M[col], M[max]] = [M[max], M[col]];
    for (let row = col + 1; row < n; row++) {
      const f = M[row][col] / M[col][col];
      for (let j = col; j <= n; j++) M[row][j] -= f * M[col][j];
    }
  }
  const x = new Array(n).fill(0);
  for (let i = n - 1; i >= 0; i--) {
    x[i] = M[i][n];
    for (let j = i + 1; j < n; j++) x[i] -= M[i][j] * x[j];
    x[i] /= M[i][i];
  }
  return x;
}

// ── Linear Stacking ───────────────────────────────────────────────────────────
// OLS meta-model. Trained on all data except the last 24 observations (test holdout).

export function runLinearStacking(data, cols, params) {
  const K = cols.length;
  const trainEnd = Math.max(K + 2, data.length - 24);
  const trainData = data.slice(0, trainEnd);

  const X = trainData.map(r => cols.map(c => r[c] || 0));
  const y = trainData.map(r => r.y_true);

  const XtX = Array.from({ length: K }, (_, i) =>
    Array.from({ length: K }, (_, j) =>
      X.reduce((s, row) => s + row[i] * row[j], 0) + (i === j ? 1e-6 : 0)
    )
  );
  const Xty = Array.from({ length: K }, (_, i) =>
    X.reduce((s, row, t) => s + row[i] * y[t], 0)
  );

  let rawWeights;
  try { rawWeights = solveLinear(XtX, Xty); }
  catch { rawWeights = new Array(K).fill(1 / K); }

  const displayWeights = vnorm(rawWeights.map(Math.abs));
  const allX = data.map(r => cols.map(c => r[c] || 0));

  return {
    predictions: allX.map(row => row.reduce((s, v, k) => s + v * rawWeights[k], 0)),
    weightHistory: data.map(() => [...displayWeights]),
  };
}

// ── XGBoost Stacking ──────────────────────────────────────────────────────────
// Gradient boosted regression stumps trained on the first 70% of data.

function buildStump(X, residuals) {
  const F = X[0].length;
  let bestGain = Infinity;
  let best = { feature: 0, threshold: 0, left: 0, right: 0 };

  for (let f = 0; f < F; f++) {
    const vals = X.map(r => r[f]);
    const sorted = [...vals].sort((a, b) => a - b);
    // Sample at most 12 candidate thresholds to keep it fast
    const step = Math.max(1, Math.floor(sorted.length / 12));
    const checked = new Set();
    for (let i = 0; i < sorted.length - 1; i += step) {
      const thresh = (sorted[i] + sorted[i + 1]) / 2;
      if (checked.has(thresh)) continue;
      checked.add(thresh);
      const left = [], right = [];
      X.forEach((r, idx) => (r[f] <= thresh ? left : right).push(residuals[idx]));
      if (!left.length || !right.length) continue;
      const lm = left.reduce((s, v) => s + v, 0) / left.length;
      const rm = right.reduce((s, v) => s + v, 0) / right.length;
      const gain = left.reduce((s, v) => s + (v - lm) ** 2, 0) +
                   right.reduce((s, v) => s + (v - rm) ** 2, 0);
      if (gain < bestGain) { bestGain = gain; best = { feature: f, threshold: thresh, left: lm, right: rm }; }
    }
  }
  return best;
}

function predictStump(stump, row) {
  return row[stump.feature] <= stump.threshold ? stump.left : stump.right;
}

export function runXGBoostStacking(data, cols, params) {
  const K = cols.length;
  const nTrees = params.nTrees || 50;
  const lr = params.xgbLr || 0.1;
  const trainEnd = Math.min(Math.max(K + 2, Math.floor(data.length * 0.7)), data.length - 24);
  const trainData = data.slice(0, trainEnd);

  const activeRegimes = REGIME_KEYS.filter(k => trainData.some(r => (r[k] || 0) !== 0));
  const buildRow = r => [...cols.map(c => r[c] || 0), ...activeRegimes.map(k => r[k] || 0)];

  const trainX = trainData.map(buildRow);
  const trainY = trainData.map(r => r.y_true);
  const initMean = trainY.reduce((s, v) => s + v, 0) / trainY.length;

  const residuals = trainY.map(v => v - initMean);
  const trees = [];

  for (let iter = 0; iter < nTrees; iter++) {
    const stump = buildStump(trainX, residuals);
    trees.push(stump);
    for (let i = 0; i < trainData.length; i++) {
      residuals[i] -= lr * predictStump(stump, trainX[i]);
    }
  }

  const allX = data.map(buildRow);
  const predictions = allX.map(row => {
    let pred = initMean;
    for (const tree of trees) pred += lr * predictStump(tree, row);
    return pred;
  });

  // Feature importance: fraction of splits using each expert feature
  const featureCounts = new Array(K + activeRegimes.length).fill(0);
  trees.forEach(t => { featureCounts[t.feature]++; });
  const expertCounts = featureCounts.slice(0, K);
  const totalExpert = expertCounts.reduce((s, v) => s + v, 0);
  const displayWeights = totalExpert > 0
    ? vnorm(expertCounts.map(c => c + 1e-6))
    : new Array(K).fill(1 / K);

  return { predictions, weightHistory: data.map(() => [...displayWeights]) };
}
