// Pure-JS neural aggregation methods (no external ML dependencies).
// All matrix ops are explicit loops to keep this browser-compatible.

const REGIME_KEYS = ["wind_norm","mom_24","mom_48","vol_24","vol_48","trend_24_gap","hour_sin","hour_cos"];

// ── Activations ───────────────────────────────────────────────────────────────

function relu(x) { return x > 0 ? x : 0; }
function sigmoidFn(x) { const c = Math.max(-15, Math.min(15, x)); return 1 / (1 + Math.exp(-c)); }
function tanhFn(x) { return Math.tanh(Math.max(-15, Math.min(15, x))); }

function softmax(arr) {
  const max = arr.reduce((m, v) => Math.max(m, v), -Infinity);
  const e = arr.map(v => Math.exp(Math.min(v - max, 100)));
  const s = e.reduce((a, b) => a + b, 0) + 1e-10;
  return e.map(v => v / s);
}

// ── Linear algebra helpers ────────────────────────────────────────────────────

// Matrix-vector product: W(m,n) × x(n) → y(m)
function mv(W, x) {
  const m = W.length, n = x.length;
  const y = new Array(m).fill(0);
  for (let i = 0; i < m; i++) for (let j = 0; j < n; j++) y[i] += W[i][j] * x[j];
  return y;
}

// Matrix transpose: W(m,n) → W^T(n,m)
function transpose(W) {
  const m = W.length, n = W[0].length;
  return Array.from({ length: n }, (_, j) => Array.from({ length: m }, (_, i) => W[i][j]));
}

function addVec(a, b) { return a.map((v, i) => v + b[i]); }
function scaleVec(a, s) { return a.map(v => v * s); }
function scaleMat(A, s) { return A.map(row => row.map(v => v * s)); }
function zeroVec(n) { return new Array(n).fill(0); }
function zeroMat(m, n) { return Array.from({ length: m }, () => new Array(n).fill(0)); }

function xavierMatrix(m, n) {
  const scale = Math.sqrt(6 / (m + n));
  return Array.from({ length: m }, () =>
    Array.from({ length: n }, () => (Math.random() * 2 - 1) * scale)
  );
}

// ── Feature scaler (fit on train, apply to all) ───────────────────────────────

function fitScaler(samples) {
  const d = samples[0].length, n = samples.length;
  const mean = zeroVec(d);
  for (const s of samples) for (let j = 0; j < d; j++) mean[j] += s[j] / n;
  const std = zeroVec(d);
  for (const s of samples) for (let j = 0; j < d; j++) std[j] += (s[j] - mean[j]) ** 2;
  for (let j = 0; j < d; j++) std[j] = Math.sqrt(std[j] / n + 1e-8);
  return { mean, std };
}

function applyScaler(x, sc) { return x.map((v, j) => (v - sc.mean[j]) / sc.std[j]); }

// ── Adam optimizer state ──────────────────────────────────────────────────────

function makeState(p) {
  const is2D = Array.isArray(p[0]);
  return {
    m: is2D ? p.map(r => zeroVec(r.length)) : zeroVec(p.length),
    v: is2D ? p.map(r => zeroVec(r.length)) : zeroVec(p.length),
  };
}

function adamStep(param, grad, state, lr, t, b1 = 0.9, b2 = 0.999, eps = 1e-8) {
  const bc1 = 1 - b1 ** t, bc2 = 1 - b2 ** t;
  if (Array.isArray(param[0])) {
    for (let i = 0; i < param.length; i++)
      for (let j = 0; j < param[i].length; j++) {
        state.m[i][j] = b1 * state.m[i][j] + (1 - b1) * grad[i][j];
        state.v[i][j] = b2 * state.v[i][j] + (1 - b2) * grad[i][j] ** 2;
        param[i][j] -= lr * (state.m[i][j] / bc1) / (Math.sqrt(state.v[i][j] / bc2) + eps);
      }
  } else {
    for (let i = 0; i < param.length; i++) {
      state.m[i] = b1 * state.m[i] + (1 - b1) * grad[i];
      state.v[i] = b2 * state.v[i] + (1 - b2) * grad[i] ** 2;
      param[i] -= lr * (state.m[i] / bc1) / (Math.sqrt(state.v[i] / bc2) + eps);
    }
  }
}

// ── MLP Stacking ──────────────────────────────────────────────────────────────
//
// Architecture: Input → Dense(16, ReLU) → Dense(8, ReLU) → Dense(K, Softmax)
// Trains on the full dataset, then averages the softmax outputs to produce
// fixed global weights — same philosophy as Ridge (100% offline, fixed weights).

export async function runMLPStacking(data, cols, params, onProgress) {
  const K = cols.length;
  const epochs = Math.max(30, Math.min(80, params.mlpEpochs || 50));
  const lr = 1e-3;
  const trainEnd = Math.max(K + 2, data.length - 24);
  const H1 = 16, H2 = 8;

  const activeRegimes = REGIME_KEYS.filter(k => data.some(r => (r[k] || 0) !== 0));
  const dIn = K + activeRegimes.length;

  function getFeatures(row) {
    return [...cols.map(c => row[c] || 0), ...activeRegimes.map(k => row[k] || 0)];
  }

  // Scaler fitted on training split only to avoid leakage
  const trainFeats = data.slice(0, trainEnd).map(getFeatures);
  const scaler = fitScaler(trainFeats);
  const allX = data.map(r => applyScaler(getFeatures(r), scaler));
  const trainX = allX.slice(0, trainEnd);
  const trainY = data.slice(0, trainEnd).map(r => r.y_true);
  const trainE = data.slice(0, trainEnd).map(r => cols.map(c => r[c] || 0));

  // Parameters
  const W1 = xavierMatrix(H1, dIn), b1 = zeroVec(H1);
  const W2 = xavierMatrix(H2, H1), b2 = zeroVec(H2);
  const W3 = xavierMatrix(K, H2),  b3 = zeroVec(K);
  const sW1 = makeState(W1), sb1 = makeState(b1);
  const sW2 = makeState(W2), sb2 = makeState(b2);
  const sW3 = makeState(W3), sb3 = makeState(b3);

  const n = trainX.length;

  for (let epoch = 0; epoch < epochs; epoch++) {
    const dW1 = zeroMat(H1, dIn), db1g = zeroVec(H1);
    const dW2 = zeroMat(H2, H1),  db2g = zeroVec(H2);
    const dW3 = zeroMat(K, H2),   db3g = zeroVec(K);

    for (let t = 0; t < n; t++) {
      const x = trainX[t];
      const e = trainE[t];
      const yTrue = trainY[t];

      // Forward
      const pre1 = addVec(mv(W1, x), b1);
      const h1 = pre1.map(relu);
      const pre2 = addVec(mv(W2, h1), b2);
      const h2 = pre2.map(relu);
      const logits = addVec(mv(W3, h2), b3);
      const w = softmax(logits);
      const yHat = w.reduce((s, wi, k) => s + wi * e[k], 0);

      // Backward — softmax + output
      const dOut = 2 * (yHat - yTrue);
      const dLogits = w.map((wi, i) => dOut * wi * (e[i] - yHat));

      for (let i = 0; i < K; i++) {
        for (let j = 0; j < H2; j++) dW3[i][j] += dLogits[i] * h2[j];
        db3g[i] += dLogits[i];
      }

      // Backprop → h2
      const dh2 = zeroVec(H2);
      for (let j = 0; j < H2; j++) for (let i = 0; i < K; i++) dh2[j] += W3[i][j] * dLogits[i];
      const dp2 = dh2.map((v, j) => v * (pre2[j] > 0 ? 1 : 0));

      for (let i = 0; i < H2; i++) {
        for (let j = 0; j < H1; j++) dW2[i][j] += dp2[i] * h1[j];
        db2g[i] += dp2[i];
      }

      // Backprop → h1
      const dh1 = zeroVec(H1);
      for (let j = 0; j < H1; j++) for (let i = 0; i < H2; i++) dh1[j] += W2[i][j] * dp2[i];
      const dp1 = dh1.map((v, j) => v * (pre1[j] > 0 ? 1 : 0));

      for (let i = 0; i < H1; i++) {
        for (let j = 0; j < dIn; j++) dW1[i][j] += dp1[i] * x[j];
        db1g[i] += dp1[i];
      }
    }

    // Average + Adam
    const step = epoch + 1;
    adamStep(W1, scaleMat(dW1, 1 / n), sW1, lr, step);
    adamStep(b1, scaleVec(db1g, 1 / n), sb1, lr, step);
    adamStep(W2, scaleMat(dW2, 1 / n), sW2, lr, step);
    adamStep(b2, scaleVec(db2g, 1 / n), sb2, lr, step);
    adamStep(W3, scaleMat(dW3, 1 / n), sW3, lr, step);
    adamStep(b3, scaleVec(db3g, 1 / n), sb3, lr, step);
    if (onProgress) {
      onProgress({ epoch: epoch + 1, total: epochs });
      await new Promise(r => setTimeout(r, 0));
    }
  }

  // Derive fixed offline weights: average softmax outputs over the training set
  const wSum = zeroVec(K);
  for (let t = 0; t < trainEnd; t++) {
    const x = allX[t];
    const h1 = addVec(mv(W1, x), b1).map(relu);
    const h2 = addVec(mv(W2, h1), b2).map(relu);
    const w = softmax(addVec(mv(W3, h2), b3));
    for (let k = 0; k < K; k++) wSum[k] += w[k];
  }
  const wFixed = wSum.map(v => v / trainEnd);

  // Apply fixed weights to all data (constant — no leakage on test period)
  const predictions = data.map(row => cols.reduce((s, c, k) => s + (row[c] || 0) * wFixed[k], 0));
  const weightHistory = data.map(() => [...wFixed]);
  return { predictions, weightHistory };
}

// ── GRU Aggregator ────────────────────────────────────────────────────────────
//
// Processes sliding windows of length seqLen to capture temporal memory.
// Architecture: GRU(H=8) → Dense(K, Softmax).
// Trained on first 70% of data, sequences sampled with stride 2.

export async function runGRUAggregator(data, cols, params, onProgress) {
  const K = cols.length;
  const epochs = Math.max(5, Math.min(80, params.gruEpochs || 30));
  const lr = 1e-3;
  const H = 8;
  // trainEnd capped at data.length; seqLen capped so at least 4 training sequences exist
  const rawTrainEnd = Math.min(data.length - 24, Math.max(K + 4, Math.floor(data.length * 0.7)));
  const maxSeqLen = Math.max(2, Math.floor(rawTrainEnd / 4));
  const seqLen = Math.max(2, Math.min(maxSeqLen, params.gruSeqLen || 8)); // hard cap = min(maxSeqLen, 725)
  const trainEnd = rawTrainEnd;

  const activeRegimes = REGIME_KEYS.filter(k => data.some(r => (r[k] || 0) !== 0));
  const dIn = K + activeRegimes.length;
  const dXH = dIn + H; // concatenated input dim for GRU gates

  function getFeatures(row) {
    return [...cols.map(c => row[c] || 0), ...activeRegimes.map(k => row[k] || 0)];
  }

  const allFeatRaw = data.map(getFeatures);
  const scaler = fitScaler(allFeatRaw.slice(0, trainEnd));
  const allX = allFeatRaw.map(f => applyScaler(f, scaler));

  // GRU parameters — gates use [h, x] concatenation
  const W_r = xavierMatrix(H, dXH), b_r = zeroVec(H);
  const W_z = xavierMatrix(H, dXH), b_z = zeroVec(H);
  const W_n = xavierMatrix(H, dXH), b_n = zeroVec(H);
  const W_o = xavierMatrix(K, H),   b_o = zeroVec(K);

  const sWr = makeState(W_r), sbr = makeState(b_r);
  const sWz = makeState(W_z), sbz = makeState(b_z);
  const sWn = makeState(W_n), sbn = makeState(b_n);
  const sWo = makeState(W_o), sbo = makeState(b_o);

  // Forward through a sequence of feature vectors; records all cell states for BPTT
  function gruSeqForward(xSeq) {
    let h = zeroVec(H);
    const states = [];
    for (const x of xSeq) {
      const hPrev = [...h];
      const xh = [...hPrev, ...x];
      const r = addVec(mv(W_r, xh), b_r).map(sigmoidFn);
      const z = addVec(mv(W_z, xh), b_z).map(sigmoidFn);
      const rh = r.map((ri, i) => ri * hPrev[i]);
      const xrh = [...rh, ...x];
      const n = addVec(mv(W_n, xrh), b_n).map(tanhFn);
      h = hPrev.map((hi, i) => (1 - z[i]) * hi + z[i] * n[i]);
      states.push({ hPrev, r, z, n, xh, xrh });
    }
    return { states, hFinal: h };
  }

  // BPTT — accumulates gradients directly into caller-provided accumulators
  function gruBptt(states, dhFinal, dWr, dbr, dWz, dbz, dWn, dbn) {
    let dh = [...dhFinal];
    for (let t = states.length - 1; t >= 0; t--) {
      const { hPrev, r, z, n, xh, xrh } = states[t];

      // h_new = (1-z)*hPrev + z*n
      const dN = dh.map((v, i) => v * z[i]);
      const dZ = dh.map((v, i) => v * (n[i] - hPrev[i]));
      const dhPrevDirect = dh.map((v, i) => v * (1 - z[i]));

      // n = tanh(preN)
      const dpN = dN.map((v, i) => v * (1 - n[i] ** 2));
      for (let i = 0; i < H; i++) {
        for (let j = 0; j < dXH; j++) dWn[i][j] += dpN[i] * xrh[j];
        dbn[i] += dpN[i];
      }
      const dxrh = mv(transpose(W_n), dpN); // (dXH,)
      const drh = dxrh.slice(0, H);         // gradient on r*hPrev

      // r * hPrev
      const dR = drh.map((v, i) => v * hPrev[i]);
      const dhPrevFromRh = drh.map((v, i) => v * r[i]);

      // z = sigmoid(preZ)
      const dpZ = dZ.map((v, i) => v * z[i] * (1 - z[i]));
      for (let i = 0; i < H; i++) {
        for (let j = 0; j < dXH; j++) dWz[i][j] += dpZ[i] * xh[j];
        dbz[i] += dpZ[i];
      }
      const dxhFromZ = mv(transpose(W_z), dpZ);

      // r = sigmoid(preR)
      const dpR = dR.map((v, i) => v * r[i] * (1 - r[i]));
      for (let i = 0; i < H; i++) {
        for (let j = 0; j < dXH; j++) dWr[i][j] += dpR[i] * xh[j];
        dbr[i] += dpR[i];
      }
      const dxhFromR = mv(transpose(W_r), dpR);

      // xh = [hPrev, x]  →  gradient on hPrev from gates
      const dhPrevFromXh = dxhFromZ.slice(0, H).map((v, i) => v + dxhFromR[i]);

      // Total dh for previous step
      dh = dhPrevDirect.map((v, i) => v + dhPrevFromRh[i] + dhPrevFromXh[i]);
    }
  }

  // Build training sequences (stride=2 for speed)
  const trainSeqs = [];
  for (let t = seqLen - 1; t < trainEnd; t += 2) {
    trainSeqs.push({
      xSeq: allX.slice(t - seqLen + 1, t + 1),
      ePreds: cols.map(c => data[t][c] || 0),
      yTrue: data[t].y_true,
    });
  }

  const nSeqs = trainSeqs.length;
  if (nSeqs === 0) {
    const w = cols.map(() => 1 / K);
    const predictions = data.map(row => cols.reduce((s, c) => s + (row[c] || 0) / K, 0));
    const weightHistory = data.map(() => [...w]);
    return { predictions, weightHistory };
  }

  for (let epoch = 0; epoch < epochs; epoch++) {
    const dWr = zeroMat(H, dXH), dbr = zeroVec(H);
    const dWz = zeroMat(H, dXH), dbz = zeroVec(H);
    const dWn = zeroMat(H, dXH), dbn = zeroVec(H);
    const dWo = zeroMat(K, H),   dbo = zeroVec(K);

    for (const { xSeq, ePreds, yTrue } of trainSeqs) {
      const { states, hFinal } = gruSeqForward(xSeq);

      // Output layer
      const logits = addVec(mv(W_o, hFinal), b_o);
      const w = softmax(logits);
      const yHat = w.reduce((s, wi, k) => s + wi * ePreds[k], 0);

      const dOut = 2 * (yHat - yTrue);
      const dLogits = w.map((wi, i) => dOut * wi * (ePreds[i] - yHat));

      for (let i = 0; i < K; i++) {
        for (let j = 0; j < H; j++) dWo[i][j] += dLogits[i] * hFinal[j];
        dbo[i] += dLogits[i];
      }

      const dhFinal = mv(transpose(W_o), dLogits);
      gruBptt(states, dhFinal, dWr, dbr, dWz, dbz, dWn, dbn);
    }

    const step = epoch + 1;
    adamStep(W_r, scaleMat(dWr, 1 / nSeqs), sWr, lr, step);
    adamStep(b_r, scaleVec(dbr, 1 / nSeqs), sbr, lr, step);
    adamStep(W_z, scaleMat(dWz, 1 / nSeqs), sWz, lr, step);
    adamStep(b_z, scaleVec(dbz, 1 / nSeqs), sbz, lr, step);
    adamStep(W_n, scaleMat(dWn, 1 / nSeqs), sWn, lr, step);
    adamStep(b_n, scaleVec(dbn, 1 / nSeqs), sbn, lr, step);
    adamStep(W_o, scaleMat(dWo, 1 / nSeqs), sWo, lr, step);
    adamStep(b_o, scaleVec(dbo, 1 / nSeqs), sbo, lr, step);
    if (onProgress) {
      onProgress({ epoch: epoch + 1, total: epochs });
      await new Promise(r => setTimeout(r, 0));
    }
  }

  // Inference on all data (pad sequences shorter than seqLen with zeros)
  const predictions = [], weightHistory = [];
  const zero = zeroVec(dIn);
  for (let t = 0; t < data.length; t++) {
    const start = Math.max(0, t - seqLen + 1);
    const raw = allX.slice(start, t + 1);
    const xSeq = [...Array.from({ length: seqLen - raw.length }, () => zero), ...raw];
    const { hFinal } = gruSeqForward(xSeq);
    const ePreds = cols.map(c => data[t][c] || 0);
    const w = softmax(addVec(mv(W_o, hFinal), b_o));
    predictions.push(w.reduce((s, wi, k) => s + wi * ePreds[k], 0));
    weightHistory.push([...w]);
  }
  return { predictions, weightHistory };
}

// ── LSTM Aggregator ───────────────────────────────────────────────────────────
//
// Architecture: LSTM(H=16) → Dense(K, Softmax).
// Same pipeline as GRU: sliding windows, 70% temporal split, full-batch Adam.

export async function runLSTMAggregator(data, cols, params, onProgress) {
  const K = cols.length;
  const epochs = Math.max(5, Math.min(80, params.lstmEpochs || 30));
  const lr = 1e-3;
  const H = 16;
  const rawTrainEnd = Math.min(data.length - 24, Math.max(K + 4, Math.floor(data.length * 0.7)));
  const maxSeqLen = Math.max(2, Math.floor(rawTrainEnd / 4));
  const seqLen = Math.max(2, Math.min(maxSeqLen, params.lstmSeqLen || 8)); // hard cap = min(maxSeqLen, 725)
  const trainEnd = rawTrainEnd;

  const activeRegimes = REGIME_KEYS.filter(k => data.some(r => (r[k] || 0) !== 0));
  const dIn = K + activeRegimes.length;
  const dXH = dIn + H;

  function getFeatures(row) {
    return [...cols.map(c => row[c] || 0), ...activeRegimes.map(k => row[k] || 0)];
  }

  const allFeatRaw = data.map(getFeatures);
  const scaler = fitScaler(allFeatRaw.slice(0, trainEnd));
  const allX = allFeatRaw.map(f => applyScaler(f, scaler));

  // LSTM gates — all use [h_prev, x] concatenation
  const Wf = xavierMatrix(H, dXH), bf = zeroVec(H); // forget gate
  const Wi = xavierMatrix(H, dXH), bi = zeroVec(H); // input gate
  const Wg = xavierMatrix(H, dXH), bg = zeroVec(H); // candidate gate
  const Wo = xavierMatrix(H, dXH), bo = zeroVec(H); // output gate
  const Wp = xavierMatrix(K, H),   bp = zeroVec(K); // projection h → experts

  const sWf = makeState(Wf), sbf = makeState(bf);
  const sWi = makeState(Wi), sbi = makeState(bi);
  const sWg = makeState(Wg), sbg = makeState(bg);
  const sWo = makeState(Wo), sbo = makeState(bo);
  const sWp = makeState(Wp), sbp = makeState(bp);

  function lstmSeqForward(xSeq) {
    let h = zeroVec(H);
    let c = zeroVec(H);
    const states = [];
    for (const x of xSeq) {
      const hPrev = [...h];
      const cPrev = [...c];
      const xh = [...hPrev, ...x];
      const f  = addVec(mv(Wf, xh), bf).map(sigmoidFn);
      const ig = addVec(mv(Wi, xh), bi).map(sigmoidFn);
      const gg = addVec(mv(Wg, xh), bg).map(tanhFn);
      const og = addVec(mv(Wo, xh), bo).map(sigmoidFn);
      c = f.map((fi, idx) => fi * cPrev[idx] + ig[idx] * gg[idx]);
      const tanhC = c.map(tanhFn);
      h = og.map((oi, idx) => oi * tanhC[idx]);
      states.push({ hPrev, cPrev, f, ig, gg, og, tanhC, xh });
    }
    return { states, hFinal: h };
  }

  function lstmBptt(states, dhFinal, dWf, dbfg, dWi, dbig, dWg, dbgg, dWo, dbog) {
    let dh = [...dhFinal];
    let dc = zeroVec(H);
    for (let t = states.length - 1; t >= 0; t--) {
      const { hPrev, cPrev, f, ig, gg, og, tanhC, xh } = states[t];

      // h = og * tanh(c)
      const dOg = dh.map((v, i) => v * tanhC[i]);
      dc = dc.map((v, i) => v + dh[i] * og[i] * (1 - tanhC[i] ** 2));

      // c = f * cPrev + ig * gg
      const dF  = dc.map((v, i) => v * cPrev[i]);
      const dIg = dc.map((v, i) => v * gg[i]);
      const dGg = dc.map((v, i) => v * ig[i]);
      dc = dc.map((v, i) => v * f[i]); // dc_prev

      const dpF  = dF.map((v, i)  => v * f[i]  * (1 - f[i]));
      const dpIg = dIg.map((v, i) => v * ig[i] * (1 - ig[i]));
      const dpGg = dGg.map((v, i) => v * (1 - gg[i] ** 2));
      const dpOg = dOg.map((v, i) => v * og[i] * (1 - og[i]));

      for (let r = 0; r < H; r++) {
        for (let cl = 0; cl < dXH; cl++) {
          dWf[r][cl] += dpF[r]  * xh[cl];
          dWi[r][cl] += dpIg[r] * xh[cl];
          dWg[r][cl] += dpGg[r] * xh[cl];
          dWo[r][cl] += dpOg[r] * xh[cl];
        }
        dbfg[r] += dpF[r];  dbig[r] += dpIg[r];
        dbgg[r] += dpGg[r]; dbog[r] += dpOg[r];
      }

      const dxhF = mv(transpose(Wf), dpF);
      const dxhI = mv(transpose(Wi), dpIg);
      const dxhG = mv(transpose(Wg), dpGg);
      const dxhO = mv(transpose(Wo), dpOg);
      dh = dxhF.slice(0, H).map((v, i) => v + dxhI[i] + dxhG[i] + dxhO[i]);
    }
  }

  // Build training sequences (stride=2 for speed)
  const trainSeqs = [];
  for (let t = seqLen - 1; t < trainEnd; t += 2) {
    trainSeqs.push({
      xSeq: allX.slice(t - seqLen + 1, t + 1),
      ePreds: cols.map(c => data[t][c] || 0),
      yTrue: data[t].y_true,
    });
  }

  const nSeqs = trainSeqs.length;
  if (nSeqs === 0) {
    const w = cols.map(() => 1 / K);
    const predictions = data.map(row => cols.reduce((s, c) => s + (row[c] || 0) / K, 0));
    const weightHistory = data.map(() => [...w]);
    return { predictions, weightHistory };
  }

  for (let epoch = 0; epoch < epochs; epoch++) {
    const dWf = zeroMat(H, dXH), dbfg = zeroVec(H);
    const dWi = zeroMat(H, dXH), dbig = zeroVec(H);
    const dWg = zeroMat(H, dXH), dbgg = zeroVec(H);
    const dWo = zeroMat(H, dXH), dbog = zeroVec(H);
    const dWp = zeroMat(K, H),   dbpg = zeroVec(K);

    for (const { xSeq, ePreds, yTrue } of trainSeqs) {
      const { states, hFinal } = lstmSeqForward(xSeq);

      const logits = addVec(mv(Wp, hFinal), bp);
      const w = softmax(logits);
      const yHat = w.reduce((s, wi, k) => s + wi * ePreds[k], 0);

      const dOut = 2 * (yHat - yTrue);
      const dLogits = w.map((wi, i) => dOut * wi * (ePreds[i] - yHat));

      for (let i = 0; i < K; i++) {
        for (let j = 0; j < H; j++) dWp[i][j] += dLogits[i] * hFinal[j];
        dbpg[i] += dLogits[i];
      }

      const dhFinal = mv(transpose(Wp), dLogits);
      lstmBptt(states, dhFinal, dWf, dbfg, dWi, dbig, dWg, dbgg, dWo, dbog);
    }

    const step = epoch + 1;
    adamStep(Wf, scaleMat(dWf, 1 / nSeqs), sWf, lr, step);
    adamStep(bf, scaleVec(dbfg, 1 / nSeqs), sbf, lr, step);
    adamStep(Wi, scaleMat(dWi, 1 / nSeqs), sWi, lr, step);
    adamStep(bi, scaleVec(dbig, 1 / nSeqs), sbi, lr, step);
    adamStep(Wg, scaleMat(dWg, 1 / nSeqs), sWg, lr, step);
    adamStep(bg, scaleVec(dbgg, 1 / nSeqs), sbg, lr, step);
    adamStep(Wo, scaleMat(dWo, 1 / nSeqs), sWo, lr, step);
    adamStep(bo, scaleVec(dbog, 1 / nSeqs), sbo, lr, step);
    adamStep(Wp, scaleMat(dWp, 1 / nSeqs), sWp, lr, step);
    adamStep(bp, scaleVec(dbpg, 1 / nSeqs), sbp, lr, step);

    if (onProgress) {
      onProgress({ epoch: epoch + 1, total: epochs });
      await new Promise(r => setTimeout(r, 0));
    }
  }

  // Inference (pad short sequences with zeros)
  const predictions = [], weightHistory = [];
  const zero = zeroVec(dIn);
  for (let t = 0; t < data.length; t++) {
    const start = Math.max(0, t - seqLen + 1);
    const raw = allX.slice(start, t + 1);
    const xSeq = [...Array.from({ length: seqLen - raw.length }, () => zero), ...raw];
    const { hFinal } = lstmSeqForward(xSeq);
    const ePreds = cols.map(c => data[t][c] || 0);
    const w = softmax(addVec(mv(Wp, hFinal), bp));
    predictions.push(w.reduce((s, wi, k) => s + wi * ePreds[k], 0));
    weightHistory.push([...w]);
  }
  return { predictions, weightHistory };
}
