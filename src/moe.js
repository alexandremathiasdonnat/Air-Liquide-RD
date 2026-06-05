// ─── Shared math helpers ──────────────────────────────────────────────────────
export function vnorm(w){const s=w.reduce((a,b)=>a+b,0);return s===0?w.map(()=>1/w.length):w.map(v=>v/s);}
export const loss_fn={mse:(x,y)=>(x-y)**2,mae:(x,y)=>Math.abs(x-y),mape:(x,y)=>Math.abs(x-y)/(Math.abs(y)+1e-8),msle:(x,y)=>(Math.log(Math.max(y,0)+1)-Math.log(Math.max(x,0)+1))**2,mspe:(x,y)=>((y-x)/(Math.abs(y)+1e-8))**2};
export const grad_fn={mse:(yh,y)=>2*(yh-y),mae:(yh,y)=>Math.sign(yh-y),mape:(yh,y)=>Math.sign(yh-y)/(Math.abs(y)+1e-8),msle:(yh,y)=>2*(Math.log(Math.max(y,0)+1)-Math.log(Math.max(yh,0)+1))*(-1/(Math.max(yh,0)+1)),mspe:(yh,y)=>-2*yh+2*y};
export function computeR(yhat,y,x,lt,ug){if(ug){const g=grad_fn[lt](yhat,y);return x.map(xk=>g*(yhat-xk));}const ly=loss_fn[lt](yhat,y);return x.map(xk=>ly-loss_fn[lt](xk,y));}

// ─── Opera MOE algorithms ─────────────────────────────────────────────────────
// All four methods: adapt on data[0..N-25], freeze weights at N-24, predict
// the last 24 observations with the frozen weight vector (no further update).

export function runBOA(data, cols, lt, ug) {
  const K = cols.length, EPS = 1 / Math.pow(2, 20);
  const testStart = Math.max(1, data.length - 24);
  let cv = new Array(K).fill(EPS), ml = new Array(K).fill(EPS),
      crr = new Array(K).fill(0), lrs = new Array(K).fill(EPS);
  const preds = [], wh = [];

  for (let t = 0; t < testStart; t++) {
    const x = cols.map(c => data[t][c] || 0), y = data[t].y_true;
    const Ra = lrs.map((lr, k) => Math.log(lr) + Math.log(1 / K) + lr * crr[k]);
    const Rm = Math.max(...Ra);
    const w = vnorm(Ra.map(v => Math.exp(v - Rm)));
    const yh = w.reduce((s, wk, k) => s + wk * x[k], 0);
    preds.push(yh); wh.push([...w]);
    const r = computeR(yh, y, x, lt, ug), r2 = r.map(v => v ** 2);
    ml = ml.map((m, k) => Math.max(m, Math.abs(r[k])));
    const B2 = ml.map(m => Math.pow(2, Math.ceil(Math.log2(m + 1e-30))));
    cv = cv.map((v, k) => v + r2[k]);
    lrs = lrs.map((_, k) => Math.min(1 / B2[k], Math.sqrt(Math.log(K) / cv[k])));
    crr = crr.map((v, k) => v + 0.5 * (r[k] - lrs[k] * r2[k] + B2[k] * (lrs[k] * r[k] > 0.5 ? 1 : 0)));
  }

  // Frozen weight: computed from final training state, applied to all test obs
  const Ra_f = lrs.map((lr, k) => Math.log(lr) + Math.log(1 / K) + lr * crr[k]);
  const Rm_f = Math.max(...Ra_f);
  const wFinal = vnorm(Ra_f.map(v => Math.exp(v - Rm_f)));

  for (let t = testStart; t < data.length; t++) {
    const x = cols.map(c => data[t][c] || 0);
    preds.push(wFinal.reduce((s, wk, k) => s + wk * x[k], 0));
    wh.push([...wFinal]);
  }
  return { predictions: preds, weightHistory: wh };
}

export function runMLpol(data, cols, lt, ug) {
  const K = cols.length, EPS = 1 / Math.pow(2, 20);
  const testStart = Math.max(1, data.length - 24);
  let cr = new Array(K).fill(0), lrs = new Array(K).fill(EPS), msr = new Array(K).fill(0);
  const preds = [], wh = [];

  for (let t = 0; t < testStart; t++) {
    const x = cols.map(c => data[t][c] || 0), y = data[t].y_true;
    const relu = cr.map(v => Math.max(v, 0)), wRaw = lrs.map((lr, k) => lr * relu[k]);
    const ws = wRaw.reduce((a, b) => a + b, 0);
    const w = ws === 0 ? new Array(K).fill(1 / K) : vnorm(wRaw);
    const yh = w.reduce((s, wk, k) => s + wk * x[k], 0);
    preds.push(yh); wh.push([...w]);
    const r = computeR(yh, y, x, lt, ug), r2 = r.map(v => v ** 2);
    cr = cr.map((v, k) => v + r[k]);
    const diff = Math.max(Math.max(...r2) - Math.max(...msr), 0);
    msr = msr.map(v => v + diff);
    lrs = lrs.map((lr, k) => 1 / (1 / lr + r2[k] + diff));
  }

  const wRaw_f = lrs.map((lr, k) => lr * Math.max(cr[k], 0));
  const ws_f = wRaw_f.reduce((a, b) => a + b, 0);
  const wFinal = ws_f === 0 ? new Array(K).fill(1 / K) : vnorm(wRaw_f);

  for (let t = testStart; t < data.length; t++) {
    const x = cols.map(c => data[t][c] || 0);
    preds.push(wFinal.reduce((s, wk, k) => s + wk * x[k], 0));
    wh.push([...wFinal]);
  }
  return { predictions: preds, weightHistory: wh };
}

export function runMLprod(data, cols, lt, ug) {
  const K = cols.length, EPS = 1e-30, I = 1 / Math.pow(2, 20);
  const testStart = Math.max(1, data.length - 24);
  let cv = new Array(K).fill(I), ml = new Array(K).fill(I),
      cr = new Array(K).fill(0), lrs = new Array(K).fill(I);
  const preds = [], wh = [];

  for (let t = 0; t < testStart; t++) {
    const x = cols.map(c => data[t][c] || 0), y = data[t].y_true;
    const w = vnorm(lrs.map((lr, k) => lr * Math.exp(cr[k])));
    const yh = w.reduce((s, wk, k) => s + wk * x[k], 0);
    preds.push(yh); wh.push([...w]);
    const r = computeR(yh, y, x, lt, ug), r2 = r.map(v => v ** 2);
    cv = cv.map((v, k) => v + r2[k]);
    ml = ml.map((m, k) => Math.max(m, Math.abs(r[k])));
    const nl = lrs.map((_, k) => Math.min(Math.min(0.5 / (ml[k] + EPS), Math.sqrt(Math.log(K + 1) / (cv[k] + EPS))), 1 / EPS));
    cr = cr.map((v, k) => (nl[k] / (lrs[k] + EPS)) * v + Math.log(1 + nl[k] * r[k] + EPS));
    lrs = nl;
  }

  const wFinal = vnorm(lrs.map((lr, k) => lr * Math.exp(cr[k])));

  for (let t = testStart; t < data.length; t++) {
    const x = cols.map(c => data[t][c] || 0);
    preds.push(wFinal.reduce((s, wk, k) => s + wk * x[k], 0));
    wh.push([...wFinal]);
  }
  return { predictions: preds, weightHistory: wh };
}

export function runFTRL(data, cols, lt, ug, params) {
  const K = cols.length;
  const testStart = Math.max(1, data.length - 24);
  let w = new Array(K).fill(1 / K), G = new Array(K).fill(0), eta = params.eta0 || 0.01;
  const preds = [], wh = [];

  for (let t = 0; t < testStart; t++) {
    const x = cols.map(c => data[t][c] || 0), y = data[t].y_true;
    const yh = w.reduce((s, wk, k) => s + wk * x[k], 0);
    preds.push(yh); wh.push([...w]);
    const gl = grad_fn[lt](yh, y), Gt = x.map(xk => gl * xk);
    eta = 1 / Math.sqrt(1 / (eta ** 2) + Gt.reduce((s, v) => s + v ** 2, 0) + 1e-30);
    G = G.map((v, k) => v + Gt[k]);
    w = vnorm(w.map((wk, k) => wk * Math.exp(-eta * G[k])));
  }

  // w is already the frozen weight after the last training update
  const wFinal = [...w];

  for (let t = testStart; t < data.length; t++) {
    const x = cols.map(c => data[t][c] || 0);
    preds.push(wFinal.reduce((s, wk, k) => s + wk * x[k], 0));
    wh.push([...wFinal]);
  }
  return { predictions: preds, weightHistory: wh };
}
