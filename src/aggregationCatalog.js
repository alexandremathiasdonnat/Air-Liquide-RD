export const ALGO_GROUPS = [
  {
    label: "Opera - MOE",
    algos: [
      { id: "BOA", name: "MOE BOA", desc: "Bernstein Online Aggregation.", params: [] },
      { id: "MLpol", name: "MOE MLpol", desc: "Multiplicative Weights Polynomial.", params: [] },
      { id: "MLprod", name: "MOE MLprod", desc: "Multiplicative Weights Prod.", params: [] },
      {
        id: "FTRL",
        name: "MOE FTRL",
        desc: "Follow The Regularized Leader.",
        params: [
          { id: "eta0", label: "Learning rate η₀", type: "slider", min: 0.001, max: 0.5, step: 0.001, default: 0.01 },
          { id: "tol", label: "Tolérance", type: "select", options: [1e-5, 1e-10, 1e-15, 1e-20], default: 1e-20 },
          { id: "maxiter", label: "Max itérations", type: "slider", min: 10, max: 200, step: 10, default: 50 },
        ],
      },
    ],
  },
  {
    label: "Opera - HMOE",
    algos: [
      { id: "HMOE_BOA", name: "HMOE BOA", desc: "BOA avec branches regime-gated HMOE.", params: [] },
      { id: "HMOE_MLpol", name: "HMOE MLpol", desc: "MLpol avec branches regime-gated HMOE.", params: [] },
      { id: "HMOE_MLprod", name: "HMOE MLprod", desc: "MLprod avec branches regime-gated HMOE.", params: [] },
      {
        id: "HMOE_FTRL",
        name: "HMOE FTRL",
        desc: "FTRL avec branches regime-gated HMOE.",
        params: [
          { id: "eta0", label: "Learning rate η₀", type: "slider", min: 0.001, max: 0.5, step: 0.001, default: 0.01 },
          { id: "tol", label: "Tolérance", type: "select", options: [1e-5, 1e-10, 1e-15, 1e-20], default: 1e-20 },
          { id: "maxiter", label: "Max itérations", type: "slider", min: 10, max: 200, step: 10, default: 50 },
        ],
      },
    ],
  },
  {
    label: "Statiques",
    algos: [
      { id: "SimpleMean", name: "Moyenne simple", desc: "Moyenne arithmétique non pondérée.", params: [] },
      { id: "Median", name: "Médiane", desc: "Médiane des prédictions.", params: [] },
      {
        id: "TrimmedMean",
        name: "Trimmed Mean",
        desc: "Moyenne après exclusion des X% d'experts.",
        params: [{ id: "trim", label: "Trim (%)", type: "slider", min: 5, max: 40, step: 5, default: 20 }],
      },
    ],
  },
  {
    label: "Adaptatifs",
    algos: [
      {
        id: "InvMSE",
        name: "Inverse MSE",
        desc: "Poids inversement proportionnels au MSE.",
        params: [{ id: "window", label: "Fenêtre (pas)", type: "slider", min: 6, max: 168, step: 6, default: 48 }],
      },
      {
        id: "BestExpert",
        name: "Best Expert",
        desc: "Sélectionne l'expert avec la plus faible MAE.",
        params: [{ id: "window", label: "Fenêtre (pas)", type: "slider", min: 6, max: 168, step: 6, default: 48 }],
      },
      {
        id: "Ridge",
        name: "Ridge Blending",
        desc: "Combinaison linéaire régularisée L2.",
        params: [{ id: "alpha", label: "Régularisation α", type: "slider", min: 0.1, max: 50, step: 0.1, default: 1 }],
      },
    ],
  },
];

export const ALGOS = ALGO_GROUPS.flatMap((group) => group.algos);
export const OPERA_ALGO_IDS = ["BOA", "MLpol", "MLprod", "FTRL"];
export const HMOE_ALGO_IDS = ["HMOE_BOA", "HMOE_MLpol", "HMOE_MLprod", "HMOE_FTRL"];
export const LOSS_TYPES = [
  { id: "mse", label: "MSE" },
  { id: "mae", label: "MAE" },
  { id: "mape", label: "MAPE" },
  { id: "msle", label: "MSLE" },
  { id: "mspe", label: "MSPE" },
];

export const ALGO_SHORT = {
  BOA: "BOA",
  MLpol: "MLpol",
  MLprod: "MLprod",
  FTRL: "FTRL",
  HMOE_BOA: "HBOA",
  HMOE_MLpol: "HMLpol",
  HMOE_MLprod: "HMLprod",
  HMOE_FTRL: "HFTRL",
  SimpleMean: "Mean",
  Median: "Median",
  TrimmedMean: "TrimMean",
  InvMSE: "InvMSE",
  BestExpert: "BestExp",
  Ridge: "Ridge Blending",
};

export const MONTE_CARLO_COMPLEXITY = {
  SimpleMean: 1.0,
  Median: 1.2,
  TrimmedMean: 1.15,
  InvMSE: 1.6,
  BestExpert: 1.45,
  Ridge: 1.8,
  BOA: 2.8,
  MLpol: 2.5,
  MLprod: 2.6,
  FTRL: 2.4,
  HMOE_BOA: 4.3,
  HMOE_MLpol: 4.0,
  HMOE_MLprod: 4.1,
  HMOE_FTRL: 3.9,
};
