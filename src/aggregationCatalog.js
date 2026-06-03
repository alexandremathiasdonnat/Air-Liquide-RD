export const ALGO_GROUPS = [
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
    label: "Adaptatifs / Statistiques",
    algos: [
      {
        id: "InvMSE",
        name: "Inverse MSE",
        desc: "Poids inversement proportionnels au MSE glissant.",
        params: [{ id: "window", label: "Fenêtre (pas)", type: "slider", min: 6, max: 168, step: 6, default: 48 }],
      },
      {
        id: "BestExpert",
        name: "Best Expert",
        desc: "Sélectionne l'expert avec la plus faible MAE glissante.",
        params: [{ id: "window", label: "Fenêtre (pas)", type: "slider", min: 6, max: 168, step: 6, default: 48 }],
      },
    ],
  },
  {
    label: "Stacking",
    algos: [
      {
        id: "LinearStacking",
        name: "Linear Regression Stacking",
        desc: "Méta-modèle OLS appris sur 70% des données (sans fuite temporelle).",
        params: [],
      },
      {
        id: "Ridge",
        name: "Ridge Regression Stacking",
        desc: "Combinaison linéaire régularisée L2 (Ridge, entraîné sur l'intégralité des données).",
        params: [{ id: "alpha", label: "Régularisation α", type: "slider", min: 0.1, max: 50, step: 0.1, default: 1 }],
      },
      {
        id: "XGBoostStacking",
        name: "XGBoost Regressor Stacking",
        desc: "Gradient boosting de stumps appris sur 70% des données.",
        params: [
          { id: "nTrees", label: "Nb arbres", type: "slider", min: 10, max: 200, step: 10, default: 50 },
          { id: "xgbLr", label: "Learning rate", type: "slider", min: 0.01, max: 0.3, step: 0.01, default: 0.1 },
        ],
      },
      {
        id: "MLPStacking",
        name: "MLP Regressor Stacking",
        desc: "Petit réseau dense (Input→16→8→K softmax) apprenant les poids experts dynamiquement.",
        params: [
          { id: "mlpEpochs", label: "Epoch", type: "slider", min: 30, max: 80, step: 5, default: 50 },
        ],
      },
    ],
  },
  {
    label: "Online Learning / Opera",
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
    label: "HMOE",
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
    label: "Temporal Neural Aggregation",
    algos: [
      {
        id: "GRU",
        name: "GRU Aggregator",
        desc: "Réseau GRU capturant la mémoire temporelle pour produire des poids experts dynamiques.",
        params: [
          { id: "gruEpochs", label: "Epoch", type: "slider", min: 5, max: 80, step: 5, default: 30 },
          { id: "gruSeqLen", label: "Longueur séquence", type: "slider", min: 4, max: 24, step: 2, default: 8 },
        ],
      },
      {
        id: "LSTM",
        name: "LSTM Aggregator",
        desc: "Réseau LSTM (H=16, cell+hidden state) capturant la mémoire temporelle longue pour produire des poids experts dynamiques.",
        params: [
          { id: "lstmEpochs", label: "Epoch", type: "slider", min: 5, max: 80, step: 5, default: 30 },
          { id: "lstmSeqLen", label: "Longueur séquence", type: "slider", min: 4, max: 24, step: 2, default: 8 },
        ],
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
  LinearStacking: "LinStk",
  Ridge: "RidgeStk",
  XGBoostStacking: "XGBStk",
  MLPStacking: "MLPStk",
  GRU: "GRU",
  LSTM: "LSTM",
};

export const MONTE_CARLO_COMPLEXITY = {
  SimpleMean: 1.0,
  Median: 1.2,
  TrimmedMean: 1.15,
  InvMSE: 1.6,
  BestExpert: 1.45,
  LinearStacking: 2.0,
  Ridge: 1.8,
  XGBoostStacking: 4.0,
  MLPStacking: 12.0,
  BOA: 2.8,
  MLpol: 2.5,
  MLprod: 2.6,
  FTRL: 2.4,
  HMOE_BOA: 4.3,
  HMOE_MLpol: 4.0,
  HMOE_MLprod: 4.1,
  HMOE_FTRL: 3.9,
  GRU: 18.0,
  LSTM: 22.0,
};
