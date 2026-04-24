import { ALGO_SHORT, ALGOS, HMOE_ALGO_IDS, LOSS_TYPES, OPERA_ALGO_IDS } from "./aggregationCatalog";
import { HMOE_REGIME_TYPES } from "./hmoe";

export const DEFAULT_LOSS_TYPE = "mse";
export const DEFAULT_USE_GRAD = true;
export const DEFAULT_FTRL_PARAMS = { eta0: 0.01, tol: 1e-20, maxiter: 50 };
export const DEFAULT_EXTRA_PARAMS = { window: 48, alpha: 1, trim: 20 };
export const DEFAULT_HMOE_REGIME_IDS = HMOE_REGIME_TYPES.map((regime) => regime.id);

function getRunChronologyValue(run, index) {
  return Number.isFinite(run?.executedAt) ? run.executedAt : index;
}

function buildDefaultAlgoConfig(algoId) {
  return {
    algoId,
    lossType: DEFAULT_LOSS_TYPE,
    useGrad: DEFAULT_USE_GRAD,
    extraP: { ...DEFAULT_EXTRA_PARAMS },
    ftrlP: { ...DEFAULT_FTRL_PARAMS },
    selectedHmoeRegimes: [...DEFAULT_HMOE_REGIME_IDS],
    source: "default",
    regimesSource: HMOE_ALGO_IDS.includes(algoId) ? "default" : null,
    hasOwnRun: false,
    sharedHmoeSourceAlgoId: null,
  };
}

export function cloneAlgoRunConfig(config) {
  return {
    ...config,
    extraP: { ...config.extraP },
    ftrlP: { ...config.ftrlP },
    selectedHmoeRegimes: [...config.selectedHmoeRegimes],
  };
}

export function resolveMonteCarloAlgoConfigs(allRuns) {
  const lastRunByAlgoId = {};
  let latestHmoeRun = null;

  allRuns.forEach((run, index) => {
    const chronology = getRunChronologyValue(run, index);
    const previous = lastRunByAlgoId[run.algoId];
    if (!previous || chronology >= previous.chronology) {
      lastRunByAlgoId[run.algoId] = { run, chronology };
    }
    if (HMOE_ALGO_IDS.includes(run.algoId) && (!latestHmoeRun || chronology >= latestHmoeRun.chronology)) {
      latestHmoeRun = { run, chronology };
    }
  });

  const sharedHmoeRegimes = latestHmoeRun?.run?.selectedHmoeRegimes?.length
    ? [...latestHmoeRun.run.selectedHmoeRegimes]
    : [...DEFAULT_HMOE_REGIME_IDS];

  const configs = {};
  ALGOS.forEach((algo) => {
    const defaults = buildDefaultAlgoConfig(algo.id);
    const algoLastRun = lastRunByAlgoId[algo.id]?.run || null;
    const isHmoeAlgo = HMOE_ALGO_IDS.includes(algo.id);

    configs[algo.id] = {
      ...defaults,
      lossType: algoLastRun?.lossType ?? defaults.lossType,
      useGrad: algoLastRun?.useGrad ?? defaults.useGrad,
      extraP: { ...defaults.extraP, ...(algoLastRun?.extraP || {}) },
      ftrlP: { ...defaults.ftrlP, ...(algoLastRun?.ftrlP || {}) },
      selectedHmoeRegimes: isHmoeAlgo ? sharedHmoeRegimes : [...defaults.selectedHmoeRegimes],
      source: algoLastRun ? "last-run" : "default",
      regimesSource: isHmoeAlgo ? (latestHmoeRun ? "shared-hmoe-run" : "default") : null,
      hasOwnRun: Boolean(algoLastRun),
      sharedHmoeSourceAlgoId: latestHmoeRun?.run?.algoId || null,
    };
  });

  return {
    configs,
    latestHmoeRun: latestHmoeRun?.run || null,
    sharedHmoeRegimes,
  };
}

function getLossLabel(lossType) {
  return LOSS_TYPES.find((loss) => loss.id === lossType)?.label || String(lossType || "").toUpperCase();
}

function formatValue(value) {
  return typeof value === "number" ? String(value) : `${value}`;
}

export function getMonteCarloAlgoParamTokens(algoId, config) {
  const tokens = [];
  const isOperaFamily = OPERA_ALGO_IDS.includes(algoId) || HMOE_ALGO_IDS.includes(algoId);

  if (isOperaFamily) {
    tokens.push(`loss ${getLossLabel(config.lossType)}`);
    tokens.push(`grad ${config.useGrad ? "on" : "off"}`);
  }
  if (algoId === "FTRL" || algoId === "HMOE_FTRL") {
    tokens.push(`eta0 ${formatValue(config.ftrlP.eta0)}`);
    tokens.push(`tol ${formatValue(config.ftrlP.tol)}`);
    tokens.push(`maxiter ${formatValue(config.ftrlP.maxiter)}`);
  }
  if (algoId === "TrimmedMean") {
    tokens.push(`trim ${formatValue(config.extraP.trim)}%`);
  }
  if (algoId === "InvMSE" || algoId === "BestExpert") {
    tokens.push(`window ${formatValue(config.extraP.window)}`);
  }
  if (algoId === "Ridge") {
    tokens.push(`alpha ${formatValue(config.extraP.alpha)}`);
  }

  return tokens;
}

const REGIME_SHORT = {
  day_night: "DayNight",
  wind: "Wind",
  updown: "UpDown",
  volatility: "Vol",
  trend: "Trend",
};

export function buildAlgoRunLabel(algoId, config, expertMode = null) {
  const short = ALGO_SHORT[algoId] || algoId;
  const isOperaFamily = OPERA_ALGO_IDS.includes(algoId) || HMOE_ALGO_IDS.includes(algoId);
  const isHmoe = HMOE_ALGO_IDS.includes(algoId);
  const parts = [];
  if (expertMode) parts.push(expertMode);
  if (isOperaFamily) {
    parts.push(getLossLabel(config.lossType));
    parts.push(config.useGrad ? "gradOn" : "gradOff");
  }
  if (algoId === "FTRL" || algoId === "HMOE_FTRL") {
    parts.push(`eta0=${formatValue(config.ftrlP.eta0)}`);
    parts.push(`tol=${formatValue(config.ftrlP.tol)}`);
    parts.push(`maxiter=${formatValue(config.ftrlP.maxiter)}`);
  }
  if (algoId === "TrimmedMean") parts.push(`trim=${formatValue(config.extraP.trim)}%`);
  if (algoId === "InvMSE" || algoId === "BestExpert") parts.push(`win=${formatValue(config.extraP.window)}`);
  if (algoId === "Ridge") parts.push(`α=${formatValue(config.extraP.alpha)}`);
  if (isHmoe && config.selectedHmoeRegimes?.length) {
    parts.push(config.selectedHmoeRegimes.map((id) => REGIME_SHORT[id] || id).join("+"));
  }
  return `${short}(${parts.join(", ")})`;
}

export function getHmoeRegimeNames(regimeIds) {
  return regimeIds.map((regimeId) => HMOE_REGIME_TYPES.find((regime) => regime.id === regimeId)?.label || regimeId);
}

export function getParamSourceLabel(source) {
  return source === "last-run" ? "Dernier run propre" : "Valeurs par defaut";
}

export function getHmoeRegimeSourceLabel(source, sourceAlgoName) {
  if (source === "shared-hmoe-run") {
    return sourceAlgoName ? `Selection HMOE partagee via ${sourceAlgoName}` : "Selection HMOE partagee";
  }
  return "Regimes HMOE par defaut";
}
