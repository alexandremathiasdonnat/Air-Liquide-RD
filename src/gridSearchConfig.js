import { ALGOS, HMOE_ALGO_IDS, OPERA_ALGO_IDS } from "./aggregationCatalog";
import {
  DEFAULT_EXTRA_PARAMS,
  DEFAULT_FTRL_PARAMS,
  DEFAULT_HMOE_REGIME_IDS,
  DEFAULT_LOSS_TYPE,
  DEFAULT_USE_GRAD,
  buildAlgoRunLabel,
  getHmoeRegimeNames,
  getMonteCarloAlgoParamTokens,
} from "./monteCarloConfig";

function mergeComboOverrides(defaultConfig, overrides) {
  return {
    ...defaultConfig,
    ...overrides,
    extraP: { ...defaultConfig.extraP, ...(overrides.extraP || {}) },
    ftrlP: { ...defaultConfig.ftrlP, ...(overrides.ftrlP || {}) },
    selectedHmoeRegimes: overrides.selectedHmoeRegimes
      ? [...overrides.selectedHmoeRegimes]
      : [...defaultConfig.selectedHmoeRegimes],
  };
}

export function buildDefaultGridSearchConfig(algoId) {
  return {
    algoId,
    lossType: DEFAULT_LOSS_TYPE,
    useGrad: DEFAULT_USE_GRAD,
    extraP: { ...DEFAULT_EXTRA_PARAMS },
    ftrlP: { ...DEFAULT_FTRL_PARAMS },
    selectedHmoeRegimes: [...DEFAULT_HMOE_REGIME_IDS],
  };
}

export function createGridSearchCombo(algoId, comboId, overrides = {}) {
  const merged = mergeComboOverrides(buildDefaultGridSearchConfig(algoId), overrides);
  delete merged.id;
  return {
    ...merged,
    id: comboId,
  };
}

export function getInitialGridSearchComboOverrides(algoId) {
  switch (algoId) {
    case "BOA":
    case "MLpol":
    case "MLprod":
    case "FTRL":
    case "HMOE_BOA":
    case "HMOE_MLpol":
    case "HMOE_MLprod":
    case "HMOE_FTRL":
      return [{}, { useGrad: false }];
    case "TrimmedMean":
      return [{}, { extraP: { trim: 10 } }];
    case "InvMSE":
    case "BestExpert":
      return [{}, { extraP: { window: 24 } }];
    case "Ridge":
      return [{}, { extraP: { alpha: 5 } }];
    default:
      return [{}];
  }
}

export function getGridSearchControlSections(algoId, lossTypes, hmoeRegimes) {
  const sections = [];
  const algo = ALGOS.find((entry) => entry.id === algoId);
  const isOperaFamily = OPERA_ALGO_IDS.includes(algoId) || HMOE_ALGO_IDS.includes(algoId);
  const isFtrlFamily = algoId === "FTRL" || algoId === "HMOE_FTRL";

  if (isOperaFamily) {
    sections.push({
      id: "opera",
      title: "Parametres Opera",
      controls: [
        {
          id: "lossType",
          label: "Loss function",
          type: "select",
          scope: "root",
          options: lossTypes.map((loss) => ({ value: loss.id, label: loss.label })),
        },
        {
          id: "useGrad",
          label: "Gradient mode",
          type: "toggle",
          scope: "root",
        },
      ],
    });
  }

  if (isFtrlFamily && algo?.params?.length) {
    sections.push({
      id: "ftrl",
      title: "Parametres FTRL",
      controls: algo.params.map((param) => ({ ...param, scope: "ftrlP" })),
    });
  } else if (algo?.params?.length) {
    sections.push({
      id: "extra",
      title: "Parametres specifiques",
      controls: algo.params.map((param) => ({ ...param, scope: "extraP" })),
    });
  }

  if (HMOE_ALGO_IDS.includes(algoId)) {
    sections.push({
      id: "regimes",
      title: "Regimes HMOE",
      controls: [
        {
          id: "selectedHmoeRegimes",
          label: "Regimes HMOE",
          type: "multiselect",
          scope: "root",
          options: hmoeRegimes.map((regime) => ({
            value: regime.id,
            label: regime.label,
            help: regime.describeFeatures,
          })),
        },
      ],
    });
  }

  return sections;
}

function getRelevantComboPayload(algoId, combo) {
  const payload = {};
  const isOperaFamily = OPERA_ALGO_IDS.includes(algoId) || HMOE_ALGO_IDS.includes(algoId);

  if (isOperaFamily) {
    payload.lossType = combo.lossType;
    payload.useGrad = combo.useGrad;
  }
  if (algoId === "FTRL" || algoId === "HMOE_FTRL") {
    payload.ftrlP = {
      eta0: combo.ftrlP.eta0,
      tol: combo.ftrlP.tol,
      maxiter: combo.ftrlP.maxiter,
    };
  }
  if (algoId === "TrimmedMean") {
    payload.extraP = { trim: combo.extraP.trim };
  }
  if (algoId === "InvMSE" || algoId === "BestExpert") {
    payload.extraP = { window: combo.extraP.window };
  }
  if (algoId === "Ridge") {
    payload.extraP = { alpha: combo.extraP.alpha };
  }
  if (HMOE_ALGO_IDS.includes(algoId)) {
    payload.selectedHmoeRegimes = [...combo.selectedHmoeRegimes].sort();
  }

  return payload;
}

export function getGridSearchComboSignature(algoId, combo) {
  return JSON.stringify(getRelevantComboPayload(algoId, combo));
}

export function buildGridSearchComboLabel(algoId, combo, index) {
  return buildAlgoRunLabel(algoId, combo, "aléatoire");
}

export function getGridSearchComboDisplayTitle(index) {
  return `Combinaison ${index + 1}`;
}
