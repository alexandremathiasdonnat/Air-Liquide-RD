import { runHmoe } from "./hmoe";
import { runBOA, runFTRL, runMLpol, runMLprod } from "./moe";
import { HMOE_ALGO_IDS } from "./aggregationCatalog";
import {
  runBestExpert,
  runInvMSE,
  runMedian,
  runRidge,
  runSimpleMean,
  runTrimmedMean,
} from "./aggregationMethods";

export function getHmoeBaseAlgoId(algoId) {
  return algoId.startsWith("HMOE_") ? algoId.replace("HMOE_", "") : algoId;
}

export function runAggregation(data, cols, algoId, lossType, useGrad, extraP, ftrlP, selectedHmoeRegimes) {
  if (HMOE_ALGO_IDS.includes(algoId)) {
    return runHmoe(
      data,
      cols,
      getHmoeBaseAlgoId(algoId),
      lossType,
      useGrad,
      extraP,
      ftrlP,
      selectedHmoeRegimes,
    );
  }

  switch (algoId) {
    case "BOA":
      return runBOA(data, cols, lossType, useGrad);
    case "MLpol":
      return runMLpol(data, cols, lossType, useGrad);
    case "MLprod":
      return runMLprod(data, cols, lossType, useGrad);
    case "FTRL":
      return runFTRL(data, cols, lossType, useGrad, ftrlP);
    case "SimpleMean":
      return runSimpleMean(data, cols);
    case "Median":
      return runMedian(data, cols);
    case "TrimmedMean":
      return runTrimmedMean(data, cols, extraP);
    case "InvMSE":
      return runInvMSE(data, cols, extraP);
    case "BestExpert":
      return runBestExpert(data, cols, extraP);
    case "Ridge":
      return runRidge(data, cols, extraP);
    default:
      return runBOA(data, cols, lossType, useGrad);
  }
}
