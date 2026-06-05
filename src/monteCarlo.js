import { MONTE_CARLO_COMPLEXITY } from "./aggregationCatalog";
import { buildAlgoRunLabel } from "./monteCarloConfig";
import { runAggregation } from "./aggregationRunner";
import { buildRankings, calcMetrics } from "./metrics";
import { buildDataWithRandExperts, generateRandExperts } from "./randomExperts";

const BASELINE_ROWS = 7000;
const BASE_GENERATION_MS = 6;
const BASE_METHOD_MS = 14;

function getRowFactor(rowCount) {
  return Math.max(0.35, rowCount / BASELINE_ROWS);
}

function getExpertFactor(expertCount) {
  return 0.6 + expertCount / 4.5;
}

function getPhaseFactor(phaseMin, phaseMax) {
  const averagePhases = (phaseMin + phaseMax) / 2;
  return 0.8 + averagePhases / 9;
}

function getNoiseFactor(noiseLevel) {
  return 1 + noiseLevel * 0.35;
}

function getMethodWeight(algoId) {
  return MONTE_CARLO_COMPLEXITY[algoId] || 2;
}

function getGenerationWeight(randomConfig) {
  return 0.5 + randomConfig.nExperts * 0.05 + (randomConfig.phaseMin + randomConfig.phaseMax) / 40 + randomConfig.noiseLevel * 0.4;
}

function getMethodEstimateMs(algoId, rowCount, randomConfig) {
  return BASE_METHOD_MS
    * getMethodWeight(algoId)
    * getRowFactor(rowCount)
    * getExpertFactor(randomConfig.nExperts)
    * getPhaseFactor(randomConfig.phaseMin, randomConfig.phaseMax)
    * getNoiseFactor(randomConfig.noiseLevel);
}

export function estimateMonteCarloMs({ rowCount, simulationCount, randomConfig, algoIds }) {
  const generationMs = BASE_GENERATION_MS
    * getGenerationWeight(randomConfig)
    * getRowFactor(rowCount)
    * getPhaseFactor(randomConfig.phaseMin, randomConfig.phaseMax)
    * getNoiseFactor(randomConfig.noiseLevel);
  const oneSimulationMs = generationMs + algoIds.reduce(
    (sum, algoId) => sum + getMethodEstimateMs(algoId, rowCount, randomConfig),
    0,
  );
  return Math.round(oneSimulationMs * simulationCount);
}

function getTotalProgressWeight(simulationCount, algoIds, randomConfig) {
  const generationWeight = getGenerationWeight(randomConfig);
  const methodsWeight = algoIds.reduce((sum, algoId) => sum + getMethodWeight(algoId), 0);
  return simulationCount * (generationWeight + methodsWeight);
}

function yieldToUi() {
  return new Promise((resolve) => {
    setTimeout(resolve, 0);
  });
}

function throwIfAborted(signal) {
  if (signal?.aborted) {
    const error = new Error("Monte Carlo simulation aborted.");
    error.name = "AbortError";
    throw error;
  }
}

export async function runMonteCarloSimulation({
  rows,
  simulationCount,
  algoIds,
  randomConfig,
  algoRunConfigs,
  syntheticIds,
  onProgress,
  signal,
}) {
  const aggregates = {};
  const totalWeight = getTotalProgressWeight(simulationCount, algoIds, randomConfig);
  const generationWeight = getGenerationWeight(randomConfig);
  let completedWeight = 0;
  const startedAt = performance.now();

  const emitProgress = ({ simulationIndex, algoIndex = -1, algoId = null, stage }) => {
    const now = performance.now();
    const elapsedMs = now - startedAt;
    const ratio = totalWeight === 0 ? 1 : Math.min(completedWeight / totalWeight, 1);
    const remainingMs = ratio > 0 && ratio < 1 ? elapsedMs * ((1 - ratio) / ratio) : 0;
    if (onProgress) {
      onProgress({
        stage,
        simulationIndex,
        algoIndex,
        currentAlgoId: algoId,
        progress: ratio,
        elapsedMs,
        remainingMs,
        completedWeight,
        totalWeight,
      });
    }
  };

  emitProgress({ simulationIndex: 0, stage: "starting" });

  for (let simulationIndex = 0; simulationIndex < simulationCount; simulationIndex += 1) {
    throwIfAborted(signal);
    await yieldToUi();
    throwIfAborted(signal);
    const randExperts = generateRandExperts(
      rows,
      randomConfig.nExperts,
      [randomConfig.phaseMin, randomConfig.phaseMax],
      randomConfig.noiseLevel,
      syntheticIds,
    );
    const augRows = buildDataWithRandExperts(rows, randExperts);
    const expertColumns = randExperts.map((expert) => expert.id);
    completedWeight += generationWeight;
    emitProgress({ simulationIndex, stage: "generation" });

    for (let algoIndex = 0; algoIndex < algoIds.length; algoIndex += 1) {
      throwIfAborted(signal);
      const algoId = algoIds[algoIndex];
      const algoConfig = algoRunConfigs[algoId];
      if (!algoConfig) {
        throw new Error(`Missing Monte Carlo config for ${algoId}.`);
      }
      await yieldToUi();
      throwIfAborted(signal);
      const run = await runAggregation(
        augRows,
        expertColumns,
        algoId,
        algoConfig.lossType,
        algoConfig.useGrad,
        algoConfig.extraP,
        algoConfig.ftrlP,
        algoConfig.selectedHmoeRegimes,
      );
      const metrics = calcMetrics(run.predictions.slice(-24), rows.slice(-24));
      if (!aggregates[algoId]) {
        aggregates[algoId] = {
          id: algoId,
          algoId,
          label: buildAlgoRunLabel(algoId, algoConfig, "aléatoire"),
          mae: 0,
          rmse: 0,
          mape: 0,
          count: 0,
        };
      }
      aggregates[algoId].mae += metrics.mae;
      aggregates[algoId].rmse += metrics.rmse;
      aggregates[algoId].mape += metrics.mape;
      aggregates[algoId].count += 1;
      completedWeight += getMethodWeight(algoId);
      emitProgress({ simulationIndex, algoIndex, algoId, stage: "aggregation" });
    }
  }

  const averages = algoIds.map((algoId) => {
    const aggregate = aggregates[algoId];
    return {
      ...aggregate,
      mae: aggregate.mae / aggregate.count,
      rmse: aggregate.rmse / aggregate.count,
      mape: aggregate.mape / aggregate.count,
    };
  });

  completedWeight = totalWeight;
  emitProgress({ simulationIndex: simulationCount, algoIndex: algoIds.length, stage: "done" });

  return {
    simulationCount,
    rowCount: rows.length,
    averages,
    rankings: buildRankings(averages),
  };
}
