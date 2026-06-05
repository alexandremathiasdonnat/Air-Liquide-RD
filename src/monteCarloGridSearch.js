import { MONTE_CARLO_COMPLEXITY } from "./aggregationCatalog";
import { runAggregation } from "./aggregationRunner";
import { buildRankings, calcMetrics } from "./metrics";
import { buildDataWithRandExperts, generateRandExperts } from "./randomExperts";

const BASELINE_ROWS = 7000;

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

function getGenerationWeight(randomConfig) {
  return 0.5 + randomConfig.nExperts * 0.05 + (randomConfig.phaseMin + randomConfig.phaseMax) / 40 + randomConfig.noiseLevel * 0.4;
}

function getMethodWeight(algoId) {
  return MONTE_CARLO_COMPLEXITY[algoId] || 2;
}

function yieldToUi() {
  return new Promise((resolve) => {
    setTimeout(resolve, 0);
  });
}

function throwIfAborted(signal) {
  if (signal?.aborted) {
    const error = new Error("Monte Carlo gridsearch aborted.");
    error.name = "AbortError";
    throw error;
  }
}

export function estimateMonteCarloGridSearchMs({
  rowCount,
  simulationCount,
  randomConfig,
  algoId,
  comboCount,
}) {
  const rowFactor = getRowFactor(rowCount);
  const generationMs = 6
    * getGenerationWeight(randomConfig)
    * rowFactor
    * getPhaseFactor(randomConfig.phaseMin, randomConfig.phaseMax)
    * getNoiseFactor(randomConfig.noiseLevel);
  const comboMethodMs = 14
    * getMethodWeight(algoId)
    * rowFactor
    * getExpertFactor(randomConfig.nExperts)
    * getPhaseFactor(randomConfig.phaseMin, randomConfig.phaseMax)
    * getNoiseFactor(randomConfig.noiseLevel);
  return Math.round((generationMs + comboMethodMs * comboCount) * simulationCount);
}

export async function runMonteCarloGridSearch({
  rows,
  simulationCount,
  algoId,
  combos,
  randomConfig,
  syntheticIds,
  onProgress,
  signal,
}) {
  const generationWeight = getGenerationWeight(randomConfig);
  const methodWeight = getMethodWeight(algoId);
  const totalWeight = simulationCount * (generationWeight + combos.length * methodWeight);
  let completedWeight = 0;
  const aggregates = {};
  const startedAt = performance.now();

  const emitProgress = ({ simulationIndex, comboIndex = -1, comboLabel = null, stage }) => {
    const now = performance.now();
    const elapsedMs = now - startedAt;
    const ratio = totalWeight === 0 ? 1 : Math.min(completedWeight / totalWeight, 1);
    const remainingMs = ratio > 0 && ratio < 1 ? elapsedMs * ((1 - ratio) / ratio) : 0;
    if (onProgress) {
      onProgress({
        stage,
        simulationIndex,
        comboIndex,
        currentComboLabel: comboLabel,
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

    for (let comboIndex = 0; comboIndex < combos.length; comboIndex += 1) {
      throwIfAborted(signal);
      const combo = combos[comboIndex];
      await yieldToUi();
      throwIfAborted(signal);
      const run = await runAggregation(
        augRows,
        expertColumns,
        algoId,
        combo.lossType,
        combo.useGrad,
        combo.extraP,
        combo.ftrlP,
        combo.selectedHmoeRegimes,
      );
      const metrics = calcMetrics(run.predictions.slice(-24), rows.slice(-24));
      if (!aggregates[combo.id]) {
        aggregates[combo.id] = {
          id: combo.id,
          label: combo.label,
          mae: 0,
          rmse: 0,
          mape: 0,
          count: 0,
        };
      }
      aggregates[combo.id].mae += metrics.mae;
      aggregates[combo.id].rmse += metrics.rmse;
      aggregates[combo.id].mape += metrics.mape;
      aggregates[combo.id].count += 1;
      completedWeight += methodWeight;
      emitProgress({ simulationIndex, comboIndex, comboLabel: combo.label, stage: "gridsearch" });
    }
  }

  const averages = combos.map((combo) => {
    const aggregate = aggregates[combo.id];
    return {
      ...aggregate,
      mae: aggregate.mae / aggregate.count,
      rmse: aggregate.rmse / aggregate.count,
      mape: aggregate.mape / aggregate.count,
    };
  });

  completedWeight = totalWeight;
  emitProgress({ simulationIndex: simulationCount, comboIndex: combos.length, stage: "done" });

  return {
    simulationCount,
    rowCount: rows.length,
    averages,
    rankings: buildRankings(averages),
  };
}
