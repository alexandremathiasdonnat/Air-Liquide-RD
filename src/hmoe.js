const EPSILON = 1e-8;

function toNumber(value, fallback = 0) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function normalize(weights) {
  const sum = weights.reduce((acc, value) => acc + value, 0);
  if (!Number.isFinite(sum) || Math.abs(sum) < EPSILON) {
    return weights.map(() => 1 / weights.length);
  }
  return weights.map((value) => value / sum);
}

function dot(left, right) {
  return left.reduce((acc, value, index) => acc + value * right[index], 0);
}

function mean(values) {
  if (!values.length) {
    return 0;
  }
  return values.reduce((acc, value) => acc + value, 0) / values.length;
}

function sampleStd(values) {
  if (values.length <= 1) {
    return 0;
  }
  const avg = mean(values);
  const variance =
    values.reduce((acc, value) => acc + (value - avg) ** 2, 0) /
    (values.length - 1);
  return Math.sqrt(Math.max(variance, 0));
}

function rollingStd(values) {
  if (values.length <= 1) {
    return 0;
  }
  return sampleStd(values);
}

function rollingTrendSlope(values) {
  if (values.length <= 1) {
    return 0;
  }

  const avgX = (values.length - 1) / 2;
  const avgY = mean(values);
  let numerator = 0;
  let denominator = 0;

  for (let index = 0; index < values.length; index += 1) {
    const centeredX = index - avgX;
    numerator += centeredX * (values[index] - avgY);
    denominator += centeredX ** 2;
  }

  return denominator > EPSILON ? numerator / denominator : 0;
}

const lossFunctions = {
  mse: (pred, actual) => (pred - actual) ** 2,
  mae: (pred, actual) => Math.abs(pred - actual),
  mape: (pred, actual) => Math.abs(pred - actual) / (Math.abs(actual) + EPSILON),
  msle: (pred, actual) =>
    (Math.log(Math.max(actual, 0) + 1) - Math.log(Math.max(pred, 0) + 1)) ** 2,
  mspe: (pred, actual) => ((actual - pred) / (Math.abs(actual) + EPSILON)) ** 2,
};

const gradientFunctions = {
  mse: (pred, actual) => 2 * (pred - actual),
  mae: (pred, actual) => Math.sign(pred - actual),
  mape: (pred, actual) => Math.sign(pred - actual) / (Math.abs(actual) + EPSILON),
  msle: (pred, actual) =>
    2 *
    (Math.log(Math.max(actual, 0) + 1) - Math.log(Math.max(pred, 0) + 1)) *
    (-1 / (Math.max(pred, 0) + 1)),
  mspe: (pred, actual) => -2 * pred + 2 * actual,
};

function computeRelativeLoss(yHat, yTrue, expertPredictions, lossType, useGradient) {
  if (useGradient) {
    const gradient = gradientFunctions[lossType](yHat, yTrue);
    return expertPredictions.map((expertPrediction) => gradient * (yHat - expertPrediction));
  }

  const aggregateLoss = lossFunctions[lossType](yHat, yTrue);
  return expertPredictions.map(
    (expertPrediction) => aggregateLoss - lossFunctions[lossType](expertPrediction, yTrue),
  );
}

function cloneRow(row) {
  return { ...row };
}

function computeTemporalFeatures(targetTime) {
  const date = new Date(targetTime);
  if (Number.isNaN(date.getTime())) {
    return { hourSin: 0, hourCos: 1, utcHour: 0 };
  }
  const utcHour =
    date.getUTCHours() +
    date.getUTCMinutes() / 60 +
    date.getUTCSeconds() / 3600;
  return {
    utcHour,
    hourSin: Math.sin((2 * Math.PI * utcHour) / 24),
    hourCos: Math.cos((2 * Math.PI * utcHour) / 24),
  };
}

export function ensureHmoeFeatures(rows) {
  const enriched = rows.map(cloneRow);
  const targets = enriched.map((row) => toNumber(row.y_true));

  for (let index = 0; index < enriched.length; index += 1) {
    const row = enriched[index];
    const { hourSin, hourCos } = computeTemporalFeatures(row.target_time);

    if (!Number.isFinite(Number(row.hour_sin))) {
      row.hour_sin = hourSin;
    }
    if (!Number.isFinite(Number(row.hour_cos))) {
      row.hour_cos = hourCos;
    }

    if (!Number.isFinite(Number(row.mom_24))) {
      row.mom_24 = index >= 25 ? targets[index - 1] - targets[index - 25] : 0;
    }
    if (!Number.isFinite(Number(row.mom_48))) {
      row.mom_48 = index >= 49 ? targets[index - 1] - targets[index - 49] : 0;
    }

    if (!Number.isFinite(Number(row.vol_24))) {
      const returns24 = [];
      for (let inner = Math.max(1, index - 24); inner <= index - 1; inner += 1) {
        returns24.push(targets[inner] - targets[inner - 1]);
      }
      row.vol_24 = rollingStd(returns24);
    }
    if (!Number.isFinite(Number(row.vol_48))) {
      const returns48 = [];
      for (let inner = Math.max(1, index - 48); inner <= index - 1; inner += 1) {
        returns48.push(targets[inner] - targets[inner - 1]);
      }
      row.vol_48 = rollingStd(returns48);
    }

    if (!Number.isFinite(Number(row.trend_24_gap))) {
      // Day-ahead trend: use a 24-point rolling window shifted by 24 steps,
      // so the current target never sees the most recent 24 future-adjacent values.
      const trendWindow =
        index >= 47 ? targets.slice(index - 47, index - 23) : [];
      row.trend_24_gap = rollingTrendSlope(trendWindow);
    }

    if (!Number.isFinite(Number(row.wind_norm))) {
      row.wind_norm = 0;
    }
  }

  return enriched;
}

export const HMOE_REGIME_TYPES = [
  {
    id: "day_night",
    label: "Day / Night",
    features: ["hour_sin", "hour_cos"],
    components: ["Day", "Night"],
    describeFeatures: "hour_sin, hour_cos",
    directionHint: (row) => {
      const { utcHour } = computeTemporalFeatures(row.target_time);
      return utcHour >= 6 && utcHour < 18 ? 1 : -1;
    },
  },
  {
    id: "wind",
    label: "Wind",
    features: ["wind_norm"],
    components: ["High wind", "Low wind"],
    describeFeatures: "wind_norm",
    directionHint: (_row, vector) => (vector[0] >= 0 ? 1 : -1),
  },
  {
    id: "updown",
    label: "Up / Down",
    features: ["mom_24", "mom_48"],
    components: ["Bull", "Bear"],
    describeFeatures: "mom_24, mom_48",
    directionHint: (_row, vector) => (vector[0] + vector[1] >= 0 ? 1 : -1),
  },
  {
    id: "volatility",
    label: "Volatility",
    features: ["vol_24", "vol_48"],
    components: ["High vol", "Low vol"],
    describeFeatures: "vol_24, vol_48",
    directionHint: (_row, vector) => (vector[0] + vector[1] >= 0 ? 1 : -1),
  },
  {
    id: "trend",
    label: "Trend",
    features: ["trend_24_gap"],
    components: ["High level", "Low level"],
    describeFeatures: "y_true (with a 24 values rolling gap)",
    directionHint: (_row, vector) => (vector[0] >= 0 ? 1 : -1),
  },
];

class BinaryRegimeGate {
  constructor({ learningRate = 0.08, strength = 0.12, exploration = 0.05 } = {}) {
    this.learningRate = learningRate;
    this.strength = strength;
    this.exploration = exploration;
    this.weights = null;
  }

  init(featureCount) {
    this.weights = Array.from({ length: featureCount }, () => [0, 0]);
  }

  predict(featureVector) {
    if (!this.weights) {
      this.init(featureVector.length);
    }

    const logits = [0, 0];
    for (let featureIndex = 0; featureIndex < featureVector.length; featureIndex += 1) {
      logits[0] += featureVector[featureIndex] * this.weights[featureIndex][0];
      logits[1] += featureVector[featureIndex] * this.weights[featureIndex][1];
    }

    const maxLogit = Math.max(...logits);
    const scaled = logits.map((logit) => Math.exp((logit - maxLogit) / 2));
    const scaledSum = scaled.reduce((acc, value) => acc + value, 0);
    const probabilities = scaled.map((value) => value / (scaledSum || 1));

    return probabilities.map(
      (value) => this.exploration / 2 + (1 - this.exploration) * value,
    );
  }

  update(featureVector, losses, directionHint = 0) {
    const probabilities = this.predict(featureVector);
    const lossAverage = mean(losses);
    const centeredLosses = losses.map((value) => value - lossAverage);
    const lossStd = sampleStd(centeredLosses);
    const scaledLosses =
      lossStd > EPSILON
        ? centeredLosses.map((value) => value / lossStd)
        : centeredLosses.slice();

    if (directionHint > 0) {
      scaledLosses[0] -= this.strength;
      scaledLosses[1] += this.strength;
    } else if (directionHint < 0) {
      scaledLosses[0] += this.strength;
      scaledLosses[1] -= this.strength;
    }

    const baseline = probabilities.reduce(
      (acc, probability, index) => acc + probability * scaledLosses[index],
      0,
    );

    for (let featureIndex = 0; featureIndex < featureVector.length; featureIndex += 1) {
      for (let componentIndex = 0; componentIndex < 2; componentIndex += 1) {
        const gradient =
          featureVector[featureIndex] * (scaledLosses[componentIndex] - baseline);
        this.weights[featureIndex][componentIndex] -= this.learningRate * gradient;
      }
    }
  }
}

function createBoaState(expertCount, lossType, useGradient) {
  const floor = 1 / 2 ** 20;
  let cumulativeVariance = new Array(expertCount).fill(floor);
  let maxLoss = new Array(expertCount).fill(floor);
  let cumulativeRegret = new Array(expertCount).fill(0);
  let learningRates = new Array(expertCount).fill(floor);
  let lastStep = null;

  function currentWeights() {
    const rawScores = learningRates.map(
      (rate, expertIndex) =>
        Math.log(rate) + Math.log(1 / expertCount) + rate * cumulativeRegret[expertIndex],
    );
    const maxScore = Math.max(...rawScores);
    return normalize(rawScores.map((score) => Math.exp(score - maxScore)));
  }

  return {
    predict(expertPredictions) {
      const weights = currentWeights();
      const prediction = dot(weights, expertPredictions);
      lastStep = { weights, prediction };
      return prediction;
    },
    update(expertPredictions, target) {
      if (!lastStep) {
        this.predict(expertPredictions);
      }

      const relativeLoss = computeRelativeLoss(
        lastStep.prediction,
        target,
        expertPredictions,
        lossType,
        useGradient,
      );
      const squaredLoss = relativeLoss.map((value) => value ** 2);
      maxLoss = maxLoss.map((value, index) => Math.max(value, Math.abs(relativeLoss[index])));
      const bounds = maxLoss.map((value) => 2 ** Math.ceil(Math.log2(value + 1e-30)));
      cumulativeVariance = cumulativeVariance.map(
        (value, index) => value + squaredLoss[index],
      );
      learningRates = learningRates.map((_, index) =>
        Math.min(1 / bounds[index], Math.sqrt(Math.log(expertCount) / cumulativeVariance[index])),
      );
      cumulativeRegret = cumulativeRegret.map(
        (value, index) =>
          value +
          0.5 *
            (relativeLoss[index] -
              learningRates[index] * squaredLoss[index] +
              bounds[index] * (learningRates[index] * relativeLoss[index] > 0.5 ? 1 : 0)),
      );
      lastStep = null;
    },
    getWeights() {
      return lastStep ? lastStep.weights.slice() : currentWeights();
    },
  };
}

function createMlPolState(expertCount, lossType, useGradient) {
  const floor = 1 / 2 ** 20;
  let cumulativeRegret = new Array(expertCount).fill(0);
  let learningRates = new Array(expertCount).fill(floor);
  let maxSquaredRegret = new Array(expertCount).fill(0);
  let lastStep = null;

  function currentWeights() {
    const positiveRegret = cumulativeRegret.map((value) => Math.max(value, 0));
    const rawWeights = learningRates.map(
      (rate, expertIndex) => rate * positiveRegret[expertIndex],
    );
    const total = rawWeights.reduce((acc, value) => acc + value, 0);
    return total === 0 ? new Array(expertCount).fill(1 / expertCount) : normalize(rawWeights);
  }

  return {
    predict(expertPredictions) {
      const weights = currentWeights();
      const prediction = dot(weights, expertPredictions);
      lastStep = { weights, prediction };
      return prediction;
    },
    update(expertPredictions, target) {
      if (!lastStep) {
        this.predict(expertPredictions);
      }

      const relativeLoss = computeRelativeLoss(
        lastStep.prediction,
        target,
        expertPredictions,
        lossType,
        useGradient,
      );
      const squaredLoss = relativeLoss.map((value) => value ** 2);
      cumulativeRegret = cumulativeRegret.map((value, index) => value + relativeLoss[index]);
      const diff = Math.max(Math.max(...squaredLoss) - Math.max(...maxSquaredRegret), 0);
      maxSquaredRegret = maxSquaredRegret.map((value) => value + diff);
      learningRates = learningRates.map(
        (rate, index) => 1 / (1 / rate + squaredLoss[index] + diff),
      );
      lastStep = null;
    },
    getWeights() {
      return lastStep ? lastStep.weights.slice() : currentWeights();
    },
  };
}

function createMlProdState(expertCount, lossType, useGradient) {
  const floor = 1 / 2 ** 20;
  let cumulativeVariance = new Array(expertCount).fill(floor);
  let maxLoss = new Array(expertCount).fill(floor);
  let cumulativeRegret = new Array(expertCount).fill(0);
  let learningRates = new Array(expertCount).fill(floor);
  let lastStep = null;

  function currentWeights() {
    return normalize(
      learningRates.map((rate, expertIndex) => rate * Math.exp(cumulativeRegret[expertIndex])),
    );
  }

  return {
    predict(expertPredictions) {
      const weights = currentWeights();
      const prediction = dot(weights, expertPredictions);
      lastStep = { weights, prediction };
      return prediction;
    },
    update(expertPredictions, target) {
      if (!lastStep) {
        this.predict(expertPredictions);
      }

      const relativeLoss = computeRelativeLoss(
        lastStep.prediction,
        target,
        expertPredictions,
        lossType,
        useGradient,
      );
      const squaredLoss = relativeLoss.map((value) => value ** 2);
      cumulativeVariance = cumulativeVariance.map(
        (value, index) => value + squaredLoss[index],
      );
      maxLoss = maxLoss.map((value, index) => Math.max(value, Math.abs(relativeLoss[index])));
      const nextLearningRates = learningRates.map((_, index) =>
        Math.min(
          Math.min(
            0.5 / (maxLoss[index] + 1e-30),
            Math.sqrt(Math.log(expertCount + 1) / (cumulativeVariance[index] + 1e-30)),
          ),
          1e30,
        ),
      );
      cumulativeRegret = cumulativeRegret.map(
        (value, index) =>
          (nextLearningRates[index] / (learningRates[index] + 1e-30)) * value +
          Math.log(1 + nextLearningRates[index] * relativeLoss[index] + 1e-30),
      );
      learningRates = nextLearningRates;
      lastStep = null;
    },
    getWeights() {
      return lastStep ? lastStep.weights.slice() : currentWeights();
    },
  };
}

function createFtrlState(expertCount, lossType, initialParams = {}) {
  let weights = new Array(expertCount).fill(1 / expertCount);
  let gradients = new Array(expertCount).fill(0);
  let eta = initialParams.eta0 || 0.01;

  return {
    predict(expertPredictions) {
      return dot(weights, expertPredictions);
    },
    update(expertPredictions, target) {
      const prediction = dot(weights, expertPredictions);
      const gradient = gradientFunctions[lossType](prediction, target);
      const gradientVector = expertPredictions.map((expertPrediction) => gradient * expertPrediction);
      eta =
        1 /
        Math.sqrt(
          1 / eta ** 2 +
            gradientVector.reduce((acc, value) => acc + value ** 2, 0) +
            1e-30,
        );
      gradients = gradients.map((value, index) => value + gradientVector[index]);
      weights = normalize(
        weights.map((value, index) => value * Math.exp(-eta * gradients[index])),
      );
    },
    getWeights() {
      return weights.slice();
    },
  };
}

function createOperaState(algoId, expertCount, lossType, useGradient, extraParams, ftrlParams) {
  switch (algoId) {
    case "BOA":
      return createBoaState(expertCount, lossType, useGradient);
    case "MLpol":
      return createMlPolState(expertCount, lossType, useGradient);
    case "MLprod":
      return createMlProdState(expertCount, lossType, useGradient);
    case "FTRL":
      return createFtrlState(expertCount, lossType, ftrlParams);
    default:
      throw new Error(`Unsupported HMOE base algorithm: ${algoId}`);
  }
}

function buildFeatureStats(rows, regimeTypes) {
  const featureNames = [...new Set(regimeTypes.flatMap((regimeType) => regimeType.features))];
  return featureNames.reduce((stats, featureName) => {
    const values = rows.map((row) => toNumber(row[featureName])).filter(Number.isFinite);
    const avg = mean(values);
    const std = sampleStd(values);
    stats[featureName] = { mean: avg, std: std > EPSILON ? std : 1 };
    return stats;
  }, {});
}

function buildFeatureVector(row, regimeType, featureStats) {
  return regimeType.features.map((featureName) => {
    const featureValue = toNumber(row[featureName]);
    const { mean: avg, std } = featureStats[featureName];
    return (featureValue - avg) / std;
  });
}

export function runHmoe(
  rows,
  expertColumns,
  baseAlgoId,
  lossType,
  useGradient,
  extraParams,
  ftrlParams,
  selectedRegimeIds,
) {
  const enrichedRows = ensureHmoeFeatures(rows);
  const selectedRegimeTypes = HMOE_REGIME_TYPES.filter((regimeType) =>
    selectedRegimeIds.includes(regimeType.id),
  );

  if (!selectedRegimeTypes.length) {
    throw new Error("HMOE needs at least one regime type.");
  }

  const featureStats = buildFeatureStats(enrichedRows, selectedRegimeTypes);
  const regimeStates = selectedRegimeTypes.map((regimeType) => ({
    id: regimeType.id,
    label: regimeType.label,
    components: regimeType.components.slice(),
    describeFeatures: regimeType.describeFeatures,
    gate: new BinaryRegimeGate(),
    models: regimeType.components.map(() =>
      createOperaState(
        baseAlgoId,
        expertColumns.length,
        lossType,
        useGradient,
        extraParams,
        ftrlParams,
      ),
    ),
    buildFeatureVector: (row) => buildFeatureVector(row, regimeType, featureStats),
    directionHint: (row, featureVector) => regimeType.directionHint(row, featureVector),
  }));

  const predictions = [];
  const weightHistory = [];
  const regimeHistory = [];
  const testStart = Math.max(1, enrichedRows.length - 24);

  for (let rowIndex = 0; rowIndex < enrichedRows.length; rowIndex++) {
    const row = enrichedRows[rowIndex];
    const isTrain = rowIndex < testStart;
    const expertPredictions = expertColumns.map((columnName) => toNumber(row[columnName]));
    const target = toNumber(row.y_true);
    let aggregatePrediction = 0;
    let aggregateWeights = new Array(expertColumns.length).fill(0);
    const stepRegimes = {};

    for (const regimeState of regimeStates) {
      const featureVector = regimeState.buildFeatureVector(row);
      const probabilities = regimeState.gate.predict(featureVector);
      const branchPredictions = regimeState.models.map((model) => model.predict(expertPredictions));
      const branchWeights = regimeState.models.map((model) => model.getWeights());
      const regimePrediction = branchPredictions.reduce(
        (acc, prediction, index) => acc + probabilities[index] * prediction,
        0,
      );
      const regimeWeights = new Array(expertColumns.length).fill(0);

      for (let branchIndex = 0; branchIndex < branchWeights.length; branchIndex += 1) {
        for (let expertIndex = 0; expertIndex < branchWeights[branchIndex].length; expertIndex += 1) {
          regimeWeights[expertIndex] += probabilities[branchIndex] * branchWeights[branchIndex][expertIndex];
        }
      }

      aggregatePrediction += regimePrediction;
      aggregateWeights = aggregateWeights.map(
        (value, index) => value + regimeWeights[index],
      );

      const regimeLosses = branchPredictions.map((prediction) =>
        lossFunctions[lossType](prediction, target),
      );
      const dominantBranch = probabilities[0] >= probabilities[1] ? 0 : 1;
      if (isTrain) {
        regimeState.models[dominantBranch].update(expertPredictions, target);
        regimeState.gate.update(
          featureVector,
          regimeLosses,
          regimeState.directionHint(row, featureVector),
        );
      }

      stepRegimes[regimeState.id] = {
        probabilities: probabilities.slice(),
        dominantBranch,
      };
    }

    const regimeCount = regimeStates.length;
    predictions.push(aggregatePrediction / regimeCount);
    weightHistory.push(normalize(aggregateWeights.map((value) => value / regimeCount)));
    regimeHistory.push(stepRegimes);
  }

  return {
    predictions,
    weightHistory,
    hmoe: {
      selectedRegimes: regimeStates.map((regimeState) => ({
        id: regimeState.id,
        label: regimeState.label,
        components: regimeState.components.slice(),
        describeFeatures: regimeState.describeFeatures,
      })),
      regimeHistory,
    },
  };
}
