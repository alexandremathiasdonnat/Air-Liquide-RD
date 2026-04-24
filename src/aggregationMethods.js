import { vnorm } from "./moe";

function solveLinear(A, b) {
  const n = b.length;
  const matrix = A.map((row, index) => [...row, b[index]]);
  for (let col = 0; col < n; col += 1) {
    let maxRow = col;
    for (let row = col + 1; row < n; row += 1) {
      if (Math.abs(matrix[row][col]) > Math.abs(matrix[maxRow][col])) {
        maxRow = row;
      }
    }
    [matrix[col], matrix[maxRow]] = [matrix[maxRow], matrix[col]];
    for (let row = col + 1; row < n; row += 1) {
      const factor = matrix[row][col] / matrix[col][col];
      for (let cursor = col; cursor <= n; cursor += 1) {
        matrix[row][cursor] -= factor * matrix[col][cursor];
      }
    }
  }
  const solution = new Array(n).fill(0);
  for (let row = n - 1; row >= 0; row -= 1) {
    solution[row] = matrix[row][n];
    for (let col = row + 1; col < n; col += 1) {
      solution[row] -= matrix[row][col] * solution[col];
    }
    solution[row] /= matrix[row][row];
  }
  return solution;
}

export function runSimpleMean(data, cols) {
  const expertCount = cols.length;
  return {
    predictions: data.map((row) => cols.reduce((sum, column) => sum + (row[column] || 0), 0) / expertCount),
    weightHistory: data.map(() => new Array(expertCount).fill(1 / expertCount)),
  };
}

export function runMedian(data, cols) {
  const expertCount = cols.length;
  const predictions = [];
  const weightHistory = [];
  for (let index = 0; index < data.length; index += 1) {
    const values = cols.map((column) => data[index][column] || 0);
    const sorted = [...values].sort((left, right) => left - right);
    const median = expertCount % 2 === 0
      ? (sorted[expertCount / 2 - 1] + sorted[expertCount / 2]) / 2
      : sorted[Math.floor(expertCount / 2)];
    const distances = values.map((value) => Math.abs(value - median) + 1e-8);
    predictions.push(median);
    weightHistory.push(vnorm(distances.map((distance) => 1 / distance)));
  }
  return { predictions, weightHistory };
}

export function runTrimmedMean(data, cols, params) {
  const expertCount = cols.length;
  const trimmedPerSide = Math.max(0, Math.floor((expertCount * (params.trim || 20)) / 100 / 2));
  const predictions = [];
  const weightHistory = [];
  for (let index = 0; index < data.length; index += 1) {
    const values = cols.map((column) => data[index][column] || 0);
    const sortedIndices = values.map((value, idx) => ({ value, idx })).sort((left, right) => left.value - right.value);
    const kept = sortedIndices.slice(trimmedPerSide, expertCount - trimmedPerSide);
    const prediction = kept.reduce((sum, item) => sum + item.value, 0) / kept.length;
    const weights = new Array(expertCount).fill(0);
    kept.forEach((item) => {
      weights[item.idx] = 1 / kept.length;
    });
    predictions.push(prediction);
    weightHistory.push(weights);
  }
  return { predictions, weightHistory };
}

export function runInvMSE(data, cols, params) {
  const expertCount = cols.length;
  const window = params.window || 48;
  const predictions = [];
  const weightHistory = [];
  for (let index = 0; index < data.length; index += 1) {
    const values = cols.map((column) => data[index][column] || 0);
    let weights;
    if (index < 2) {
      weights = new Array(expertCount).fill(1 / expertCount);
    } else {
      const slice = data.slice(Math.max(0, index - window), index);
      const mses = cols.map((column) => {
        const errors = slice.map((row) => (row[column] || 0) - row.y_true);
        return errors.reduce((sum, error) => sum + error ** 2, 0) / slice.length + 1e-6;
      });
      weights = vnorm(mses.map((mse) => 1 / mse));
    }
    predictions.push(weights.reduce((sum, weight, expertIndex) => sum + weight * values[expertIndex], 0));
    weightHistory.push([...weights]);
  }
  return { predictions, weightHistory };
}

export function runBestExpert(data, cols, params) {
  const expertCount = cols.length;
  const window = params.window || 48;
  const predictions = [];
  const weightHistory = [];
  for (let index = 0; index < data.length; index += 1) {
    const values = cols.map((column) => data[index][column] || 0);
    let bestIndex = 0;
    if (index >= 2) {
      const slice = data.slice(Math.max(0, index - window), index);
      const maes = cols.map((column) => {
        const errors = slice.map((row) => Math.abs((row[column] || 0) - row.y_true));
        return errors.reduce((sum, error) => sum + error, 0) / slice.length;
      });
      bestIndex = maes.indexOf(Math.min(...maes));
    }
    const weights = new Array(expertCount).fill(0);
    weights[bestIndex] = 1;
    predictions.push(values[bestIndex]);
    weightHistory.push([...weights]);
  }
  return { predictions, weightHistory };
}

export function runRidge(data, cols, params) {
  const expertCount = cols.length;
  const alpha = params.alpha || 1;
  const X = data.map((row) => cols.map((column) => row[column] || 0));
  const y = data.map((row) => row.y_true);
  const XtX = Array.from({ length: expertCount }, (_, left) => (
    Array.from({ length: expertCount }, (_, right) => (
      X.reduce((sum, row) => sum + row[left] * row[right], 0) + (left === right ? alpha : 0)
    ))
  ));
  const Xty = Array.from(
    { length: expertCount },
    (_, expertIndex) => X.reduce((sum, row, rowIndex) => sum + row[expertIndex] * y[rowIndex], 0),
  );
  const rawWeights = solveLinear(XtX, Xty);
  const displayedWeights = vnorm(rawWeights.map(Math.abs));
  return {
    predictions: X.map((row) => row.reduce((sum, value, expertIndex) => sum + value * rawWeights[expertIndex], 0)),
    weightHistory: data.map(() => [...displayedWeights]),
  };
}
