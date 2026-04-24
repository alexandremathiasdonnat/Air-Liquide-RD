export function calcMetrics(predictions, rows) {
  const count = rows.length;
  const mae = predictions.reduce((sum, prediction, index) => sum + Math.abs(prediction - rows[index].y_true), 0) / count;
  const rmse = Math.sqrt(
    predictions.reduce((sum, prediction, index) => sum + (prediction - rows[index].y_true) ** 2, 0) / count,
  );
  const mape = predictions.reduce(
    (sum, prediction, index) => sum + Math.abs(prediction - rows[index].y_true) / (Math.abs(rows[index].y_true) + 1),
    0,
  ) / count * 100;
  return { mae, rmse, mape };
}

export function buildRankings(entries) {
  if (!entries.length) {
    return null;
  }
  const byMAE = [...entries].sort((left, right) => left.mae - right.mae);
  const byRMSE = [...entries].sort((left, right) => left.rmse - right.rmse);
  const byMAPE = [...entries].sort((left, right) => left.mape - right.mape);
  const scoreMap = {};
  entries.forEach((entry) => {
    scoreMap[entry.id] = { mae: 0, rmse: 0, mape: 0 };
  });
  byMAE.forEach((entry, index) => {
    scoreMap[entry.id].mae = index + 1;
  });
  byRMSE.forEach((entry, index) => {
    scoreMap[entry.id].rmse = index + 1;
  });
  byMAPE.forEach((entry, index) => {
    scoreMap[entry.id].mape = index + 1;
  });
  const general = [...entries].sort((left, right) => {
    const leftScore = (scoreMap[left.id].mae + scoreMap[left.id].rmse + scoreMap[left.id].mape) / 3;
    const rightScore = (scoreMap[right.id].mae + scoreMap[right.id].rmse + scoreMap[right.id].mape) / 3;
    return leftScore - rightScore;
  });
  return { byMAE, byRMSE, byMAPE, general, scoreMap };
}

export function formatDuration(durationMs) {
  if (!Number.isFinite(durationMs) || durationMs <= 0) {
    return "0 s";
  }
  if (durationMs < 1000) {
    return `${Math.max(1, Math.round(durationMs))} ms`;
  }
  const totalSeconds = Math.round(durationMs / 1000);
  if (totalSeconds < 60) {
    return `${totalSeconds} s`;
  }
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  if (minutes < 60) {
    return `${minutes} min ${seconds.toString().padStart(2, "0")} s`;
  }
  const hours = Math.floor(minutes / 60);
  const remainingMinutes = minutes % 60;
  return `${hours} h ${remainingMinutes.toString().padStart(2, "0")} min`;
}
