import { ensureHmoeFeatures, runHmoe } from "./hmoe";

function buildRows(count = 72) {
  return Array.from({ length: count }, (_, index) => {
    const timestamp = new Date(Date.UTC(2025, 1, 1, index, 0, 0)).toISOString();
    const yTrue = 800 + Math.sin(index / 4) * 120 + index * 2;
    return {
      target_time: timestamp,
      y_true: yTrue,
      ridge_full: yTrue * 0.98,
      rf_full: yTrue * 1.01,
      lgbm_full: yTrue * 1.02,
      wind_norm: 20 + Math.cos(index / 5) * 4,
    };
  });
}

describe("HMOE helpers", () => {
  test("fills missing temporal regime features", () => {
    const [row] = ensureHmoeFeatures([{ target_time: "2025-02-01T06:00:00.000Z", y_true: 900 }]);
    expect(typeof row.hour_sin).toBe("number");
    expect(typeof row.hour_cos).toBe("number");
    expect(typeof row.mom_24).toBe("number");
    expect(typeof row.vol_24).toBe("number");
    expect(typeof row.trend_24_gap).toBe("number");
  });

  test.each(["BOA", "MLpol", "MLprod", "FTRL"])(
    "runs HMOE with %s base algorithm",
    (baseAlgoId) => {
      const rows = buildRows();
      const result = runHmoe(
        rows,
        ["ridge_full", "rf_full", "lgbm_full"],
        baseAlgoId,
        "mse",
        true,
        { window: 24 },
        { eta0: 0.01 },
        ["day_night", "wind", "updown", "volatility", "trend"],
      );

      expect(result.predictions).toHaveLength(rows.length);
      expect(result.weightHistory).toHaveLength(rows.length);
      expect(result.hmoe.selectedRegimes).toHaveLength(5);
      result.weightHistory.forEach((weights) => {
        const sum = weights.reduce((acc, value) => acc + value, 0);
        expect(sum).toBeCloseTo(1, 6);
        weights.forEach((value) => expect(Number.isFinite(value)).toBe(true));
      });
      result.predictions.forEach((prediction) => expect(Number.isFinite(prediction)).toBe(true));
    },
  );
});
