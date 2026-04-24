function randInt(min, max) {
  return Math.floor(Math.random() * (max - min + 1)) + min;
}

export function addNoise(value, noiseLevel) {
  return value * (1 + (Math.random() * 2 - 1) * noiseLevel);
}

export function buildDataWithRandExperts(rows, randExperts) {
  return rows.map((row, index) => {
    const next = { ...row };
    randExperts.forEach((expert) => {
      next[expert.id] = expert.values[index];
    });
    return next;
  });
}

export function buildConfiguredExperts(rows, configuredExperts) {
  return configuredExperts.map((configuredExpert) => ({
    ...configuredExpert,
    values: rows.map((row, rowIndex) => {
      const phase = configuredExpert.phases.find((candidate) => rowIndex >= candidate.start && rowIndex < candidate.end)
        || configuredExpert.phases[configuredExpert.phases.length - 1];
      const baseValue = row[phase.expert] || 0;
      return phase.noise > 0 ? addNoise(baseValue, phase.noise) : baseValue;
    }),
  }));
}

export function generateRandExperts(rows, nExperts, phaseRange, noiseLevel, syntheticIds) {
  const rowCount = rows.length;
  const experts = [];
  for (let expertIndex = 0; expertIndex < nExperts; expertIndex += 1) {
    const phaseCount = randInt(phaseRange[0], phaseRange[1]);
    const breakpoints = [0];
    const randomPoints = [];
    for (let phaseIndex = 0; phaseIndex < phaseCount - 1; phaseIndex += 1) {
      randomPoints.push(randInt(1, rowCount - 1));
    }
    randomPoints.sort((left, right) => left - right);
    breakpoints.push(...new Set(randomPoints), rowCount);

    const phases = [];
    for (let phaseIndex = 0; phaseIndex < breakpoints.length - 1; phaseIndex += 1) {
      const expertId = syntheticIds[randInt(0, syntheticIds.length - 1)];
      phases.push({ start: breakpoints[phaseIndex], end: breakpoints[phaseIndex + 1], expert: expertId });
    }

    const columnId = `rand_expert_${expertIndex + 1}`;
    const values = rows.map((row, rowIndex) => {
      const phase = phases.find((candidate) => rowIndex >= candidate.start && rowIndex < candidate.end)
        || phases[phases.length - 1];
      const baseValue = row[phase.expert] || 0;
      return noiseLevel > 0 ? addNoise(baseValue, noiseLevel) : baseValue;
    });
    experts.push({ id: columnId, label: `R-Expert ${expertIndex + 1}`, phases, values, noiseLevel });
  }
  return experts;
}
