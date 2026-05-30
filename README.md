# Air Liquide R&D - Mixture of Experts Day-Ahead Wind Power Forecasting for Renewable Hydrogen Production
**Hi there! 👋**

This repository presents a forecasting and aggregation framework designed for day-ahead wind power prediction in the Air Liquide context of renewable hydrogen production.

Accurate wind power forecasts are critical to ensure a continuous and cost-efficient electricity supply to electrolyzers, which operate under strong industrial and operational constraints. In this setting, forecast errors directly impact the balance between renewable generation and fossil-based compensation under meteorological uncertainty.

**Engine Open Source access : https://moe-runner.netlify.app**

## Problem Context

Objective: predict wind power production at a specific industrial site (Belgium) on a day-ahead, hourly basis.

Operational constraint: full 24-hour vector forecasting with no intermediate observations available during the delivery period.

Industrial setting: forecasts are used to support energy balancing decisions for electrolyzers requiring stable power input.

Data characteristics:

- historical wind power generation,
- heterogeneous meteorological forecasts from multiple providers,
- strong temporal dependency and non-stationarity.

![pb](/figures/problem.jpg)

## Modeling Approach

The project explores a range of forecasting and aggregation strategies tailored to strict temporal and operational constraints:

Base forecasting models:

- regularized linear models,
- tree-based models,
- ARMA-type time series models.

Evaluation protocol:

- strict train / validation / test separation,
- expanding-window backtesting to ensure full temporal consistency and realistic generalization.

![model](/figures/modelisation.jpg)

## Mixture-of-Experts

A Mixture-of-Experts (MoE) approach is generally implemented to dynamically aggregate multiple forecasting models:

- online aggregation using continuous-time learning formulations,
- constant-weight rebalancing mechanisms,
- algorithms including BOA, ML-Prod, ML-Pol, and FTRL,
- full 24-hour vector prediction handled either through direct vector forecasting or probabilistic directional modeling.

The framework is built using and extending the OPERA Python library concepts, adapted here into an interactive React runner for the day-ahead forecasting setting.

## Current Application

This repository now contains a lightweight React application to run, compare, and visualize aggregation methods on wind forecasting CSV datasets.

![trail1](/figures/trail1.jpg)

![trail3](/figures/trail3.jpg)



The interface supports:

- CSV upload and parsing with `y_true` and expert prediction columns,
- selection of forecasting experts and aggregation algorithms,
- comparison charts for predictions, errors, cumulative loss, and expert weights,
- ranking tables using MAE, RMSE, and MAPE,
- configurable loss functions: MSE, MAE, MAPE, MSLE, and MSPE,
- gradient-based or loss-based online updates,
- synthetic/random expert generation for robustness experiments.

## Aggregation Methods

Available method families include:

- Opera-style MoE: BOA, MLpol, MLprod, FTRL,
- Hierarchical MoE: HMOE BOA, HMOE MLpol, HMOE MLprod, HMOE FTRL,
- static baselines: simple mean, median, trimmed mean,
- adaptive baselines: inverse-MSE weighting, best expert, ridge blending.

![trail1](/figures/trail2.jpg)

![t](/figures/trail4.jpg)

## Experimentation Tools

The project has also been enriched with:

- Monte Carlo simulations for repeated randomized comparisons,
- Monte Carlo grid search for parameter exploration,
- regime-gated HMOE features based on temporal, momentum, volatility, trend, and production regimes,
- reusable configuration files for algorithms, grid search, and simulation settings.

The app can run either in production mode on real input datasets or in random mode, where synthetic experts are regenerated at each run from configurable ranges, randomly distributed expert characteristics, and random phase counts/durations to support independence assumptions and statistical robustness in Monte Carlo simulations as `n` grows.


![trail6](/figures/trail5.jpg)

![trail6](/figures/trail6.jpg)

## Expected CSV Format

The app expects a CSV file with:

- `y_true`: observed wind power production,
- one column per expert prediction,
- optional temporal columns such as `decision_time`, `target_time`, and `horizon`.

Example expert columns can include benchmark models, specialist models, derived/degraded models, or feature-restricted models.

## Run Locally

Install dependencies:

```bash
npm install
```

Start the development server:

```bash
npm start
```

Build for production:

```bash
npm run build
```

Run tests:

```bash
npm test
```

---
***Alexandre Mathias DONNAT, Sr***
