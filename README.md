# Air Liquide R&D - Mixture of Experts Day-Ahead Wind Power Forecasting for Renewable Hydrogen Production

**Hi there! 👋**

This repository presents a forecasting and aggregation framework designed for day-ahead wind power prediction in the Air Liquide context of renewable hydrogen production.

Accurate wind power forecasts are critical to ensure a continuous and cost-efficient electricity supply to electrolyzers, which operate under strong industrial and operational constraints. In this setting, forecast errors directly impact the balance between renewable generation and fossil-based compensation under meteorological uncertainty.

## Problem Context

**Objective:** predict wind power production at a specific industrial site (Belgium) on a day-ahead, hourly basis.

**Operational constraint:** full 24-hour vector forecasting with no intermediate observations available during the delivery period.

**Industrial setting:** forecasts are used to support energy balancing decisions for electrolyzers requiring stable power input.

**Data characteristics:**
- historical wind power generation,
- heterogeneous meteorological forecasts from multiple providers,
- strong temporal dependency and non-stationarity.

## Modeling Approach

The project explores a range of forecasting and aggregation strategies tailored to strict temporal and operational constraints:

**Base forecasting models:**
- regularized linear models,
- tree-based models,
- ARMA-type time series models.

**Evaluation protocol:**
- strict train / validation / test separation,
- expanding-window backtesting to ensure full temporal consistency and realistic generalization.

## Mixture-of-Experts

A Mixture-of-Experts (MoE) approach is implemented to dynamically aggregate multiple forecasting models:
- online aggregation using continuous-time learning formulations,
- constant-weight rebalancing mechanisms,
- algorithms including BOA, ML-Prod, and FTRL,
- full 24-hour vector prediction handled either through direct vector forecasting or probabilistic directional modeling.

The framework is built using and extending the OPERA Python library, adapted to the day-ahead forecasting setting.

## Ensemble Extensions to MoE

Beyond standard online mixtures, the project investigates additional ensemble strategies under realistic industrial constraints, including:
- constrained linear stacking with rolling windows,
- state-space mixtures with latent dynamic weights,
- gating networks for conditional expert selection,
- Bayesian Model Averaging,
- quantile-based ensembles optimized via pinball loss.
