# Negative results and limitations

This file is generated from results/summary.json, results/agent_eval.json and the fine-tune compare artifact. It records where the methods did not help.

## Fine-tuning vs zero-shot
- WQL: fine-tuned minus zero-shot = -0.0014 on 20 series.
- CRPS: fine-tuned minus zero-shot = +0.8784 on 20 series.
- MASE: fine-tuned minus zero-shot = -0.0035 on 20 series.
- Paired 95% CI for the WQL delta is [-0.0506, +0.0533], which includes zero: there is no statistically detectable WQL gain from fine-tuning.

## Ensembles vs the best single model
- synthetic_supply: ensemble_best_on_backtest vs autoets is identical to autoets by construction (the selection rule picks that same model).
- synthetic_supply: ensemble_inverse_error vs autoets WQL delta +0.0436 [+0.0058, +0.0830], not significantly better than autoets.
- synthetic_supply: ensemble_vincentization vs autoets WQL delta +0.0848 [+0.0395, +0.1315], not significantly better than autoets.
- m4_daily: ensemble_best_on_backtest vs chronos is identical to chronos by construction (the selection rule picks that same model).
- m4_daily: ensemble_inverse_error vs chronos WQL delta +0.0613 [-0.0047, +0.1423], not significantly better than chronos.
- m4_daily: ensemble_vincentization vs chronos WQL delta +0.0741 [+0.0006, +0.1654], not significantly better than chronos.
- No ensemble/reconciliation arm beat the best single model with a 95% CI excluding zero.

## Covariates
- synthetic_supply: promo-aware seasonal naive minus plain seasonal naive WQL -0.3776 [-0.4598, -0.3003]; helped.
- Chronos T5 does not consume exogenous covariates in this implementation, so there is no foundation-model covariate arm.

## Multivariate / covariate-capable foundation models
- Status: unavailable. TimesFM and Moirai (the covariate/multivariate-capable adapters) are not installed; see unavailable_models

## Agent routing cost
- Mean selection regret vs oracle: +0.0911 WQL; mean cost per task 0.003060 USD; mean latency 6.22s; recovery rate 1.00.
- Multi-model routing added cost and latency without beating the oracle model on WQL.

## Unavailable models
- chronos-small: disabled in configs/models.yaml (not in the benchmark model list)
- timesfm: timesfm not importable (ModuleNotFoundError: No module named 'timesfm'); install the [fm] extra (not in the benchmark model list)
- moirai: uni2ts not importable (ModuleNotFoundError: No module named 'uni2ts'); install the [fm] extra (not in the benchmark model list)

