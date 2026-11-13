# Contributing

## Adding a forecaster adapter (about 20 lines)

Implement the `Forecaster` protocol in a new module under `forecastmcp/forecasters/`,
then register it in `configs/models.yaml`.

```python
import time
from forecastmcp.forecasters.base import (
    BaseForecaster, ForecastRequest, ForecastResult, check_import,
)

class MyForecaster(BaseForecaster):
    supports_covariates = False

    def available(self) -> tuple[bool, str]:
        ok, reason = check_import("my_package")          # never raise here
        return (False, reason) if not ok else (True, "ok")

    def predict(self, req: ForecastRequest) -> ForecastResult:
        started = time.perf_counter()
        point, quantiles = my_package.forecast(req.context, req.horizon, req.quantile_levels)
        return self.finalize(                            # clamps + sorts quantiles
            req, point, quantiles, started,
            device="cpu", used_covariates=False,
            revision=self.revision_for(),
        )
```

Register it:

```yaml
models:
  my-model:
    adapter: my_adapter          # map this to your class in registry._instantiate
    checkpoint: org/checkpoint
    revision: main
    context_length: 512
    device: cpu
    cost_per_call_usd: 0.0005
    supports_covariates: false
    training_domain: "describe the pretraining corpus"
```

Then add a test in `tests/test_forecasters.py` for quantile shape/monotonicity and
confirm `forecastmcp models list` reports availability without downloading weights.

## Ground rules

- Every metric in `results/` and `README.md` must come from a real pipeline run.
- Optional dependencies must fail soft: `available()` returns the verbatim reason.
- New tools need a Pydantic input/output schema, a role in `mcp/permissions.py`, and
  a budget estimate in `configs/costs.yaml`.
- Run `ruff check forecastmcp tests` and `pytest` before opening a PR.
