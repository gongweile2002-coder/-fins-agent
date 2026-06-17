# ruff: noqa
"""Print the Week 3 forecast target and model configuration."""

from __future__ import annotations

from fins2026.week3.code import AUSTRALIA_FORECAST_SPECS, MODEL_LABELS, ONE_STEP_ONLY_MODELS


def describe_forecast_data() -> str:
    """Return the Week 3 forecast target and model inventory."""

    lines = ["Week 3 forecast configuration", ""]
    lines.append("Australia targets:")
    for spec in AUSTRALIA_FORECAST_SPECS.values():
        lines.append(
            f"- {spec.label}: {spec.frequency}, target `{spec.target}`, units `{spec.units}`"
        )
    lines.append("")
    lines.append("Approved models:")
    for model, label in MODEL_LABELS.items():
        scope = (
            "one-step only"
            if model in ONE_STEP_ONLY_MODELS
            else "forward path + one-step backtest"
        )
        lines.append(f"- {label}: `{model}` ({scope})")
    return "\n".join(lines)


def main() -> int:
    print(describe_forecast_data())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
