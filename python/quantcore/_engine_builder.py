"""
python/quantcore/_engine_builder.py
=====================================
Internal helper for constructing a fully configured BacktestEngine
from a BacktestConfig. Shared by __init__.py and walk_forward.py.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List

if TYPE_CHECKING:
    from quantcore.walk_forward import BacktestConfig


def parse_fill_mode(mode):
    """Accept a FillMode or 'same_bar_close' / 'next_bar_open'."""
    import quantcore as qc
    if isinstance(mode, qc.FillMode):
        return mode
    table = {'same_bar_close': qc.FillMode.SAME_BAR_CLOSE,
             'next_bar_open':  qc.FillMode.NEXT_BAR_OPEN}
    try:
        return table[str(mode).lower()]
    except KeyError:
        raise ValueError(f"Unknown fill_mode '{mode}'. Valid values: {list(table)}") from None


def build_engine(data: Dict[str, list], capital: float, config: "BacktestConfig"):
    """
    Construct and configure a BacktestEngine from a BacktestConfig.

    Returns the engine before run() is called.
    """
    import quantcore as qc

    exec_config              = qc.ExecutionConfig()
    exec_config.maker_fee    = config.maker_fee
    exec_config.taker_fee    = config.taker_fee
    exec_config.latency_ns   = config.latency_ns
    exec_config.slippage_pct = config.slippage_pct

    engine = qc.BacktestEngine(capital, exec_config)

    for symbol, bars in data.items():
        engine.add_data(symbol, bars)

    sizer_classes = {
        'FixedPercentage':     qc.FixedPercentage,
        'RiskBased':           qc.RiskBased,
        'KellyCriterion':      qc.KellyCriterion,
        'EqualWeight':         qc.EqualWeight,
        'VolatilityTargeting': qc.VolatilityTargeting,
        'FixedShares':         qc.FixedShares,
    }
    cls = sizer_classes.get(config.sizer_type)
    if cls is None:
        raise ValueError(
            f"Unknown sizer_type '{config.sizer_type}'. "
            f"Valid values: {list(sizer_classes)}"
        )
    engine.set_position_sizer(cls(*config.sizer_args))
    engine.set_bars_per_year(config.bars_per_year)
    engine.set_fill_mode(parse_fill_mode(config.fill_mode))

    limits = qc.RiskLimits()
    limits.max_position_pct = config.max_position_pct
    limits.max_leverage     = config.max_leverage
    limits.max_loss_pct     = config.max_loss_pct
    limits.max_order_value  = config.max_order_value
    limits.enabled          = config.risk_enabled
    engine.set_risk_limits(limits)

    return engine