from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from statistics import mean, pstdev
from typing import Sequence


@dataclass(frozen=True)
class EvaluationMetrics:
    """
    Risk and performance metrics for one simulated policy.
    """

    total_return: float
    average_step_return: float
    volatility: float
    maximum_drawdown: float
    total_reward: float
    steps: int


class RiskMetricsCalculator:
    """
    Calculates evaluation metrics from a simulated policy trajectory.

    These metrics describe historical/synthetic simulation results
    only. They are not intended as real-world investment advice.
    """

    def calculate(
        self,
        step_returns: Sequence[float],
        rewards: Sequence[float],
    ) -> EvaluationMetrics:
        """
        Calculate return, volatility, drawdown, and reward metrics.
        """

        normalized_returns = tuple(
            float(value)
            for value in step_returns
        )

        normalized_rewards = tuple(
            float(value)
            for value in rewards
        )

        if not normalized_returns:
            return EvaluationMetrics(
                total_return=0.0,
                average_step_return=0.0,
                volatility=0.0,
                maximum_drawdown=0.0,
                total_reward=sum(
                    normalized_rewards
                ),
                steps=0,
            )

        equity_curve = self._build_equity_curve(
            normalized_returns
        )

        return EvaluationMetrics(
            total_return=equity_curve[-1] - 1.0,
            average_step_return=mean(
                normalized_returns
            ),
            volatility=self._volatility(
                normalized_returns
            ),
            maximum_drawdown=self._maximum_drawdown(
                equity_curve
            ),
            total_reward=sum(
                normalized_rewards
            ),
            steps=len(normalized_returns),
        )

    @staticmethod
    def _build_equity_curve(
        returns: Sequence[float],
    ) -> tuple[float, ...]:
        """
        Build a compounded simulated equity curve.

        Starting capital is normalized to 1.0.
        """

        equity = 1.0
        curve = [equity]

        for step_return in returns:
            equity *= 1.0 + step_return
            curve.append(equity)

        return tuple(curve)

    @staticmethod
    def _volatility(
        returns: Sequence[float],
    ) -> float:
        """
        Calculate population standard deviation of step returns.

        The simulation currently uses discrete timesteps, so this
        metric is intentionally not annualized.
        """

        if len(returns) < 2:
            return 0.0

        return pstdev(returns)

    @staticmethod
    def _maximum_drawdown(
        equity_curve: Sequence[float],
    ) -> float:
        """
        Calculate the maximum peak-to-trough drawdown.

        Example:

            peak = 1.20
            trough = 1.08

            drawdown = (1.08 - 1.20) / 1.20
                     = -0.10
        """

        if not equity_curve:
            return 0.0

        peak = equity_curve[0]
        maximum_drawdown = 0.0

        for equity in equity_curve:
            peak = max(peak, equity)

            if peak == 0.0:
                continue

            drawdown = (
                equity - peak
            ) / peak

            maximum_drawdown = min(
                maximum_drawdown,
                drawdown,
            )

        return maximum_drawdown