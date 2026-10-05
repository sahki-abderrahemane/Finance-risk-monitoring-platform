from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RewardBreakdown:
    

    portfolio_return: float
    risk_penalty: float
    transaction_cost: float
    drawdown_penalty: float
    total_reward: float


class RiskAdjustedReward:
   
    def __init__(
        self,
        risk_penalty_coefficient: float = 0.0,
        transaction_cost_coefficient: float = 0.0,
        drawdown_penalty_coefficient: float = 0.0,
    ) -> None:
        for name, value in (
            (
                "risk_penalty_coefficient",
                risk_penalty_coefficient,
            ),
            (
                "transaction_cost_coefficient",
                transaction_cost_coefficient,
            ),
            (
                "drawdown_penalty_coefficient",
                drawdown_penalty_coefficient,
            ),
        ):
            if value < 0:
                raise ValueError(
                    f"{name} cannot be negative."
                )

        self._risk_penalty_coefficient = float(
            risk_penalty_coefficient
        )

        self._transaction_cost_coefficient = float(
            transaction_cost_coefficient
        )

        self._drawdown_penalty_coefficient = float(
            drawdown_penalty_coefficient
        )

    def calculate(
        self,
        position: float,
        market_return: float,
        risk_value: float = 0.0,
        transaction_cost: float = 0.0,
        drawdown: float = 0.0,
    ) -> RewardBreakdown:
        """
        Calculate one simulated risk-aware reward.

        Formula:

            R =
                position * market_return
                - lambda_r * risk
                - lambda_c * transaction_cost
                - lambda_d * drawdown

        drawdown is expected as a non-negative magnitude.
        """

        if risk_value < 0:
            raise ValueError(
                "risk_value cannot be negative."
            )

        if transaction_cost < 0:
            raise ValueError(
                "transaction_cost cannot be negative."
            )

        if drawdown < 0:
            raise ValueError(
                "drawdown cannot be negative."
            )

        portfolio_return = (
            position * market_return
        )

        risk_penalty = (
            self._risk_penalty_coefficient
            * risk_value
        )

        cost_penalty = (
            self._transaction_cost_coefficient
            * transaction_cost
        )

        drawdown_penalty = (
            self._drawdown_penalty_coefficient
            * drawdown
        )

        total_reward = (
            portfolio_return
            - risk_penalty
            - cost_penalty
            - drawdown_penalty
        )

        return RewardBreakdown(
            portfolio_return=portfolio_return,
            risk_penalty=risk_penalty,
            transaction_cost=cost_penalty,
            drawdown_penalty=drawdown_penalty,
            total_reward=total_reward,
        )