from __future__ import annotations

from dataclasses import dataclass

from app.environment.market import SimulatedMarketEnvironment
from app.environment.portfolio import SimulatedPortfolio
from app.environment.reward import RiskAdjustedReward
from app.environment.state import Action, MarketState


@dataclass(frozen=True)
class EnvironmentTransition:
    

    state: MarketState
    action: int
    reward: float
    next_state: MarketState
    done: bool


class ReinforcementLearningEnvironment:
    

    def __init__(
        self,
        market: SimulatedMarketEnvironment,
        portfolio: SimulatedPortfolio,
        reward_function: RiskAdjustedReward,
    ) -> None:
        self._market = market
        self._portfolio = portfolio
        self._reward_function = reward_function

        self._previous_return = 0.0

    @property
    def state(self) -> MarketState:
        """
        Return the current observable RL state.
        """

        return MarketState(
            return_value=self._previous_return,
            volatility=self._estimate_volatility(),
            position=self._portfolio.position,
        )

    def reset(self) -> MarketState:
        """
        Reset both the market and portfolio simulation.
        """

        self._market.reset()
        self._portfolio.reset()

        self._previous_return = 0.0

        return self.state

    def step(
        self,
        action: int,
    ) -> EnvironmentTransition:
        """
        Apply one RL action and advance the simulation.

        Sequence:

            1. Observe current state.
            2. Validate and apply action.
            3. Advance simulated market.
            4. Calculate reward.
            5. Construct next state.
            6. Return complete transition.
        """

        Action.validate(action)

        current_state = self.state

        portfolio_transition = (
            self._portfolio.apply_action(action)
        )

        market_step = self._market.step()

        reward_result = self._reward_function.calculate(
            position=(
                portfolio_transition.new_position
            ),
            market_return=market_step.return_value,
        )

        self._previous_return = (
            market_step.return_value
        )

        next_state = self.state

        return EnvironmentTransition(
            state=current_state,
            action=action,
            reward=reward_result.total_reward,
            next_state=next_state,
            done=market_step.done,
        )

    def _estimate_volatility(self) -> float:
        

        return abs(self._previous_return)