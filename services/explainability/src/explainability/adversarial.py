from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np


PredictionFunction = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class PerturbationConfig:
    

    epsilon: float = 0.01
    random_state: int = 42
    norm: str = "linf"

    def __post_init__(self) -> None:
        if self.epsilon <= 0:
            raise ValueError("epsilon must be greater than zero.")

        if self.norm not in {"linf", "l2"}:
            raise ValueError(
                "norm must be either 'linf' or 'l2'."
            )


@dataclass(frozen=True)
class RobustnessResult:
    """
    Result of a controlled adversarial robustness experiment.

    Attributes
    ----------
    sample_count:
        Number of observations evaluated.

    changed_predictions:
        Number of observations whose predicted class changed.

    flip_rate:
        Fraction of observations whose predicted class changed.

    stability_rate:
        Fraction of observations whose predicted class remained stable.

    mean_probability_change:
        Mean absolute change in the probability of the monitored class.

    max_probability_change:
        Maximum observed probability change.

    epsilon:
        Perturbation budget used in the experiment.

    norm:
        Perturbation norm used by the experiment.
    """

    sample_count: int
    changed_predictions: int
    flip_rate: float
    stability_rate: float
    mean_probability_change: float
    max_probability_change: float
    epsilon: float
    norm: str

    def __post_init__(self) -> None:
        if self.sample_count <= 0:
            raise ValueError(
                "sample_count must be greater than zero."
            )

        if self.changed_predictions < 0:
            raise ValueError(
                "changed_predictions cannot be negative."
            )

        if self.changed_predictions > self.sample_count:
            raise ValueError(
                "changed_predictions cannot exceed sample_count."
            )

        if not 0.0 <= self.flip_rate <= 1.0:
            raise ValueError(
                "flip_rate must be between 0 and 1."
            )

        if not 0.0 <= self.stability_rate <= 1.0:
            raise ValueError(
                "stability_rate must be between 0 and 1."
            )

        if self.mean_probability_change < 0:
            raise ValueError(
                "mean_probability_change cannot be negative."
            )

        if self.max_probability_change < 0:
            raise ValueError(
                "max_probability_change cannot be negative."
            )


class AdversarialPerturbationGenerator:
    """
    Generate bounded perturbations for robustness experiments.

    This class deliberately operates on feature matrices only. It has no
    knowledge of markets, orders, brokers, or external execution systems.

    The generated perturbations are intended for offline model evaluation.
    """

    def __init__(
        self,
        config: PerturbationConfig | None = None,
    ) -> None:
        self._config = config or PerturbationConfig()

    @property
    def config(self) -> PerturbationConfig:
        """Return the perturbation configuration."""
        return self._config

    def generate(
        self,
        instances: np.ndarray,
    ) -> np.ndarray:
        """
        Generate perturbed observations.

        Parameters
        ----------
        instances:
            Two-dimensional feature matrix.

        Returns
        -------
        numpy.ndarray
            Perturbed feature matrix with the same shape as ``instances``.
        """
        values = self._validate_matrix(
            instances,
            name="instances",
        )

        rng = np.random.default_rng(
            self._config.random_state,
        )

        noise = rng.normal(
            loc=0.0,
            scale=1.0,
            size=values.shape,
        )

        if self._config.norm == "linf":
            perturbation = np.sign(noise) * self._config.epsilon

        else:
            norms = np.linalg.norm(
                noise,
                ord=2,
                axis=1,
                keepdims=True,
            )

            norms = np.maximum(norms, np.finfo(float).eps)

            perturbation = (
                noise
                / norms
                * self._config.epsilon
            )

        return values + perturbation

    @staticmethod
    def _validate_matrix(
        matrix: np.ndarray,
        name: str,
    ) -> np.ndarray:
        """Validate a finite two-dimensional feature matrix."""
        values = np.asarray(
            matrix,
            dtype=float,
        )

        if values.ndim != 2:
            raise ValueError(
                f"{name} must be a two-dimensional matrix."
            )

        if values.shape[0] == 0:
            raise ValueError(
                f"{name} cannot be empty."
            )

        if values.shape[1] == 0:
            raise ValueError(
                f"{name} must contain at least one feature."
            )

        if not np.isfinite(values).all():
            raise ValueError(
                f"{name} contains NaN or infinite values."
            )

        return values


class AdversarialRobustnessEvaluator:
    """
    Offline robustness evaluator for trained classification models.

    The evaluator requires a prediction function supplied by the caller.
    This keeps the robustness framework independent from scikit-learn,
    PyTorch, or any specific Phase 1-4 implementation.

    The prediction function must return class probabilities with shape:

        (n_samples, n_classes)

    The evaluator compares clean and perturbed predictions without
    modifying the underlying model.
    """

    def __init__(
        self,
        predict_proba: PredictionFunction,
        output_index: int = 1,
        perturbation_generator: (
            AdversarialPerturbationGenerator | None
        ) = None,
    ) -> None:
        if not callable(predict_proba):
            raise TypeError(
                "predict_proba must be callable."
            )

        if output_index < 0:
            raise ValueError(
                "output_index must be non-negative."
            )

        self._predict_proba = predict_proba
        self._output_index = output_index
        self._generator = (
            perturbation_generator
            or AdversarialPerturbationGenerator()
        )

    @property
    def output_index(self) -> int:
        """Return the monitored class/output index."""
        return self._output_index

    @property
    def perturbation_generator(
        self,
    ) -> AdversarialPerturbationGenerator:
        """Return the configured perturbation generator."""
        return self._generator

    def evaluate(
        self,
        instances: np.ndarray,
    ) -> RobustnessResult:
        """
        Evaluate prediction stability under controlled perturbations.
        """
        clean = self._validate_matrix(
            instances,
            name="instances",
        )

        perturbed = self._generator.generate(clean)

        clean_probabilities = self._predict(
            clean,
        )

        perturbed_probabilities = self._predict(
            perturbed,
        )

        clean_class = np.argmax(
            clean_probabilities,
            axis=1,
        )

        perturbed_class = np.argmax(
            perturbed_probabilities,
            axis=1,
        )

        changed = clean_class != perturbed_class

        probability_change = np.abs(
            perturbed_probabilities[:, self._output_index]
            - clean_probabilities[:, self._output_index]
        )

        changed_predictions = int(
            np.sum(changed)
        )

        sample_count = int(
            clean.shape[0]
        )

        flip_rate = (
            changed_predictions
            / sample_count
        )

        return RobustnessResult(
            sample_count=sample_count,
            changed_predictions=changed_predictions,
            flip_rate=float(flip_rate),
            stability_rate=float(1.0 - flip_rate),
            mean_probability_change=float(
                np.mean(probability_change)
            ),
            max_probability_change=float(
                np.max(probability_change)
            ),
            epsilon=self._generator.config.epsilon,
            norm=self._generator.config.norm,
        )

    def evaluate_budgets(
        self,
        instances: np.ndarray,
        epsilons: Sequence[float],
    ) -> list[RobustnessResult]:
        """
        Evaluate robustness across multiple perturbation budgets.

        A fresh perturbation generator is created for every epsilon while
        retaining the configured random seed and perturbation norm.
        """
        if not epsilons:
            raise ValueError(
                "epsilons cannot be empty."
            )

        results: list[RobustnessResult] = []

        for epsilon in epsilons:
            if epsilon <= 0:
                raise ValueError(
                    "Every epsilon must be greater than zero."
                )

            generator = AdversarialPerturbationGenerator(
                PerturbationConfig(
                    epsilon=epsilon,
                    random_state=(
                        self._generator.config.random_state
                    ),
                    norm=self._generator.config.norm,
                )
            )

            evaluator = AdversarialRobustnessEvaluator(
                predict_proba=self._predict_proba,
                output_index=self._output_index,
                perturbation_generator=generator,
            )

            results.append(
                evaluator.evaluate(instances)
            )

        return results

    def _predict(
        self,
        instances: np.ndarray,
    ) -> np.ndarray:
        """Validate prediction-function output."""
        try:
            probabilities = np.asarray(
                self._predict_proba(instances),
                dtype=float,
            )
        except Exception as exc:
            raise RuntimeError(
                "The prediction function failed during robustness "
                "evaluation."
            ) from exc

        if probabilities.ndim != 2:
            raise ValueError(
                "predict_proba must return a two-dimensional matrix."
            )

        if probabilities.shape[0] != instances.shape[0]:
            raise ValueError(
                "Prediction row count does not match input row count."
            )

        if probabilities.shape[1] <= self._output_index:
            raise ValueError(
                "output_index exceeds the model output count."
            )

        if not np.isfinite(probabilities).all():
            raise ValueError(
                "Prediction output contains NaN or infinite values."
            )

        if np.any(probabilities < 0.0) or np.any(probabilities > 1.0):
            raise ValueError(
                "Prediction probabilities must be between 0 and 1."
            )

        return probabilities

    @staticmethod
    def _validate_matrix(
        matrix: np.ndarray,
        name: str,
    ) -> np.ndarray:
        """Validate an input feature matrix."""
        values = np.asarray(
            matrix,
            dtype=float,
        )

        if values.ndim != 2:
            raise ValueError(
                f"{name} must be a two-dimensional matrix."
            )

        if values.shape[0] == 0:
            raise ValueError(
                f"{name} cannot be empty."
            )

        if values.shape[1] == 0:
            raise ValueError(
                f"{name} must contain at least one feature."
            )

        if not np.isfinite(values).all():
            raise ValueError(
                f"{name} contains NaN or infinite values."
            )

        return values