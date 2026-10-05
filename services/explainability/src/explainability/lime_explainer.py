from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd
from lime.lime_tabular import LimeTabularExplainer

from .interfaces import LocalExplainer, validate_feature_names
from .schemas import FeatureContribution, ShapExplanation


class LimeExplainer:
   
    def __init__(
        self,
        model: Any,
        background: np.ndarray | pd.DataFrame,
        feature_names: Sequence[str],
        class_names: Sequence[str],
        output_index: int = 1,
        random_state: int = 42,
        num_samples: int = 5000,
    ) -> None:
        if model is None:
            raise ValueError("model cannot be None.")

        self._feature_names = validate_feature_names(feature_names)

        if not class_names:
            raise ValueError("class_names cannot be empty.")

        self._class_names = tuple(class_names)

        if any(
            not isinstance(name, str) or not name.strip()
            for name in self._class_names
        ):
            raise ValueError(
                "Every class name must be a non-empty string."
            )

        if output_index < 0:
            raise ValueError("output_index must be non-negative.")

        if output_index >= len(self._class_names):
            raise ValueError(
                "output_index must reference an existing class."
            )

        if num_samples <= 0:
            raise ValueError(
                "num_samples must be greater than zero."
            )

        self._model = model
        self._output_index = output_index
        self._random_state = random_state
        self._num_samples = num_samples

        self._background = self._validate_matrix(
            background,
            name="background",
        )

        if self._background.shape[1] != len(self._feature_names):
            raise ValueError(
                "Background feature count does not match feature_names."
            )

        try:
            self._explainer = LimeTabularExplainer(
                training_data=self._background,
                feature_names=list(self._feature_names),
                class_names=list(self._class_names),
                mode="classification",
                random_state=self._random_state,
            )
        except Exception as exc:
            raise RuntimeError(
                "Unable to initialize LimeTabularExplainer."
            ) from exc

        self._validate_model_interface()

    @property
    def feature_names(self) -> tuple[str, ...]:
        """Return the ordered model feature names."""
        return self._feature_names

    @property
    def output_index(self) -> int | None:
        """Return the configured class/output index."""
        return self._output_index

    @property
    def class_names(self) -> tuple[str, ...]:
        """Return the configured model class names."""
        return self._class_names

    def explain(
        self,
        instance: np.ndarray | pd.DataFrame | Sequence[float],
        metadata: dict[str, Any] | None = None,
    ) -> ShapExplanation:
       
        matrix = self._validate_instance(instance)
        row = matrix[0]

        try:
            explanation = self._explainer.explain_instance(
                data_row=row,
                predict_fn=self._predict_proba,
                labels=(self._output_index,),
                num_features=len(self._feature_names),
                num_samples=self._num_samples,
            )
        except Exception as exc:
            raise RuntimeError(
                "Unable to generate the LIME explanation."
            ) from exc

        local_map = explanation.as_map()

        if self._output_index not in local_map:
            raise RuntimeError(
                "LIME did not return an explanation for the requested "
                "output index."
            )

        weights_by_index = dict(
            local_map[self._output_index]
        )

        prediction = self._extract_prediction(row)

        base_value = self._extract_intercept(
            explanation,
            self._output_index,
        )

        contributions = tuple(
            FeatureContribution(
                feature_name=self._feature_names[index],
                feature_value=float(row[index]),
                shap_value=float(weights_by_index.get(index, 0.0)),
            )
            for index in range(len(self._feature_names))
        )

        return ShapExplanation(
            base_value=base_value,
            prediction=prediction,
            contributions=contributions,
            feature_names=self._feature_names,
            output_index=self._output_index,
            metadata={
                **(metadata or {}),
                "explanation_method": "lime",
                "lime_num_samples": self._num_samples,
                "lime_random_state": self._random_state,
            },
        )

    def _predict_proba(
        self,
        instances: np.ndarray,
    ) -> np.ndarray:
        
        values = np.asarray(instances, dtype=float)

        if values.ndim == 1:
            values = values.reshape(1, -1)

        if values.ndim != 2:
            raise ValueError(
                "LIME prediction input must be a two-dimensional matrix."
            )

        if values.shape[1] != len(self._feature_names):
            raise ValueError(
                "LIME prediction input feature count does not match "
                "feature_names."
            )

        if not np.isfinite(values).all():
            raise ValueError(
                "LIME prediction input contains NaN or infinite values."
            )

        try:
            probabilities = np.asarray(
                self._model.predict_proba(values),
                dtype=float,
            )
        except Exception as exc:
            raise RuntimeError(
                "The supplied model failed during predict_proba()."
            ) from exc

        if probabilities.ndim != 2:
            raise ValueError(
                "model.predict_proba() must return a two-dimensional "
                "probability matrix."
            )

        if probabilities.shape[0] != values.shape[0]:
            raise ValueError(
                "Model probability output row count does not match input."
            )

        if probabilities.shape[1] != len(self._class_names):
            raise ValueError(
                "Model probability output class count does not match "
                "class_names."
            )

        if not np.isfinite(probabilities).all():
            raise ValueError(
                "Model probability output contains NaN or infinite values."
            )

        if np.any(probabilities < 0.0) or np.any(probabilities > 1.0):
            raise ValueError(
                "Model probabilities must be between 0 and 1."
            )

        return probabilities

    def _extract_prediction(
        self,
        instance: np.ndarray,
    ) -> float:
        """Return the probability of the explained class."""
        probabilities = self._predict_proba(
            instance.reshape(1, -1)
        )

        return float(
            probabilities[0, self._output_index]
        )

    @staticmethod
    def _extract_intercept(
        explanation: Any,
        output_index: int,
    ) -> float:
        
        try:
            intercept = explanation.intercept[output_index]
        except (AttributeError, KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(
                "Unable to extract the LIME surrogate intercept."
            ) from exc

        return float(intercept)

    def _validate_instance(
        self,
        instance: np.ndarray | pd.DataFrame | Sequence[float],
    ) -> np.ndarray:
        """Validate and normalize one model observation."""
        if isinstance(instance, pd.DataFrame):
            if instance.shape[0] != 1:
                raise ValueError(
                    "instance DataFrame must contain exactly one row."
                )

            if tuple(instance.columns) != self._feature_names:
                raise ValueError(
                    "instance DataFrame columns must exactly match "
                    "feature_names."
                )

            matrix = instance.to_numpy(dtype=float)

        else:
            matrix = np.asarray(instance, dtype=float)

            if matrix.ndim == 1:
                matrix = matrix.reshape(1, -1)

        if matrix.ndim != 2 or matrix.shape[0] != 1:
            raise ValueError(
                "instance must represent exactly one observation."
            )

        if matrix.shape[1] != len(self._feature_names):
            raise ValueError(
                "instance feature count does not match feature_names."
            )

        if not np.isfinite(matrix).all():
            raise ValueError(
                "instance contains NaN or infinite values."
            )

        return matrix

    @staticmethod
    def _validate_matrix(
        matrix: np.ndarray | pd.DataFrame,
        name: str,
    ) -> np.ndarray:
        """Validate and normalize a two-dimensional feature matrix."""
        if isinstance(matrix, pd.DataFrame):
            values = matrix.to_numpy(dtype=float)
        else:
            values = np.asarray(matrix, dtype=float)

        if values.ndim != 2:
            raise ValueError(
                f"{name} must be a two-dimensional feature matrix."
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

    def _validate_model_interface(self) -> None:
        """Verify that the model exposes the interface required by LIME."""
        predict_proba = getattr(
            self._model,
            "predict_proba",
            None,
        )

        if predict_proba is None or not callable(predict_proba):
            raise TypeError(
                "LimeExplainer requires a classification model exposing "
                "a callable predict_proba() method."
            )


# Explicit structural typing assertion for static type checkers.
_LimeExplainerContract: LocalExplainer = LimeExplainer(
    model=object(),
    background=np.zeros((1, 1)),
    feature_names=("feature",),
    class_names=("class_0", "class_1"),
)