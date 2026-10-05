from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd
import shap

from .schemas import FeatureContribution, ShapExplanation

class ShapExplainer:
   

    def __init__(
        self,
        model: Any,
        background: np.ndarray | pd.DataFrame,
        feature_names: Sequence[str],
        output_index: int | None = None,
    ) -> None:
        if model is None:
            raise ValueError("model cannot be None.")

        if not feature_names:
            raise ValueError("feature_names cannot be empty.")

        self._feature_names = tuple(feature_names)

        if len(set(self._feature_names)) != len(self._feature_names):
            raise ValueError("feature_names must be unique.")

        if any(
            not isinstance(name, str) or not name.strip()
            for name in self._feature_names
        ):
            raise ValueError(
                "Every feature name must be a non-empty string."
            )

        if output_index is not None and output_index < 0:
            raise ValueError("output_index must be non-negative.")

        self._model = model
        self._background = self._validate_matrix(
            background,
            name="background",
        )
        self._output_index = output_index

        if self._background.shape[1] != len(self._feature_names):
            raise ValueError(
                "Background feature count does not match feature_names."
            )

        try:
            self._explainer = shap.TreeExplainer(
                self._model,
                data=self._background,
            )
        except Exception as exc:
            raise TypeError(
                "The supplied model is not compatible with "
                "shap.TreeExplainer."
            ) from exc

    @property
    def feature_names(self) -> tuple[str, ...]:
        """Return the ordered feature names used by the explainer."""
        return self._feature_names

    @property
    def output_index(self) -> int | None:
        """Return the configured output/class index."""
        return self._output_index

    def explain(
        self,
        instance: np.ndarray | pd.DataFrame | Sequence[float],
        metadata: dict[str, Any] | None = None,
    ) -> ShapExplanation:
        
        matrix = self._validate_instance(instance)

        shap_values = self._explainer.shap_values(matrix)
        base_value = self._extract_base_value()

        selected_values = self._select_output(
            shap_values,
            self._output_index,
        )

        selected_values = np.asarray(
            selected_values,
            dtype=float,
        ).reshape(-1)

        if selected_values.shape[0] != len(self._feature_names):
            raise ValueError(
                "SHAP output feature count does not match feature_names."
            )

        prediction = self._extract_prediction(
            matrix,
            output_index=self._output_index,
        )

        contributions = tuple(
            FeatureContribution(
                feature_name=feature_name,
                feature_value=float(value),
                shap_value=float(shap_value),
            )
            for feature_name, value, shap_value in zip(
                self._feature_names,
                matrix[0],
                selected_values,
            )
        )

        return ShapExplanation(
            base_value=base_value,
            prediction=prediction,
            contributions=contributions,
            feature_names=self._feature_names,
            output_index=self._output_index,
            metadata=metadata,
        )

    def explain_dataframe(
        self,
        instances: pd.DataFrame,
    ) -> list[ShapExplanation]:
      
        if instances.empty:
            raise ValueError("instances cannot be empty.")

        if tuple(instances.columns) != self._feature_names:
            raise ValueError(
                "DataFrame columns must exactly match feature_names "
                "in both names and order."
            )

        matrix = self._validate_matrix(
            instances,
            name="instances",
        )

        return [
            self.explain(matrix[index])
            for index in range(matrix.shape[0])
        ]

    def global_importance(
        self,
        instances: np.ndarray | pd.DataFrame,
    ) -> pd.DataFrame:
    
        matrix = self._validate_matrix(
            instances,
            name="instances",
        )

        shap_values = self._explainer.shap_values(matrix)

        selected_values = self._select_output(
            shap_values,
            self._output_index,
        )

        values = np.asarray(
            selected_values,
            dtype=float,
        )

        if values.ndim != 2:
            raise ValueError(
                "Expected SHAP values for multiple observations to have "
                "two dimensions."
            )

        if values.shape[1] != len(self._feature_names):
            raise ValueError(
                "SHAP output feature count does not match feature_names."
            )

        importance = np.mean(np.abs(values), axis=0)

        result = pd.DataFrame(
            {
                "feature_name": self._feature_names,
                "mean_abs_shap": importance,
            }
        )

        return result.sort_values(
            "mean_abs_shap",
            ascending=False,
            ignore_index=True,
        )

    def _extract_prediction(
        self,
        matrix: np.ndarray,
        output_index: int | None,
    ) -> float:
        try:
            if hasattr(self._model, "predict_proba"):
                prediction = self._model.predict_proba(matrix)

                if prediction.ndim == 1:
                    return float(prediction[0])

                if output_index is None:
                    if prediction.shape[1] == 1:
                        return float(prediction[0, 0])

                    raise ValueError(
                        "A multi-output classifier requires output_index "
                        "to identify the explained class."
                    )

                if output_index >= prediction.shape[1]:
                    raise ValueError(
                        "output_index exceeds the model's output count."
                    )

                return float(prediction[0, output_index])

            prediction = np.asarray(
                self._model.predict(matrix),
                dtype=float,
            ).reshape(-1)

            if prediction.size != 1:
                raise ValueError(
                    "Model prediction is not scalar. "
                    "Specify a compatible model/output configuration."
                )

            return float(prediction[0])

        except ValueError:
            raise
        except Exception as exc:
            raise RuntimeError(
                "Unable to obtain the model prediction for the "
                "SHAP explanation."
            ) from exc

    def _extract_base_value(self) -> float:
        """Extract a scalar expected/base value from TreeExplainer."""
        expected_value = np.asarray(
            self._explainer.expected_value,
            dtype=float,
        ).reshape(-1)

        if self._output_index is not None:
            if self._output_index >= expected_value.size:
                raise ValueError(
                    "output_index exceeds the explainer's base-value outputs."
                )

            return float(expected_value[self._output_index])

        if expected_value.size != 1:
            raise ValueError(
                "A multi-output explanation requires output_index."
            )

        return float(expected_value[0])

    @staticmethod
    def _select_output(
        shap_values: Any,
        output_index: int | None,
    ) -> np.ndarray:
        
        if isinstance(shap_values, list):
            if output_index is None:
                if len(shap_values) != 1:
                    raise ValueError(
                        "A multi-output SHAP result requires output_index."
                    )

                return np.asarray(shap_values[0])

            if output_index >= len(shap_values):
                raise ValueError(
                    "output_index exceeds the SHAP output count."
                )

            return np.asarray(shap_values[output_index])

        values = np.asarray(shap_values)

        if values.ndim == 3:
            if output_index is None:
                if values.shape[-1] != 1:
                    raise ValueError(
                        "A multi-output SHAP result requires output_index."
                    )

                return values[:, :, 0]

            if output_index >= values.shape[-1]:
                raise ValueError(
                    "output_index exceeds the SHAP output count."
                )

            return values[:, :, output_index]

        if values.ndim == 2:
            return values

        if values.ndim == 1:
            return values.reshape(1, -1)

        raise ValueError(
            "Unsupported SHAP output shape."
        )

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
