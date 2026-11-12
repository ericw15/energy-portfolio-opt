from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import List

import cvxpy as cp
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import OptimizeResult


def get_returns(tickers, start_date, end_date):
    data = yf.download(tickers=tickers, start=start_date, end=end_date, progress=False)
    return data["Close"].pct_change().dropna()


@dataclass(frozen=True)
class OptimizationDiagnostic:
    """Numerical evidence for one portfolio-construction solve."""

    solver: str
    status: str
    objective_value: float
    lower_bound: float
    upper_bound: float
    optimality_gap: float
    certified: bool
    iterations: int | None

    def as_record(self) -> dict[str, object]:
        return {
            "solver": self.solver,
            "status": self.status,
            "objective_value": self.objective_value,
            "lower_bound": self.lower_bound,
            "upper_bound": self.upper_bound,
            "optimality_gap": self.optimality_gap,
            "certified": self.certified,
            "iterations": self.iterations,
        }


class Portfolio_Strategy:
    _SOLVER = cp.CLARABEL
    _SOLVER_NAME = "CLARABEL"
    _BISSECTION_TOLERANCE = 1e-6

    def __init__(self, risk_free_rate: float):
        """``risk_free_rate`` is the rate for one return period."""
        self.r_free = risk_free_rate
        self.last_optimization_diagnostic: OptimizationDiagnostic | None = None

    @staticmethod
    def _validated_optimization_inputs(
        covariance_matrix,
        port_expected_returns,
        equity_names: List[str],
    ) -> tuple[np.ndarray, np.ndarray]:
        num_equities = len(equity_names)
        covariance = np.asarray(covariance_matrix, dtype=float)
        expected_returns = np.asarray(port_expected_returns, dtype=float).reshape(-1)
        if covariance.shape != (num_equities, num_equities):
            raise ValueError("covariance_matrix must be square with one row per asset")
        if expected_returns.shape != (num_equities,):
            raise ValueError("port_expected_returns must contain one value per asset")
        if not np.isfinite(covariance).all() or not np.isfinite(expected_returns).all():
            raise ValueError(
                "covariance_matrix and port_expected_returns must be finite"
            )
        covariance = (covariance + covariance.T) / 2.0
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        scale = max(1.0, float(np.max(np.abs(eigenvalues))))
        if eigenvalues.min() < -1e-10 * scale:
            raise ValueError("covariance_matrix must be positive semidefinite")
        covariance = (eigenvectors * np.maximum(eigenvalues, 0.0)) @ eigenvectors.T
        return covariance, expected_returns

    @staticmethod
    def _result(
        weights: np.ndarray,
        objective_value: float,
        diagnostic: OptimizationDiagnostic,
    ) -> OptimizeResult:
        return OptimizeResult(
            x=weights,
            fun=-objective_value,
            success=diagnostic.certified,
            status=0 if diagnostic.certified else 1,
            message=f"{diagnostic.solver}: {diagnostic.status}",
            optimization_diagnostic=diagnostic,
        )

    @staticmethod
    def _ordinary_sharpe(
        weights: np.ndarray, covariance: np.ndarray, excess_returns: np.ndarray
    ) -> float:
        variance = float(weights @ covariance @ weights)
        if variance <= 0.0:
            raise RuntimeError("portfolio variance must be positive")
        return float(weights @ excess_returns / np.sqrt(variance))

    @staticmethod
    def _covariance_factor(covariance: np.ndarray) -> np.ndarray:
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        return np.sqrt(np.maximum(eigenvalues, 0.0))[:, None] * eigenvectors.T

    def _solve_problem(self, problem: cp.Problem) -> None:
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="Solution may be inaccurate.*",
                    category=UserWarning,
                )
                problem.solve(
                    solver=self._SOLVER,
                    tol_gap_abs=1e-10,
                    tol_gap_rel=1e-10,
                    tol_feas=1e-10,
                )
        except cp.SolverError as error:
            raise RuntimeError("convex portfolio optimization failed") from error

    def optimize_towards_sharpe_ratio(
        self, covariance_matrix, port_expected_returns, equity_names: List[str]
    ) -> OptimizeResult:
        """Globally maximize long-only, fully-invested expected Sharpe ratio.

        With positive optimal excess return, rescaling transforms the fractional
        problem into a convex quadratic program. The diagnostic records the
        solver's optimal status for the transformed problem.
        """
        covariance, expected_returns = self._validated_optimization_inputs(
            covariance_matrix, port_expected_returns, equity_names
        )
        excess_returns = expected_returns - self.r_free
        if float(np.max(excess_returns)) <= 0.0:
            volatility = np.sqrt(np.maximum(np.diag(covariance), 0.0))
            nonzero_volatility = volatility > 0.0
            if not nonzero_volatility.any():
                raise ValueError("covariance_matrix must contain positive variance")
            scores = np.full(len(equity_names), -np.inf)
            scores[nonzero_volatility] = (
                excess_returns[nonzero_volatility] / volatility[nonzero_volatility]
            )
            weights = np.zeros(len(equity_names))
            weights[int(np.argmax(scores))] = 1.0
            objective_value = float(np.max(scores))
            diagnostic = OptimizationDiagnostic(
                solver="analytic negative-excess-return case",
                status="optimal",
                objective_value=objective_value,
                lower_bound=objective_value,
                upper_bound=objective_value,
                optimality_gap=0.0,
                certified=True,
                iterations=0,
            )
            self.last_optimization_diagnostic = diagnostic
            return self._result(weights, objective_value, diagnostic)

        scaled_weights = cp.Variable(len(equity_names), nonneg=True)
        problem = cp.Problem(
            cp.Minimize(cp.quad_form(scaled_weights, covariance)),
            [excess_returns @ scaled_weights == 1.0],
        )
        self._solve_problem(problem)
        if problem.status != cp.OPTIMAL or scaled_weights.value is None:
            raise RuntimeError(f"convex Sharpe optimization failed: {problem.status}")

        scaled_value = np.maximum(np.asarray(scaled_weights.value).reshape(-1), 0.0)
        weights = scaled_value / scaled_value.sum()
        objective_value = self._ordinary_sharpe(weights, covariance, excess_returns)
        diagnostic = OptimizationDiagnostic(
            solver=self._SOLVER_NAME,
            status=str(problem.status),
            objective_value=objective_value,
            lower_bound=objective_value,
            upper_bound=objective_value,
            optimality_gap=0.0,
            certified=True,
            iterations=getattr(problem.solver_stats, "num_iters", None),
        )
        self.last_optimization_diagnostic = diagnostic
        return self._result(weights, objective_value, diagnostic)

    @staticmethod
    def get_cvar_percentile(
        weights: pd.Series | np.ndarray | list[float],
        in_sample_returns: pd.DataFrame,
        cvar_percentile: float = 0.95,
    ) -> float:
        """Return empirical expected shortfall as a signed portfolio return.

        The calculation averages the worst ``1 - cvar_percentile`` share of
        portfolio outcomes and fractionally weights the boundary observation.
        """
        if not isinstance(in_sample_returns, pd.DataFrame) or in_sample_returns.empty:
            raise ValueError("in_sample_returns must be a non-empty DataFrame")
        if not 0.0 < cvar_percentile < 1.0:
            raise ValueError("cvar_percentile must be between zero and one")
        if (
            in_sample_returns.columns.has_duplicates
            or in_sample_returns.isna().any().any()
            or not np.isfinite(in_sample_returns.to_numpy(dtype=float)).all()
        ):
            raise ValueError(
                "in_sample_returns must have unique columns and finite values"
            )

        if isinstance(weights, pd.Series):
            if not weights.index.is_unique or set(weights.index) != set(
                in_sample_returns.columns
            ):
                raise ValueError(
                    "labelled weights must match the in_sample_returns columns"
                )
            aligned_weights = weights.reindex(in_sample_returns.columns).astype(float)
        else:
            aligned_weights = pd.Series(weights, index=in_sample_returns.columns)
        if (
            len(aligned_weights) != len(in_sample_returns.columns)
            or not np.isfinite(aligned_weights.to_numpy(dtype=float)).all()
        ):
            raise ValueError("weights must contain one finite value per asset")

        portfolio_returns = in_sample_returns.astype(float) @ aligned_weights
        losses = np.sort(-portfolio_returns.to_numpy(dtype=float))[::-1]
        tail_observations = (1.0 - cvar_percentile) * len(losses)
        full_observations = int(np.floor(tail_observations))
        fractional_observation = tail_observations - full_observations
        weighted_tail_loss = float(losses[:full_observations].sum())
        if fractional_observation > 0.0:
            weighted_tail_loss += fractional_observation * float(
                losses[full_observations]
            )
        return -weighted_tail_loss / tail_observations

    def optimize_towards_tail_adjusted_sharpe_ratio(
        self,
        covariance_matrix,
        port_expected_returns,
        in_sample_returns: pd.DataFrame,
        equity_names: List[str],
        tail_loss_weight: float = 1.0,
        cvar_percentile: float = 0.95,
    ) -> OptimizeResult:
        """Globally solve the tail-adjusted Sharpe fractional program.

        The numerator is expected excess return. The denominator combines
        volatility with nonnegative empirical CVaR loss. Dinkelbach iterations
        solve a sequence of concave maximization problems and yield an audited
        lower and upper bound on the globally optimal ratio.
        """
        if tail_loss_weight < 0:
            raise ValueError("tail_loss_weight cannot be negative")
        if not 0.0 < cvar_percentile < 1.0:
            raise ValueError("cvar_percentile must be between zero and one")
        covariance, expected_returns = self._validated_optimization_inputs(
            covariance_matrix, port_expected_returns, equity_names
        )
        if not isinstance(in_sample_returns, pd.DataFrame):
            raise TypeError("in_sample_returns must be a DataFrame")
        if set(in_sample_returns.columns) != set(equity_names):
            raise ValueError("in_sample_returns columns must match equity_names")
        returns = in_sample_returns.reindex(columns=equity_names).to_numpy(dtype=float)
        if not np.isfinite(returns).all() or len(returns) == 0:
            raise ValueError("in_sample_returns must contain finite observations")
        excess_returns = expected_returns - self.r_free
        if float(np.max(excess_returns)) <= 0.0:
            raise ValueError(
                "tail-adjusted Sharpe requires an asset with positive expected excess return"
            )
        if tail_loss_weight == 0.0:
            return self.optimize_towards_sharpe_ratio(
                covariance, expected_returns, equity_names
            )
        if np.all(returns >= 0.0):
            return self.optimize_towards_sharpe_ratio(
                covariance, expected_returns, equity_names
            )

        ordinary_result = self.optimize_towards_sharpe_ratio(
            covariance, expected_returns, equity_names
        )
        ordinary_upper_bound = -float(ordinary_result.fun)

        def score_tail_adjusted(
            candidate: np.ndarray,
        ) -> tuple[float, np.ndarray, float]:
            normalized = np.maximum(np.asarray(candidate).reshape(-1), 0.0)
            if normalized.sum() <= 0.0:
                return -np.inf, normalized, np.inf
            normalized /= normalized.sum()
            tail_loss_value = max(
                0.0,
                -self.get_cvar_percentile(
                    normalized, in_sample_returns[equity_names], cvar_percentile
                ),
            )
            volatility_value = np.sqrt(float(normalized @ covariance @ normalized))
            denominator = volatility_value + tail_loss_weight * tail_loss_value
            score = float(normalized @ excess_returns / denominator)
            return score, normalized, denominator / return_scale

        asset_count = len(equity_names)
        observation_count = len(returns)
        tail_probability = 1.0 - cvar_percentile
        return_scale = max(
            float(np.max(np.abs(returns))),
            float(np.max(np.sqrt(np.maximum(np.diag(covariance), 0.0)))),
            1e-8,
        )
        scaled_returns = returns / return_scale
        scaled_excess_returns = excess_returns / return_scale
        weights = cp.Variable(asset_count, nonneg=True)
        threshold = cp.Variable()
        tail_loss = cp.Variable(nonneg=True)
        target = cp.Parameter(nonneg=True)
        factor = self._covariance_factor(covariance) / return_scale
        volatility = cp.norm(factor @ weights, 2)
        cvar_loss = threshold + cp.sum(
            cp.pos(-scaled_returns @ weights - threshold)
        ) / (tail_probability * observation_count)
        denominator = volatility + tail_loss_weight * tail_loss
        constraints = [
            cp.sum(weights) == 1.0,
            tail_loss >= cvar_loss,
        ]
        minimum_denominator_problem = cp.Problem(cp.Minimize(denominator), constraints)
        self._solve_problem(minimum_denominator_problem)
        if minimum_denominator_problem.status not in {
            cp.OPTIMAL,
            cp.OPTIMAL_INACCURATE,
        }:
            raise RuntimeError(
                "tail-adjusted minimum-denominator problem did not solve optimally"
            )
        denominator_floor = float(minimum_denominator_problem.value)
        if denominator_floor <= 0.0:
            raise RuntimeError("tail-adjusted denominator must be positive")

        dinkelbach_problem = cp.Problem(
            cp.Maximize(scaled_excess_returns @ weights - target * denominator),
            constraints,
        )
        lower_bound, candidate_weights, _ = score_tail_adjusted(ordinary_result.x)
        residual = np.inf
        iterations = 0
        for iterations in range(1, 51):
            target.value = lower_bound
            self._solve_problem(dinkelbach_problem)
            if (
                dinkelbach_problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}
                or weights.value is None
            ):
                raise RuntimeError(
                    "tail-adjusted Dinkelbach subproblem did not solve optimally"
                )
            candidate_score, normalized_candidate, candidate_denominator = (
                score_tail_adjusted(weights.value)
            )
            if candidate_score < lower_bound - 1e-8:
                raise RuntimeError("tail-adjusted optimizer returned a worse candidate")
            residual = max(
                0.0,
                (candidate_score - lower_bound) * candidate_denominator,
            )
            lower_bound = candidate_score
            candidate_weights = normalized_candidate
            if residual / denominator_floor <= self._BISSECTION_TOLERANCE:
                break

        objective_value, candidate_weights, _ = score_tail_adjusted(candidate_weights)
        upper_bound = min(
            ordinary_upper_bound,
            objective_value + residual / denominator_floor,
        )
        optimality_gap = max(0.0, upper_bound - objective_value)
        diagnostic = OptimizationDiagnostic(
            solver=f"{self._SOLVER_NAME} Dinkelbach",
            status="optimal within residual tolerance",
            objective_value=objective_value,
            lower_bound=objective_value,
            upper_bound=upper_bound,
            optimality_gap=optimality_gap,
            certified=optimality_gap <= self._BISSECTION_TOLERANCE,
            iterations=iterations,
        )
        self.last_optimization_diagnostic = diagnostic
        return self._result(candidate_weights, objective_value, diagnostic)

    def weights_from_returns(self, in_sample_returns: pd.DataFrame) -> pd.Series:
        """Fit this strategy to an in-sample return panel and return target weights."""
        if not isinstance(in_sample_returns, pd.DataFrame) or in_sample_returns.empty:
            raise ValueError("in_sample_returns must be a non-empty DataFrame")
        covariance_matrix = self.get_covariance_matrix(None, None, in_sample_returns)
        expected_returns = self.get_expected_returns(None, None, in_sample_returns)
        result = self.optimize_towards_sharpe_ratio(
            covariance_matrix, expected_returns, list(in_sample_returns.columns)
        )
        if not result.success:
            raise RuntimeError(f"portfolio optimization failed: {result.message}")
        return pd.Series(result.x, index=in_sample_returns.columns, name="weight")

    def _get_returns(self, tickers, start_date, end_date):
        return get_returns(tickers=tickers, start_date=start_date, end_date=end_date)

    def get_equity_data(self, start_date, end_date, equity_names):
        return self._get_returns(
            tickers=equity_names, start_date=start_date, end_date=end_date
        )

    def get_covariance_matrix(self, start_date, end_date, equity_data):
        raise NotImplementedError

    def get_expected_returns(self, start_date, end_date, equity_data):
        raise NotImplementedError

    def get_portfolio_weights_sharpe(self, start_date, end_date, equity_names):
        equity_data = self.get_equity_data(start_date, end_date, equity_names)
        covariance_matrix = self.get_covariance_matrix(
            start_date, end_date, equity_data
        )
        port_expected_returns = self.get_expected_returns(
            start_date, end_date, equity_data
        )
        return self.optimize_towards_sharpe_ratio(
            covariance_matrix, port_expected_returns, equity_names
        )
