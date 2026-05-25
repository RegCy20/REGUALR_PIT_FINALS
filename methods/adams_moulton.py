"""Adams–Moulton Predictor–Corrector method implementation."""
from __future__ import annotations

import ast
import math
from dataclasses import dataclass
from typing import Callable, Dict, List


ALLOWED_NAMES: Dict[str, float | Callable] = {
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "exp": math.exp,
    "log": math.log,
    "sqrt": math.sqrt,
    "pi": math.pi,
    "e": math.e,
    "abs": abs,
    "pow": pow,
}

ALLOWED_NODES = (
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.Constant,
    ast.Name,
    ast.Load,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.Pow,
    ast.Mod,
    ast.USub,
    ast.UAdd,
    ast.Call,
)


@dataclass
class StepResult:
    n: int
    x: float
    y: float
    f: float
    predictor: float | None = None
    corrector: float | None = None


def validate_expression(expr: str) -> ast.Expression:
    """Validate user function input before evaluation."""
    if not expr:
        raise ValueError("Please enter a function.")

    tree = ast.parse(expr, mode="eval")

    for node in ast.walk(tree):
        if not isinstance(node, ALLOWED_NODES):
            raise ValueError(f"Invalid expression element: {type(node).__name__}")

        if isinstance(node, ast.Name):
            if node.id not in ALLOWED_NAMES and node.id not in {"x", "y"}:
                raise ValueError(f"Invalid variable or function name: {node.id}")

        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise ValueError("Only simple function calls like sin(x) are allowed.")
            if node.func.id not in ALLOWED_NAMES:
                raise ValueError(f"Function not allowed: {node.func.id}")

    return tree


def make_function(expr: str) -> Callable[[float, float], float]:
    """Create a safe numerical function f(x,y)."""
    tree = validate_expression(expr)
    code = compile(tree, "<user_function>", "eval")

    def f(x: float, y: float) -> float:
        env = {"x": x, "y": y, **ALLOWED_NAMES}
        return float(eval(code, {"__builtins__": {}}, env))

    return f


def rk4_one_step(f: Callable[[float, float], float], x: float, y: float, h: float) -> float:
    """Use RK4 to generate the second starting value needed by AB2-AM2."""
    k1 = f(x, y)
    k2 = f(x + h / 2, y + h * k1 / 2)
    k3 = f(x + h / 2, y + h * k2 / 2)
    k4 = f(x + h, y + h * k3)
    return y + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)


def solve(expr: str, x0: float, y0: float, h: float, steps: int, corrections: int = 1) -> List[StepResult]:
    """Solve y'=f(x,y) using AB2 predictor and AM2 corrector."""
    if h == 0:
        raise ValueError("Step size h cannot be zero.")
    if steps < 1:
        raise ValueError("Steps must be at least 1.")
    if corrections < 1:
        raise ValueError("Corrections must be at least 1.")

    f = make_function(expr)
    rows: List[StepResult] = []

    f0 = f(x0, y0)
    rows.append(StepResult(n=0, x=x0, y=y0, f=f0))

    y1 = rk4_one_step(f, x0, y0, h)
    x1 = x0 + h
    f1 = f(x1, y1)
    rows.append(StepResult(n=1, x=x1, y=y1, f=f1))

    for n in range(1, steps):
        prev = rows[n - 1]
        curr = rows[n]

        x_next = curr.x + h

        # Adams-Bashforth 2-step predictor
        y_pred = curr.y + (h / 2) * (3 * curr.f - prev.f)

        # Adams-Moulton 2-step corrector
        y_corr = y_pred
        for _ in range(corrections):
            f_pred = f(x_next, y_corr)
            y_corr = curr.y + (h / 12) * (5 * f_pred + 8 * curr.f - prev.f)

        f_next = f(x_next, y_corr)
        rows.append(
            StepResult(
                n=n + 1,
                x=x_next,
                y=y_corr,
                f=f_next,
                predictor=y_pred,
                corrector=y_corr,
            )
        )

    return rows
