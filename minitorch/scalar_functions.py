from __future__ import annotations

from typing import TYPE_CHECKING

import minitorch

from . import operators
from .autodiff import Context

if TYPE_CHECKING:
    from typing import Tuple

    from .scalar import Scalar, ScalarLike


def wrap_tuple(x):  # type: ignore
    "Turn a possible value into a tuple"
    if isinstance(x, tuple):
        return x
    return (x,)


# NOTE: might not need to implement for operators because python provides unpacking
def unwrap_tuple(x):  # type: ignore
    "Turn a singleton tuple into a value"
    if len(x) == 1:
        return x[0]
    return x


class ScalarFunction:
    """
    A wrapper for a mathematical function that processes and produces
    Scalar variables.

    This is a static class and is never instantiated. We use `class`
    here to group together the `forward` and `backward` code.
    """

    @classmethod
    def _backward(cls, ctx: Context, d_out: float) -> Tuple[float, ...]:
        return wrap_tuple(cls.backward(ctx, d_out))  # type: ignore

    @classmethod
    def _forward(cls, ctx: Context, *inps: float) -> float:
        return cls.forward(ctx, *inps)  # type: ignore

    @classmethod
    def apply(cls, *vals: "ScalarLike") -> Scalar:
        raw_vals = []
        scalars = []
        for v in vals:
            if isinstance(v, minitorch.scalar.Scalar):
                scalars.append(v)
                raw_vals.append(v.data)
            else:
                scalars.append(minitorch.scalar.Scalar(v))
                raw_vals.append(v)

        # Create the context.
        ctx = Context(False)

        # Call forward with the variables.
        c = cls._forward(ctx, *raw_vals)
        assert isinstance(c, float), "Expected return type float got %s" % (type(c))

        # Create a new variable from the result with a new history.
        back = minitorch.scalar.ScalarHistory(cls, ctx, scalars)
        return minitorch.scalar.Scalar(c, back)


# Examples
class Add(ScalarFunction):
    "Addition function $f(x, y) = x + y$"

    @staticmethod
    def forward(ctx: Context, x: float, y: float) -> float:
        x, y = operators.id(x), operators.id(y)
        ctx.save_for_backward(x, y)
        return operators.add(x, y)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, ...]:
        return d_output, d_output


class Log(ScalarFunction):
    "Log function $f(x) = log(x)$"

    @staticmethod
    def forward(ctx: Context, x: float) -> float:
        x = operators.id(x)
        ctx.save_for_backward(x)
        return operators.log(x)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        (x,) = ctx.saved_values
        return operators.log_back(x, d_output)


# To implement.


class Mul(ScalarFunction):
    "Multiplication function"

    @staticmethod
    def forward(ctx: Context, x: float, y: float) -> float:
        x, y = operators.id(x), operators.id(y)
        ctx.save_for_backward(x, y)
        return operators.mul(x, y)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        grad_y, grad_x = ctx.saved_values
        return (
            d_output * grad_x,
            d_output * grad_y,
        )  # order doesn't matter because they get summed


class Inv(ScalarFunction):
    "Inverse function"

    @staticmethod
    def forward(ctx: Context, x: float) -> float:
        x = operators.id(x)
        ctx.save_for_backward(x)
        return operators.inv(x)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        (x,) = ctx.saved_values
        grad_x = operators.neg(operators.mul(d_output, x))
        return grad_x


class Neg(ScalarFunction):
    "Negation function"

    @staticmethod
    def forward(ctx: Context, x: float) -> float:
        x = operators.id(x)
        ctx.save_for_backward(x)
        return operators.neg(x)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        (x,) = ctx.saved_values
        grad_x = operators.mul(d_output, x)
        return grad_x


class Sigmoid(ScalarFunction):
    "Sigmoid function"

    @staticmethod
    def forward(ctx: Context, x: float) -> float:
        ctx.save_for_backward(x)
        return operators.sigmoid(x)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        (x,) = ctx.saved_values
        if x >= 0.0:
            return d_output * 1.0
        else:
            return d_output * (1.0 + operators.exp(x))


class ReLU(ScalarFunction):
    "ReLU function"

    @staticmethod
    def forward(ctx: Context, x: float) -> float:
        x = operators.id(x)
        ctx.save_for_backward(x)
        return operators.relu(x)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        (x,) = ctx.saved_values
        grad_x = operators.relu_back(x, d_output)
        return grad_x


class Exp(ScalarFunction):
    "Exp function"

    @staticmethod
    def forward(ctx: Context, x: float) -> float:
        ctx.save_for_backward(x)
        return operators.exp(x)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        (x,) = ctx.saved_values
        grad_x = d_output * operators.exp(x)
        return grad_x


class LT(ScalarFunction):
    "Less-than function $f(x) =$ 1.0 if x is less than y else 0.0"

    @staticmethod
    def forward(ctx: Context, x: float, y: float) -> float:
        x, y = operators.id(x), operators.id(y)
        ctx.save_for_backward(x, y)
        return operators.lt(x, y)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        # x, y = ctx.saved_values
        return d_output, d_output


class EQ(ScalarFunction):
    "Equal function $f(x) =$ 1.0 if x is equal to y else 0.0"

    @staticmethod
    def forward(ctx: Context, x: float, y: float) -> float:
        x, y = operators.id(x), operators.id(y)
        ctx.save_for_backward(x, y)
        return operators.eq(x, y)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        # x, y = ctx.saved_values
        return d_output, d_output
