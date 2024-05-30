from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple

from typing_extensions import Protocol

# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""
    Computes an _approximation_ to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
        f       : an arbitrary function that _reduces_ n-scalar args to one value
        *vals   : a tensor of n-float values $x_0 \ldots x_{n-1}$
        arg     : the number $i$ of the arg to compute the derivative
        epsilon : a small stepsize constant $h$

    Returns:
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$
    """
    right, left = [*vals], [*vals]
    right[arg] += epsilon  # t[x+h]
    left[arg] -= epsilon  # t[x-h]
    return (f(*right) - f(*left)) / (2 * epsilon)


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        pass

    @property
    def unique_id(self) -> int:
        pass

    def is_leaf(self) -> bool:
        return self.history.last_fn is None

    def is_constant(self) -> bool:
        pass

    @property
    def parents(self) -> Iterable["Variable"]:
        pass

    def chain_rule(self, d_output: Any) -> Iterable[Tuple["Variable", Any]]:
        pass


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """
    Computes the topological order of the computation graph.

    Args:
        variable: The right-most variable

    Returns:
        Non-constant Variables in topological order starting from the right.
    """
    map: List[Variable] = []
    visited = set()

    def map_from(var: Variable) -> None:
        if var.unique_id in visited:
            return
        if not var.is_leaf():
            for input in var.history.inputs:
                if not Variable.is_constant(input):
                    map_from(input)
        visited.add(var.unique_id)
        map.insert(0, var)

    map_from(variable)
    return map


def backpropagate(variable: Variable, deriv: Any) -> None:
    """
    Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
        variable: The right-most variable
        deriv  : Its derivative that we want to propagate backward to the leaves.

    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.
    """

    # TODO: Implement for Task 1.4.
    remaining = [variable]

    d = deriv
    while len(remaining) > 0:
        current = remaining.pop(0)

        if current.history.is_leaf():
            current.accumulate_derivative(d)
        else:
            terms = current.history.last_fn.chain_rule(
                variable.history.ctx, variable.history.inputs, d
            )
            for each in terms:
                remaining += [each]
        pass
    #
    # NOTE: the variable info must be filled in, including the accumulated derivative.


@dataclass
class Context:
    """
    Context class is used by `Function` to store information during the forward pass.
    """

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        "Store the given `values` if they need to be used during backpropagation."
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        return self.saved_values
