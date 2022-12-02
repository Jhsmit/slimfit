from functools import reduce
from operator import add
from typing import Optional, Callable, Type

import numpy as np
from sympy import Matrix, zeros, Symbol

OPERATORS = ["<->", "<-", "->"]


def generate_transition_matrix(
    connectivity: list[str],
    parameter_prefix="k",
    check_mass_balance=True,
    symbol_class: Type[Symbol] = Symbol,
):
    all_states = extract_states(connectivity)

    b = ["_" in s for s in all_states]
    if any(b):
        raise ValueError("Underscores are not allowed in state names")

    trs_matrix = zeros(len(all_states), len(all_states))
    for conn in connectivity:
        split = conn.split(" ")
        states = [s for s in split if not s in OPERATORS]

        for current_state in states:
            i = split.index(current_state)
            current_idx = all_states.index(current_state)

            # look to the left
            if i >= 2:
                op = split[i - 1]
                other_state = split[i - 2]  # refactor other
                other_idx = all_states.index(other_state)
                if op in ["->", "<->"]:  # flux from other state to current state
                    # elem = self.create_element(other_state, current_state)
                    elem = symbol_class(f"{parameter_prefix}_{other_state}_{current_state}")
                    trs_matrix[current_idx, other_idx] += elem

                if op in ["<-", "<->"]:  # flux from current state to other state
                    # elem = self.create_element(current_state, other_state)
                    elem = symbol_class(f"{parameter_prefix}_{current_state}_{other_state}")
                    trs_matrix[current_idx, current_idx] -= elem

            # look to the right
            if i <= len(split) - 2:
                op = split[i + 1]
                other_state = split[i + 2]
                other_idx = all_states.index(other_state)

                if op in ["<-", "<->"]:  # flux from other state to current state
                    # elem = self.create_element(other_state, current_state)
                    elem = symbol_class(f"{parameter_prefix}_{other_state}_{current_state}")
                    trs_matrix[current_idx, other_idx] += elem
                if op in ["->", "<->"]:  # flux from current state to other state
                    # elem = self.create_element(current_state, other_state)
                    elem = symbol_class(f"{parameter_prefix}_{current_state}_{other_state}")
                    trs_matrix[current_idx, current_idx] -= elem

    return trs_matrix


def exp_elements(
    m: Matrix, base=10, sub_name: Optional[Callable] = lambda x: "u" + x[1:]
) -> Matrix:
    """
    Exponentiate elements of matrix to a given base, and optionally substitute parameters with new parameters

    Args:
        m: Input matrix which elements to exponentiate.
        base: Base of the exponent.
        sub_name: Function which takes parameter name and returns substitute parameter name.

    Returns:
        Exponentiated matrix

    """
    out = zeros(*m.shape)
    for i, j in np.ndindex(m.shape):
        elem = m[i, j]
        if elem == 0.0:
            continue
        else:
            out[i, j] = base ** elem

    if sub_name is not None:
        subs = [(p, Parameter(name=sub_name(p.name))) for p in t.free_symbols]
        out = out.subs(subs)

    return out


def extract_states(connectivity: list[str]) -> list[str]:
    """
    Args:
        connectivity: List of reaction equations.

    Returns:
        List of states found in all reaction equations.

    """

    # extract states from connectivity list
    all_states = [
        s for s in reduce(add, [eqn.split(" ") for eqn in connectivity]) if s not in OPERATORS
    ]

    # Remove duplicates, keep order
    all_states = list(dict.fromkeys(all_states))

    return all_states
