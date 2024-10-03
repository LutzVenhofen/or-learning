from collections import defaultdict
from itertools import product
from dataclasses import dataclass, field
from enum import Enum
from ortools.math_opt.python import mathopt
import rustworkx as rwx
from rustworkx.visualization import graphviz_draw
import numpy as np


class ModelOptions(Enum):
    RELAXED = "relaxed"
    INTEGER = "integer"


@dataclass
class TspMip:
    x_vars: dict[int, dict[int, mathopt.Variable]]
    model: mathopt.Model
    model_option: ModelOptions
    result: mathopt.SolveResult | None = None
    subtour_cuts: list[set[int]] = field(default_factory=list)


@dataclass
class SubtourCutResult:
    subtour_cuts_added: bool
    resulting_model: TspMip


def create_basic_tsp_model(
    name: str,
    num_cities: int,
    dist: list[list[float]],
    model_option: ModelOptions,
    subtour_cuts: list[set[int]] | None = None,
) -> TspMip:
    tsp_mip = TspMip(defaultdict(dict), mathopt.Model(name=name), model_option=model_option)
    # variables
    for i, j in product(range(num_cities), repeat=2):
        if i != j:
            match model_option:
                case ModelOptions.INTEGER:
                    tsp_mip.x_vars[i][j] = tsp_mip.model.add_binary_variable(name=f"x_{i}_{j}")
                case ModelOptions.RELAXED:
                    tsp_mip.x_vars[i][j] = tsp_mip.model.add_variable(lb=0, ub=1, name=f"x_{i}_{j}_r")
    # visit every node
    for i in range(num_cities):
        tsp_mip.model.add_linear_constraint(
            mathopt.fast_sum(tsp_mip.x_vars[i][j] for j in tsp_mip.x_vars[i]) == 1
        )
        tsp_mip.model.add_linear_constraint(
            mathopt.fast_sum(tsp_mip.x_vars[j][i] for j in tsp_mip.x_vars if j != i) == 1
        )
    # two node subtour elimination
    for i, j in product(range(num_cities), repeat=2):
        if i != j:
            tsp_mip.model.add_linear_constraint(tsp_mip.x_vars[i][j] + tsp_mip.x_vars[j][i] <= 1)
    # if we already have some valid subtour cuts add them
    if subtour_cuts:
        add_subtour_cuts(tsp_mip, subtour_cuts)
    tsp_mip.model.minimize(
        mathopt.fast_sum(
            tsp_mip.x_vars[i][j] * dist[i][j] for i in tsp_mip.x_vars for j in tsp_mip.x_vars[i]
        )
    )
    return tsp_mip


def run_model(model: mathopt.Model) -> mathopt.SolveResult:
    params = mathopt.SolveParameters(enable_output=True)
    result = mathopt.solve(model, solver_type=mathopt.SolverType.HIGHS, params=params)
    if result.termination.reason != mathopt.TerminationReason.OPTIMAL:
        raise RuntimeError(f"model failed to solve: {result.termination}")
    print("MathOpt solve succeeded")
    print(f"Objective value: {result.objective_value()}")
    return result


def extract_and_add_subtour_cuts(
    tsp_mip: TspMip,
    potential_edges: list[tuple[int, int, float]],
    n: int
) -> SubtourCutResult:
    result_graph = create_graph_from(potential_edges, n)
    subtours = rwx.connected_components(result_graph)
    if len(subtours) == 1:
        return SubtourCutResult(False, tsp_mip)
    tsp_mip.subtour_cuts += subtours
    add_subtour_cuts(tsp_mip, subtours)
    return SubtourCutResult(True, tsp_mip)


def add_subtour_cuts(tsp_mip: TspMip, subtours: list[set[int]]):
    for subtour in subtours:
        tsp_mip.model.add_linear_constraint(
            mathopt.fast_sum(tsp_mip.x_vars[i][j] for i in subtour for j in subtour if i != j)
            <=
            len(subtour) - 1
        )


def create_graph_from(
    potential_edges: list[tuple[int, int, float]],
    n: int,
) -> rwx.PyGraph:
    result_graph = rwx.PyGraph()
    result_graph.add_nodes_from(range(n))
    result_graph.add_edges_from(potential_edges)
    return result_graph


def extract_edges_from_model(
    tsp_mip: TspMip,
    dist: list[list[float]],
    eps: float = 0.5
) -> list[tuple[int, int, float]]:
    var_values =  tsp_mip.result.variable_values()
    return [
        (i, j, dist[i][j]) for i in tsp_mip.x_vars for j in tsp_mip.x_vars[i]
        if var_values[tsp_mip.x_vars[i][j]] > eps
    ]


def solve_with_iterative_subtour_cuts(
    tsp_mip: TspMip,
    dist: list[list[float]],
    n: int,
    eps=float,
) -> tuple[TspMip, list[float]]:
    scr = SubtourCutResult(True, tsp_mip)
    iterations: list[float] = []
    while scr.subtour_cuts_added:
        scr.resulting_model.result = run_model(scr.resulting_model.model)
        iterations.append(scr.resulting_model.result.objective_value())
        scr = extract_and_add_subtour_cuts(
            scr.resulting_model,
            extract_edges_from_model(scr.resulting_model, dist, eps),
            n,
        )
    print(f"Finished after {len(iterations)} iterations.")
    print(f"Objective value history {iterations}")
    return scr.resulting_model, iterations


def create_random_tsp(n: int):
    np.random.seed(0)

    X = 100 * np.random.rand(n)
    Y = 100 * np.random.rand(n)

    # Compute distance matrix
    dist = np.ceil(np.sqrt ((X.reshape(n,1) - X.reshape(1,n))**2 +
                            (Y.reshape(n,1) - Y.reshape(1,n))**2))
    return dist


def main():
    n = 20
    dist = create_random_tsp(n)

    # first run the model as lp, collect subtour elimination constraints fast
    eps = 1.e-6
    tsp_lp = create_basic_tsp_model("Relaxed TSP", n, dist, ModelOptions.RELAXED)
    tsp_lp, lp_iterations = solve_with_iterative_subtour_cuts(tsp_lp, dist, n, eps)

    # when no more lp subtours found rebuild and rerun as integer model
    eps = 0.8
    tsp_mip = create_basic_tsp_model(
        "Integer TSP", n, dist, ModelOptions.INTEGER, subtour_cuts=tsp_lp.subtour_cuts
    )
    tsp_mip, mip_iterations = solve_with_iterative_subtour_cuts(tsp_mip, dist, n, eps)

    # visualize and report results
    final_edges = extract_edges_from_model(tsp_mip, dist)
    final_graph = create_graph_from(final_edges, n)
    graphviz_draw(final_graph, filename="tsp.png")
    print(f"Number lp iterations {len(lp_iterations)}")
    print(f"LP obj values {lp_iterations}")
    print(f"Number MIP iterations {len(mip_iterations)}")
    print(f"MIP obj values {mip_iterations}")
    print("Finished")
    


if __name__ == "__main__":
    main()