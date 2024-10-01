from collections import defaultdict
from itertools import product
from ortools.math_opt.python import mathopt


def create_basic_tsp_model(name: str, num_cities: int, dist: list[list[int]]) -> mathopt.Model:
    model = mathopt.Model(name=name)
    x_vars: dict[int, dict[int, mathopt.Variable]] = defaultdict(dict)
    for i, j in product(range(num_cities), repeat=2):
        if i != j:
            x_vars[i][j] = model.add_binary_variable(name=f"x_{i}_{j}")
    for i in range(num_cities):
        model.add_linear_constraint(mathopt.fast_sum(x_vars[i][j] for j in x_vars[i]) == 1)
        model.add_linear_constraint(mathopt.fast_sum(x_vars[j][i] for j in x_vars if j != i) == 1)
    model.minimize(
        mathopt.fast_sum(x_vars[i][j] * dist[i][j] for i in x_vars for j in x_vars[i])
    )
    return model


def run_model(model: mathopt.Model) -> mathopt.SolveResult:
    params = mathopt.SolveParameters(enable_output=True)
    result = mathopt.solve(model, solver_type=mathopt.SolverType.HIGHS, params=params)
    return result


def main():
    model = create_basic_tsp_model("my first tsp", 3, [[0, 1,  2], [3, 0, 3], [4, 4, 0]])
    result = run_model(model)
    if result.termination.reason != mathopt.TerminationReason.OPTIMAL:
        raise RuntimeError(f"model failed to solve: {result.termination}")
    print("MathOpt solve succeeded")
    print("Objective value:", result.objective_value())


if __name__ == "__main__":
    main()