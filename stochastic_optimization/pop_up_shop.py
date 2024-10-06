from dataclasses import dataclass
from ortools.math_opt.python import mathopt

r"""Notes
Taken from the MO-Book:
https://mobook.github.io/MO-book/notebooks/09/02-pop-up-shop.html

@online{PZGK2024online,
author = {Postek, Krzysztof and Zocca, Alessandro and Gromicho, Joaquim and Kantor, Jeffrey},
title = {Companion Jupyter Book for {Hands-On Mathematical Optimization with Python}},
year = {2024},
publisher = {GitHub},
howpublished = {\url{https://github.com/mobook/MO-book}},
}
"""


@dataclass
class Prices:
    sales_price: float
    unit_cost: float
    salvage_value: float


@dataclass
class WeatherScenario:
    name: str
    demand: int
    probability: float


def build_scenario_model(
    name: str,
    prices: Prices,
    scenarios: list[WeatherScenario]
) -> mathopt.Model:
    model = mathopt.Model(name=name)
    sales_for_scenario_var: list[mathopt.Variable] = [
        model.add_variable(lb=0, ub=scenarios[s].demand, name=f"sales_scenario_{s}")
        for s in range(len(scenarios))
    ]
    planned_sales = model.add_variable(lb=0, ub=max(s.demand for s in scenarios), name="planned_sales")
    profit_for_scenario_var: list[mathopt.Variable] = [
        model.add_variable(name=f"profit_scenario_{s}") for s in range(len(scenarios))
    ]
    for s in range(len(scenarios)):
        model.add_linear_constraint(sales_for_scenario_var[s] <= planned_sales)
        model.add_linear_constraint(
            profit_for_scenario_var[s]
            ==
            sales_for_scenario_var[s] * (prices.sales_price - prices.salvage_value) + planned_sales * (prices.salvage_value - prices.unit_cost)
        )
    model.maximize(mathopt.fast_sum(profit_for_scenario_var[s] * scenarios[s].probability for s in range(len(scenarios))))
    return model


def run_model(model: mathopt.Model) -> mathopt.SolveResult:
    params = mathopt.SolveParameters(enable_output=True)
    result = mathopt.solve(model, solver_type=mathopt.SolverType.HIGHS, params=params)
    if result.termination.reason != mathopt.TerminationReason.OPTIMAL:
        raise RuntimeError(f"model failed to solve: {result.termination}")
    print("MathOpt solve succeeded")
    print(f"Objective value: {result.objective_value()}")
    return result


def main():
    scenarios: list[WeatherScenario] = [
        WeatherScenario("sunny skies", 650, 0.1),
        WeatherScenario("good weather", 400, 0.6),
        WeatherScenario("poor weather", 200, 0.3)
    ]
    prices = Prices(40, 12, 2)
    model = build_scenario_model("Book Example", prices, scenarios)
    model_result = run_model(model)
    print("Finished")


if __name__ == "__main__":
    main()
