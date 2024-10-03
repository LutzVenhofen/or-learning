# Traveling Sales Person Problem
A classic in OR with many different variants and countless options to solve it. In this folder I will try out different techniques. For my own curiosity this might include solution approaches, which are not the fastest.

# MIP based approach
The TSP can be modelled as a MIP. In the base case our subtour elimination constraints will quickly overwhelm us and our computer's memory. As such subtours are added on a case by case basis. A solution approach can look like this:

1. Build the problem without subtour constraints.
2. Solve it.
3. Check the solution for subtours.
4. If subtours identified, add constraints for those found. Go back to step 2.
5. If no more subtours identified, we have our optimal TSP tour.

Cool people will add subtours as lazy constraints via callbacks. For now I tried out MathOpt part of Google OR Tools and fairly new. No callbacks are available for open source solvers via the python MathOpt interface, as of writing.

I employ another trick: solve the LP relexation first. Basically we solve a doubley relaxed problem (not all subtour elimination constraints + lp). Goal is that the lp is solved very fast and we can already collect a bunch of subtour elimination constraints, relevant for our instance.