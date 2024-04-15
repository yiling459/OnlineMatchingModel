import cvxpy as cp
import numpy as np

# Parameters
n = 5  # Number of products
T = 300  # Number of periods
prices = np.linspace(15, 30, n)  # Prices of products
inventory_levels = [1, 5, 20]  # Inventory levels
probabilities = np.array([1 / (20 - i) for i in range(1, 6)])  # Geometric distribution parameters

# Prepare survival probabilities for the geometric distribution (product availability over time)
survival_probabilities = np.array([(1 - probabilities[i]) ** np.arange(T) for i in range(n)])

# Each inventory level simulation
for inventory_level in inventory_levels:
    # Decision variable: X[t, i] is now continuous between 0 and 1
    X = cp.Variable((T, n), nonneg=True)

    # Objective function: Maximize expected total revenue
    revenue = cp.sum(cp.multiply(X, prices.reshape(1, n)))

    # Constraints
    constraints = []

    # Constraint 1: The sum of probabilities for matching items in each time period cannot exceed 1
    constraints += [cp.sum(X, axis=1) <= 1]

    # Constraint 2: Cumulative expected matching for each item should not exceed its maximum capacity
    for i in range(n):
        constraints += [cp.sum(cp.multiply(X[:, i], survival_probabilities[i])) <= inventory_level]

    # Constraint 3: X[t, i] must be between 0 and 1
    constraints += [X <= 1]

    # Constraint 4: Matching only if there is an edge - assuming all items are always available for simplicity
    # If there are specific conditions where an item is not available, l_ti should be defined accordingly
    # l_ti = np.ones((T, n))  # Example definition if you have specific availability data
    # constraints += [X <= l_ti]

    # Define the optimization problem
    prob = cp.Problem(cp.Maximize(revenue), constraints)

    # Solve the problem
    prob.solve()

    # Output results
    print(f"Inventory level {inventory_level}: Expected Max Revenue = ${prob.value:.2f}")
    print("Expected matched items matrix:")
    print(X.value)
