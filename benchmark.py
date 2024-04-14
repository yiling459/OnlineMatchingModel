import pulp
import numpy as np

# Configuration
N = 5  # Number of products
M = 5  # Number of customer types
T = 300  # Number of periods

# Inventory scenarios
inventory_levels = {'scarce': 1, 'moderate': 5, 'abundant': 20}
inventories = {level: [inventory_levels[level]] * N for level in inventory_levels}

# Prices
prices = np.linspace(15, 30, N)

# Usage time parameters: Geometric distribution parameters between 0.05 and 0.07
# 1/(20-i) ensures that product type i has pi = 1/(20-i), giving longer expected usage time to more expensive products
usage_parameters = [1/(20 - i) for i in range(N)]  # Adjusted to provide increasing availability time for more expensive products

# Customer arrival probabilities
customer_probabilities = [1/M] * M  # Uniform distribution for simplicity

# Create the LP model
model = pulp.LpProblem("Dynamic_Assortment_Optimization", pulp.LpMaximize)

# Decision variables: Probability of selling product i to customer type m at time t
X = {(t, i, m): pulp.LpVariable(f"X_{t}_{i}_{m}", lowBound=0, upBound=1, cat='Continuous') 
     for t in range(T) for i in range(N) for m in range(M)}

# Objective Function: Maximize total expected revenue
model += pulp.lpSum([prices[i] * X[t, i, m] * customer_probabilities[m] for t in range(T) for i in range(N) for m in range(M)])

# Constraints
# Inventory constraints for each product under each inventory scenario
for level in inventories:
    for i in range(N):
        model += pulp.lpSum([X[t, i, m] * (1 - usage_parameters[i]**(t - s)) 
                             for t in range(T) for m in range(M) for s in range(t)]) <= inventories[level][i], f"Inventory_{level}_limit_for_product_{i}"

# Constraint to ensure sales probabilities respect geometric distribution constraints over time
for i in range(N):
    for t in range(1, T):
        for m in range(M):
            model += X[t, i, m] <= usage_parameters[i] * X[t-1, i, m], f"Usage_time_adjustment_{t}_{i}_{m}"

# Solve the model (example for 'moderate' inventory level)
model.solve()

# Output results
print("Optimized Revenue under Moderate Inventory:", pulp.value(model.objective))
for v in model.variables():
    if v.varValue > 0:
        print(v.name, "=", v.varValue)
