import numpy as np
import cvxpy as cp

class RankBasedAlgorithm:
    def __init__(self, N, T, E, rewards, usage_duration_distributions, inventory_levels):
        self.N = N  # Number of resources
        self.T = T  # Number of time steps or requests
        self.E = E  # Edges representing possible matches between resources and requests
        self.rewards = rewards  # Rewards for allocating each resource
        self.usage_duration_distributions = usage_duration_distributions  # Usage durations for each resource

        # Initialize resource units' data structures
        self.unit_availability = {}
        self.unit_rank = {}
        self.unit_return_time = {}
        self.unit_usage_count = {}
        self.inventory_levels = {}

        # Assign inventory levels cyclically across resources
        level_count = len(inventory_levels)
        for n in range(N):
            num_units = len(self.usage_duration_distributions[n])
            self.unit_availability[n] = [True] * num_units
            self.unit_rank[n] = list(range(num_units))
            self.unit_return_time[n] = [-1] * num_units
            self.unit_usage_count[n] = [0] * num_units
            self.inventory_levels[n] = inventory_levels[n % level_count]  # Cyclic assignment of inventory levels

    def g(self, x):
        return np.exp(-x)
    
    def update_availability(self, t):
        for i in range(self.N):
            for k in range(len(self.unit_availability[i])):
                if t >= self.unit_return_time[i][k]:
                    self.unit_availability[i][k] = True

    def allocate_resource(self, t):
        self.update_availability(t)
        scores = {}

        for i in self.E[t]:
            available_units = [k for k, available in enumerate(self.unit_availability[i]) 
                               if available and self.unit_usage_count[i][k] < self.inventory_levels[i]]
            if available_units:
                highest_ranked_unit = max(available_units, key=lambda x: self.unit_rank[i][x])
                score = self.rewards[i] * (1 - self.g(self.unit_rank[i][highest_ranked_unit] / len(self.unit_rank[i])))
                scores[i] = score

        if scores:
            selected_resource = max(scores, key=scores.get)
            selected_unit = max([k for k, available in enumerate(self.unit_availability[selected_resource]) 
                                 if available], key=lambda x: self.unit_rank[selected_resource][x])
            self.unit_availability[selected_resource][selected_unit] = False
            duration = np.random.choice(self.usage_duration_distributions[selected_resource])
            self.unit_return_time[selected_resource][selected_unit] = t + duration
            self.unit_usage_count[selected_resource][selected_unit] += 1  # Increment usage count
            return selected_resource, selected_unit

        return None, None

class InventoryBalancingAlgorithm:
    def __init__(self, N, T, E, rewards, usage_duration_distributions, inventory_levels):
        self.N = N  # Number of offline vertices (resources)
        self.T = T  # Number of online vertices (requests)
        self.E = E  # Edges in the bipartite graph
        self.rewards = rewards  # Reward for each resource
        self.usage_duration_distributions = usage_duration_distributions  # Usage duration distributions
        self.inventory_levels = inventory_levels  # Maximum allocations allowed for each resource
        
        # Initialize inventory, return times, and usage count for each resource
        self.inventory = [1] * self.N  # All resources start as available
        self.return_times = [[] for _ in range(self.N)]   # Return times for each resource
        self.usage_count = [0] * self.N  # Usage count for each resource

    def g(self, x):
        # Adjustment function used to modify resource selection priority
        return np.exp(-x)

    def update_inventory(self, current_time):
        # Updates the availability of resources based on their return times and usage limits
        for n in range(self.N):
            self.return_times[n] = [time for time in self.return_times[n] if time > current_time]
            self.inventory[n] = 1 if len(self.return_times[n]) == 0 and self.usage_count[n] < self.inventory_levels[n] else 0

    def allocate_resource(self, t):
        # Update inventory based on the current time
        self.update_inventory(t)
        
        # Calculate scores for each available resource at time t
        scores = {}
        for n in self.E[t]:
            if self.inventory[n]:  # Only consider available resources
                score = self.rewards[n] * (1 - self.g(self.inventory[n]))
                scores[n] = score
        
        if scores:
            # Select the resource with the highest score
            selected_resource = max(scores, key=scores.get)
            
            # Determine the duration the resource will be unavailable
            duration = np.random.choice(self.usage_duration_distributions[selected_resource])
            self.return_times[selected_resource].append(t + duration)
            
            # Increment usage count and mark the resource as unavailable if limit reached
            self.usage_count[selected_resource] += 1
            if self.usage_count[selected_resource] >= self.inventory_levels[selected_resource]:
                self.inventory[selected_resource] = 0  # Mark as permanently unavailable if max usage reached
            
            return (selected_resource, None)
        else:
            # If no resources are available, return None
            return None


    def __init__(self, N, T, E, rewards, usage_duration_distributions, inventory_levels):
        self.N = N  # Number of offline vertices (resources)
        self.T = T  # Number of online vertices (requests)
        self.E = E  # Edges in the bipartite graph
        self.rewards = rewards  # Reward for each resource
        self.usage_duration_distributions = usage_duration_distributions  # Usage duration distributions
        self.inventory_levels = inventory_levels  # Maximum allocations allowed for each resource
        
        # Initialize inventory, return times, and usage count for each resource
        self.inventory = [1] * self.N  # All resources start as available
        self.return_times = [[] for _ in range(self.N)]  # Return times for each resource
        self.usage_count = [0] * self.N  # Usage count for each resource

        # Initialize resource availability and return times
        self.unit_availability = [True] * self.N
        self.unit_return_time = [-1] * self.N  # -1 indicates the resource is available

    def update_availability(self, t):
        # Update the availability of resources based on the current time
        for i in range(self.N):
            if self.unit_return_time[i] <= t:
                self.unit_availability[i] = True

    def allocate_resource(self, t):
        # Update resource availability at the current time step
        self.update_availability(t)
        
        # Determine available resources and their corresponding rewards
        available_rewards = [(i, self.rewards[i]) for i in self.E[t] if self.unit_availability[i]]
        
        # Proceed with allocation if there are available resources
        if available_rewards:
            # Select the resource with the highest reward
            selected_resource, _ = max(available_rewards, key=lambda x: x[1])
            selected_unit = None  # In this simpler model, we are not distinguishing between units
            
            # Mark the selected resource as unavailable
            self.unit_availability[selected_resource] = False
            # Set the return time based on the usage duration
            duration = np.random.choice(self.usage_duration_distributions[selected_resource])
            self.unit_return_time[selected_resource] = t + duration
            
            return (selected_resource, selected_unit)
        else:
            return None  # No resources are available to allocate
        
class GreedyAlgorithm:
    def __init__(self, N, T, E, rewards, usage_duration_distributions, inventory_levels):
        self.N = N  # Number of resources
        self.T = T  # Number of time steps or requests
        self.E = E  # Edges representing possible matches between resources and requests
        self.rewards = rewards  # Rewards for allocating each resource
        self.usage_duration_distributions = usage_duration_distributions  # Usage durations for each resource
        self.inventory_levels = inventory_levels  # Maximum allocation counts for each resource
        
        # Initialize resource availability, return times, and usage counts
        self.unit_availability = [True] * self.N
        self.unit_return_time = [-1] * self.N  # -1 indicates the resource is available
        self.unit_usage_count = [0] * self.N  # Track how many times each resource has been allocated

    def update_availability(self, t):
        # Update the availability of resources based on the current time
        for i in range(self.N):
            if self.unit_return_time[i] <= t:
                self.unit_availability[i] = True

    def allocate_resource(self, t):
        # Update resource availability at the current time step
        self.update_availability(t)
        
        # Determine available resources that have not exceeded their inventory limits
        available_rewards = [(i, self.rewards[i]) for i in self.E[t] 
                             if self.unit_availability[i] and self.unit_usage_count[i] < self.inventory_levels[i]]
        
        # Proceed with allocation if there are available resources
        if available_rewards:
            # Select the resource with the highest reward
            selected_resource, _ = max(available_rewards, key=lambda x: x[1])
            
            # Mark the selected resource as unavailable and increment its usage count
            self.unit_availability[selected_resource] = False
            self.unit_usage_count[selected_resource] += 1  # Increment the usage count for the resource
            
            # Set the return time based on the usage duration
            duration = np.random.choice(self.usage_duration_distributions[selected_resource])
            self.unit_return_time[selected_resource] = t + duration
            
            return (selected_resource, None)  # There are no distinct units in this model
        else:
            return None  # No resources are available to allocate
