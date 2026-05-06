import math
import numpy as np
from mesa import DataCollector, Model
from mesa.discrete_space import OrthogonalMooreGrid, PropertyLayer
from Agent import CreatureAgent
from Agent import State

class Environment(Model):
    def __init__(self, width=60, height=60, num_agents=50, 
                 n_food_clusters=12, food_area_percentage=0.15, 
                 food_base_quantity=10, pheromone_decay_rate=0.02,
                 safety_buffer_steps=2, foraging_start_threshold=0.9,
                 pheromone_memory_weight=0.1,
                 pheromone_base_drop=1.0, pheromone_follow_prob=0.7,
                 seed=None): #RNG default for reproducibility
        
        super().__init__(seed=seed)
            
        # Grid and Agent Setup
        self.grid = OrthogonalMooreGrid((width, height), torus=False, capacity=100, random=self.random)
        self.food_layer = PropertyLayer("food", (width, height), default_value=0.0, dtype=float)
        self.pheromone_layer = PropertyLayer("pheromone", (width, height), default_value=0.0, dtype=float)
        self.grid.add_property_layer(self.food_layer)
        self.grid.add_property_layer(self.pheromone_layer)

        self.num_agents = num_agents
        self.pheromone_decay_rate = pheromone_decay_rate
        self.safety_buffer_steps = safety_buffer_steps
        self.foraging_start_threshold = foraging_start_threshold
        self.pheromone_memory_weight = pheromone_memory_weight
        self.pheromone_base_drop = pheromone_base_drop
        self.pheromone_follow_prob = pheromone_follow_prob
        self.nest_coords = (width // 2, height // 2)
        self.nest_layer = PropertyLayer("nest", (width, height), default_value=0.0, dtype=float)
        self.nest_layer.data[self.nest_coords] = 1.0
        self.grid.add_property_layer(self.nest_layer)
        self.total_grid_cells = width * height
        self.steps_elapsed = 0
        self.cumulative_thermal_load = 0.0
        self.completed_agent_food_collections = []

        self.deaths_energy = 0
        self.deaths_temperature = 0
        self.deaths_foraging = 0
        self.deaths_returning = 0
        self.deaths_resting = 0
        self.deaths_foraging_energy = 0
        self.deaths_foraging_temperature = 0
        self.deaths_returning_energy = 0
        self.deaths_returning_temperature = 0
        self.deaths_resting_energy = 0
        self.deaths_resting_temperature = 0

        total_cells = width * height
        target_food_cells = int(total_cells * food_area_percentage)
        cells_per_cluster = max(1, target_food_cells // n_food_clusters)
        nest_food_free_radius = 1

        for _ in range(n_food_clusters):
            valid_start = False
            cx = cy = 0
            while not valid_start:
                cx = self.random.randint(0, width - 1)
                cy = self.random.randint(0, height - 1)
                dist_to_nest = max(abs(cx - self.nest_coords[0]), abs(cy - self.nest_coords[1]))

                if dist_to_nest > nest_food_free_radius:
                    valid_start = True

            current_cell = self.grid[cx, cy]
            unique_cells_placed = 0

            while unique_cells_placed < cells_per_cluster:
                curr_x, curr_y = current_cell.coordinate
                dist = max(abs(curr_x - self.nest_coords[0]), abs(curr_y - self.nest_coords[1]))

                if dist > nest_food_free_radius:
                    if self.food_layer.data[curr_x, curr_y] <= 0:
                        unique_cells_placed += 1

                    self.food_layer.data[curr_x, curr_y] += food_base_quantity

                neighbors = current_cell.get_neighborhood(radius=1, include_center=False)

                if len(neighbors) > 0:
                    current_cell = neighbors.select_random_cell()

        food_positions = np.argwhere(self.food_layer.data > 0)
        if len(food_positions) > 0:
            self.nearest_food_distance = min(
                max(abs(int(x) - self.nest_coords[0]), abs(int(y) - self.nest_coords[1]))
                for x, y in food_positions
            )
        else:
            self.nearest_food_distance = None

        nest_cell = self.grid[self.nest_coords[0], self.nest_coords[1]]
        for _ in range(self.num_agents):
            CreatureAgent(self, nest_cell)

        self.initial_food = float(self.food_layer.data.sum())

        # Data Collector
        self.datacollector = DataCollector(
            model_reporters={
                "Foraging": lambda m: sum(1 for a in m.agents_by_type[CreatureAgent] if a.state == State.FORAGING),
                "Returning": lambda m: sum(1 for a in m.agents_by_type[CreatureAgent] if a.state == State.RETURNING),
                "Resting": lambda m: sum(1 for a in m.agents_by_type[CreatureAgent] if a.state == State.RESTING),
                "Alive": lambda m: len(m.agents_by_type[CreatureAgent]),
                "Dead (Energy)": lambda m: m.deaths_energy,
                "Dead (Temperature)": lambda m: m.deaths_temperature,
                "Deaths Foraging (Energy)": lambda m: m.deaths_foraging_energy,
                "Deaths Foraging (Temperature)": lambda m: m.deaths_foraging_temperature,
                "Deaths Returning (Energy)": lambda m: m.deaths_returning_energy,
                "Deaths Returning (Temperature)": lambda m: m.deaths_returning_temperature,
                "Deaths Resting (Energy)": lambda m: m.deaths_resting_energy,
                "Deaths Resting (Temperature)": lambda m: m.deaths_resting_temperature,
                "Remaining Food (Units)": lambda m: m.get_remaining_food_units(),
                "Remaining Food (%)": lambda m: (m.get_remaining_food_units() / max(1, m.initial_food)) * 100,
                "Total Food Collected": lambda m: m.total_food_collected,
                "Resource Retrieval Rate": lambda m: m.calculate_resource_retrieval_rate(),
                "Load Gini": lambda m: m.calculate_load_gini(),
                "Cumulative Thermal Load": lambda m: m.cumulative_thermal_load,
                "Shannon Entropy": lambda m: m.calculate_spatial_entropy(),
                "Thermal Efficiency": lambda m: m.calculate_thermal_efficiency(),
                "Mean Agent Energy": lambda m: m.calculate_mean_agent_energy(),
                "Mean Agent Temperature": lambda m: m.calculate_mean_agent_temperature(),
                "Mean Distance to Nest": lambda m: m.calculate_mean_distance_to_nest(),
                "Mean Lifetime Food Collected": lambda m: m.calculate_mean_lifetime_food_collected(),
            },
            # CHANGE HERE: Use agent_reporters directly instead of agenttype_reporters
            agent_reporters={
                "Energy": lambda a: a.energy,
                "Temperature": lambda a: a.temperature,
                "State": lambda a: a.state,
                "Distance_to_Nest": lambda a: max(
                    abs(a.cell.coordinate[0] - a.model.nest_coords[0]),
                    abs(a.cell.coordinate[1] - a.model.nest_coords[1]),
                ),
                "Age": lambda a: a.age_steps,
                "Lifetime Food Collected": lambda a: a.lifetime_food_collected,
                "Has Food": lambda a: a.has_food,
            }
        )
        self.datacollector.collect(self)
        self.running = True

    def get_remaining_food_units(self):
        return float(self.food_layer.data.sum())

    @property
    def total_food_collected(self):
        return max(0, self.initial_food - self.get_remaining_food_units())

    def _all_agent_food_collections(self):
        active_collections = [
            agent.lifetime_food_collected
            for agent in self.agents_by_type[CreatureAgent]
        ]
        return [*self.completed_agent_food_collections, *active_collections]

    def _gini_coefficient(self, values):
        filtered_values = [float(value) for value in values if value is not None]
        if not filtered_values:
            return 0.0

        total = sum(filtered_values)
        if total <= 0:
            return 0.0

        sorted_values = sorted(value for value in filtered_values if value >= 0)
        if not sorted_values:
            return 0.0

        n = len(sorted_values)
        weighted_sum = sum(index * value for index, value in enumerate(sorted_values, start=1))
        gini = (2.0 * weighted_sum) / (n * total) - (n + 1.0) / n
        return max(0.0, min(1.0, gini))

    def _update_thermal_load(self):
        active_agents = self.agents_by_type[CreatureAgent]
        step_heat_load = sum(max(0.0, agent.temperature - agent.T_safe) for agent in active_agents)
        self.cumulative_thermal_load += step_heat_load

    def calculate_resource_retrieval_rate(self):
        if self.steps_elapsed <= 0:
            return 0.0

        return self.total_food_collected / self.steps_elapsed

    def calculate_spatial_entropy(self):
        agents = self.agents_by_type[CreatureAgent]
        if len(agents) <= 1:
            return 0.0

        position_counts = {}
        for agent in agents:
            coordinate = agent.cell.coordinate
            position_counts[coordinate] = position_counts.get(coordinate, 0) + 1

        total_agents = len(agents)
        probabilities = [count / total_agents for count in position_counts.values()]
        entropy = -sum(probability * math.log2(probability) for probability in probabilities if probability > 0)

        max_entropy = math.log2(max(1, min(total_agents, self.total_grid_cells)))
        if max_entropy <= 0:
            return 0.0

        return entropy / max_entropy

    def calculate_thermal_efficiency(self):
        if self.cumulative_thermal_load <= 0:
            return 0.0

        return self.total_food_collected / self.cumulative_thermal_load

    def calculate_load_gini(self):
        return self._gini_coefficient(self._all_agent_food_collections())

    def calculate_mean_agent_energy(self):
        agents = self.agents_by_type[CreatureAgent]
        if not agents:
            return 0.0

        return sum(agent.energy for agent in agents) / len(agents)

    def calculate_mean_agent_temperature(self):
        agents = self.agents_by_type[CreatureAgent]
        if not agents:
            return 0.0

        return sum(agent.temperature for agent in agents) / len(agents)

    def calculate_mean_distance_to_nest(self):
        agents = self.agents_by_type[CreatureAgent]
        if not agents:
            return 0.0

        return sum(
            max(
                abs(agent.cell.coordinate[0] - self.nest_coords[0]),
                abs(agent.cell.coordinate[1] - self.nest_coords[1]),
            )
            for agent in agents
        ) / len(agents)

    def calculate_mean_lifetime_food_collected(self):
        all_collections = self._all_agent_food_collections()
        if not all_collections:
            return 0.0

        return sum(all_collections) / len(all_collections)

    def step(self):
        self.agents_by_type[CreatureAgent].shuffle_do("step")

        self.pheromone_layer.data *= (1.0 - self.pheromone_decay_rate)
        self._update_thermal_load()
        self.steps_elapsed += 1

        self.datacollector.collect(self)

        if len(self.agents_by_type[CreatureAgent]) == 0:
            self.running = False