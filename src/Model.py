from mesa.discrete_space import OrthogonalMooreGrid
from mesa import Model
import numpy as np


class Environment(Model):
    def __init__(self, n_food_clusters: int = 12, food_area_percentage: float = 0.15, food_base_quantity: int = 10, pheromone_decay_rate: float = 0.05):
        super().__init__()

        self.grid = OrthogonalMooreGrid((60, 60), False, None, self.random)

        self.nest_position = (
            self.random.randint(0, self.grid.width - 1),
            self.random.randint(0, self.grid.height - 1)
        )

        self.pheromone_grid = np.zeros((self.grid.width, self.grid.height))
        self.pheromone_decay_rate = pheromone_decay_rate

        self.food_grid = np.zeros((self.grid.width, self.grid.height))
        self._generate_food_cluster(n_food_clusters, food_area_percentage, food_base_quantity)

    def _generate_food_cluster(self, n_food_clusters: int, food_area_percentage: float, food_base_quantity: int):
        total_cells = self.grid.width * self.grid.height
        target_food_cells = total_cells * food_area_percentage

        # Place the central cells of the food clusters
        food_cells_positions = []
        while len(food_cells_positions) < n_food_clusters:
            x = self.random.randint(0, self.grid.width - 1)
            y = self.random.randint(0, self.grid.height - 1)

            if (x, y) != self.nest_position and self.food_grid[(x, y)] == 0:
                self.food_grid[(x, y)] = food_base_quantity
                food_cells_positions.append((x, y))

        # Add the neighbourhood cells for the food clusters
        while len(food_cells_positions) < target_food_cells:
            (x, y) = self.random.choice(food_cells_positions)

            neighbor_cell = self.random.choice(
                list(self.grid[(x, y)].get_neighborhood(radius=1, include_center=False))
            )
            # Since we don't care anymore of the first (x, y) we overwrite them
            (x, y) = neighbor_cell.coordinate

            if (x, y) != self.nest_position and self.food_grid[(x, y)] == 0:
                self.food_grid[(x, y)] = food_base_quantity
                food_cells_positions.append((x, y))

    def step(self):
        # Update the pheromone traces
        self.pheromone_grid *= (1 - self.pheromone_decay_rate)

        # Update the agents in random order, to avoid agents order bias
        self.agents.shuffle_do("step")

        dead_creatures = [c for c in self.agents if c.is_dead()]
        for creature in dead_creatures:
            creature.remove()