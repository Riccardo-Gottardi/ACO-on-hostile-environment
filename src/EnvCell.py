from mesa.discrete_space import FixedAgent

class EnvCell(FixedAgent):
    PHEROMONE_EPSILON = 0.001

    def __init__(self, model, cell):
        super().__init__(model)
        self.cell = cell
        self.is_nest = False
        self.food_quantity = 0      
        self.pheromone_level = 0.0

    def step(self):
        if self.pheromone_level > 0:
            self.pheromone_level *= (1.0 - self.model.pheromone_decay_rate)

            # Residuals cleanup (for optimal performance)
            if self.pheromone_level < self.PHEROMONE_EPSILON:
                self.pheromone_level = 0.0