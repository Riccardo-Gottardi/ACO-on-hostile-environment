from mesa.discrete_space import CellAgent

class EnvCell(CellAgent):
    def __init__(self, model, unique_id, cell):
        super().__init__(model)
        self.unique_id = unique_id
        self.cell = cell
        self.is_nest = False
        self.food_quantity = 0      
        self.pheromone_level = 0.0

    def step(self):
        
        if self.pheromone_level > 0:
            self.pheromone_level *= (1.0 - self.model.pheromone_decay_rate)

            # Residuals cleanup (for optimal performance)
            if self.pheromone_level < 0.001:
                self.pheromone_level = 0.0