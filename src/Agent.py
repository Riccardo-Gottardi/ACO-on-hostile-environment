import numpy as np
from mesa.discrete_space import CellAgent
from CellAgent import EnvCell

class CreatureAgent(CellAgent):
    def __init__(self, model, unique_id, cell):
        super().__init__(model)
        self.unique_id = unique_id
        self.cell = cell
        
        # State Variables
        self.state = "FORAGING"
        self.E_max = 1000.0          # Max energy 
        self.energy = self.E_max     # E_i(t) bounded by [0, E_max]
        self.temperature = 30.0      # T_i(t) start at safe nest temp
        
        # Physiology / Cost Constants
        self.T_safe = 30.0           # Nest temperature (approx. 30°C underground)
        self.T_crit = 53.0           # Critical thermal max (Saharan ant limit is ~53°C)
        self.cost_E_move = 2.0       # Energy cost per step moving
        self.cost_E_rest = 1.0       # Energy cost while resting/basal metabolism
        self.cost_T_env = 0.05       # Base Temp increase per minute (step) outside
        self.cost_T_move = 0.05      # Extra Temp increase per minute when moving
        self.cool_rate = 2.0         # Temp decrease per minute in nest
        self.max_speed = 2
        self.food_energy_gain = 750.0  # Energy gained per food item consumed
        
        # Memory
        self.food_richness_memory = 0.0
        self.has_food = False

    def _get_env_cell(self, target_cell):
        """Helper to extract the EnvCell object from a grid coordinate."""
        for obj in target_cell.agents:
            if isinstance(obj, EnvCell):
                return obj
        return None

    def _chebyshev_distance(self, coord1, coord2):
        return max(abs(coord1[0] - coord2[0]), abs(coord1[1] - coord2[1]))

    def is_dead(self):
        return self.energy <= 0 or self.temperature >= self.T_crit

    def is_in_nest(self):
        return self.cell.coordinate == self.model.nest_coords

    def base_tirement(self):
        self.energy -= self.cost_E_rest

    def movement_tirement(self):
        self.energy -= self.cost_E_move
        self.temperature += self.cost_T_move

    def cool_down(self):
        self.temperature = max(self.T_safe, self.temperature - self.cool_rate)

    def should_return_home(self):
        """Calculates if the agent is in danger of dying before reaching the nest."""
        dist_to_nest = self._chebyshev_distance(self.cell.coordinate, self.model.nest_coords)
        
        # 1. Calculate TRUE costs per step (movement + basal metabolism/environment)
        total_energy_per_step = self.cost_E_move + self.cost_E_rest
        total_heat_per_step = self.cost_T_move + self.cost_T_env
        
        # 2. Add a flat safety buffer
        safe_distance = dist_to_nest + self.model.safety_buffer_steps
        
        # 3. Heat danger: Will my temp exceed T_crit before I get back?
        estimated_max_temp = self.temperature + (safe_distance * total_heat_per_step)
        heat_danger = estimated_max_temp >= self.T_crit
        
        # 4. Energy danger: Will I run out of energy before I get back?
        required_energy = safe_distance * total_energy_per_step
        energy_danger = self.energy <= required_energy
        
        # 5. If in ANY danger, go home immediately. No gambling.
        return heat_danger or energy_danger

    def should_start_foraging(self):
        """Determines if a resting agent is ready to leave the nest."""
        if self.temperature <= self.T_safe:
            if self.energy < (self.E_max * self.model.foraging_start_threshold):
                return True
        return False

    def is_in_danger(self):
        """Wrapper method routing to specific state-based checks."""
        if self.state == "FORAGING":
            return self.should_return_home()
        elif self.state == "RESTING":
            return self.should_start_foraging()
        return False

    def can_return_home(self):
        return True 

    def is_on_food(self):
        neighbors = self.cell.get_neighborhood(radius=1, include_center=True)
        return any(self._get_env_cell(c).food_quantity > 0 for c in neighbors)

    def consume_food(self):
        neighbors = self.cell.get_neighborhood(radius=1, include_center=True)
        food_cells = [c for c in neighbors if self._get_env_cell(c).food_quantity > 0]
        if food_cells:
            target_cell = self.model.random.choice(food_cells)
            self.cell = target_cell
            env_cell = self._get_env_cell(self.cell)
            
            env_cell.food_quantity -= 1
            self.energy = min(self.E_max, self.energy + self.food_energy_gain)
            self.has_food = True
            self.food_richness_memory = env_cell.food_quantity

    def drop_pheromone(self):
        if self.has_food:
            env_cell = self._get_env_cell(self.cell)
            env_cell.pheromone_level += (self.food_richness_memory * self.model.pheromone_memory_weight) + self.model.pheromone_base_drop

    def move(self):
        match self.state:
            case "FORAGING":
                self.movement_tirement()
                self.move_logic_foraging()
            case "RETURNING":
                self.movement_tirement()
                self.move_logic_returning()

    def move_logic_foraging(self):
        if self.is_on_food():
            pass # Consumption handled directly
        else:
            neighbors = self.cell.get_neighborhood(radius=1, include_center=True)
            if self.model.random.random() < self.model.pheromone_follow_prob:
                best_cell = max(neighbors, key=lambda c: self._get_env_cell(c).pheromone_level)
                if self._get_env_cell(best_cell).pheromone_level > 0:
                    dx = best_cell.coordinate[0] - self.cell.coordinate[0]
                    dy = best_cell.coordinate[1] - self.cell.coordinate[1]
                    
                    tx = max(0, min(self.model.grid.width - 1, self.cell.coordinate[0] + dx * 2))
                    ty = max(0, min(self.model.grid.height - 1, self.cell.coordinate[1] + dy * 2))
                    
                    self.cell = self.model.grid[tx, ty]
                else:
                    self.cell = neighbors.select_random_cell()
            else:
                step_size = self.model.random.randint(1, self.max_speed)
                exploratory_neighbors = self.cell.get_neighborhood(radius=step_size, include_center=False)
                self.cell = exploratory_neighbors.select_random_cell()

    def move_logic_returning(self):
        if self.is_in_nest():
            return
        
        self.drop_pheromone()
        
        neighbors = self.cell.get_neighborhood(radius=1, include_center=False)
        best_cell = min(neighbors, key=lambda c: self._chebyshev_distance(c.coordinate, self.model.nest_coords))
        self.cell = best_cell

    def step(self):
        if self.is_dead():
            if self.state == "FORAGING":
                self.model.deaths_foraging += 1
                if self.energy <= 0:
                    self.model.deaths_foraging_energy += 1
                else:
                    self.model.deaths_foraging_temperature += 1
            elif self.state == "RETURNING":
                self.model.deaths_returning += 1
                if self.energy <= 0:
                    self.model.deaths_returning_energy += 1
                else:
                    self.model.deaths_returning_temperature += 1
            elif self.state == "RESTING":
                self.model.deaths_resting += 1
                if self.energy <= 0:
                    self.model.deaths_resting_energy += 1
                else:
                    self.model.deaths_resting_temperature += 1

            if self.energy <= 0:
                self.model.deaths_energy += 1
            else:
                self.model.deaths_temperature += 1
            self.remove()
            return
            
        self.energy -= self.cost_E_rest
        if not self.is_in_nest():
            self.temperature += self.cost_T_env
            
        match self.state:
            case "FORAGING":
                if self.is_on_food():
                    self.consume_food()
                    self.state = "RETURNING"
                elif self.is_in_danger():
                    self.state = "RETURNING"
                else:
                    self.move()

            case "RETURNING":
                if self.is_in_nest():
                    self.state = "RESTING"
                    self.has_food = False
                else:
                    self.move()
                    
            case "RESTING":
                self.cool_down()
                if self.is_in_danger():
                    self.state = "FORAGING"