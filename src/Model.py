from mesa import DataCollector, Model
from mesa.discrete_space import OrthogonalMooreGrid
from Agent import CreatureAgent
from CellAgent import EnvCell

class Environment(Model):
    def __init__(self, width=60, height=60, num_agents=50, 
                 n_food_clusters=12, food_area_percentage=0.15, 
                 food_base_quantity=1, pheromone_decay_rate=0.02,
                 safety_buffer_steps=2, foraging_start_threshold=0.9,
                 pheromone_memory_weight=0.1,
                 pheromone_base_drop=1.0, pheromone_follow_prob=0.7,
                 rng=None): #RNG default for reproducibility
        
        super().__init__()
        
        if rng is not None:
            self.random = rng
            
        # Grid and Agent Setup
        self.grid = OrthogonalMooreGrid([width, height], torus=False, capacity=100, random=self.random)
        self.num_agents = num_agents
        self.pheromone_decay_rate = pheromone_decay_rate
        self.safety_buffer_steps = safety_buffer_steps
        self.foraging_start_threshold = foraging_start_threshold
        self.pheromone_memory_weight = pheromone_memory_weight
        self.pheromone_base_drop = pheromone_base_drop
        self.pheromone_follow_prob = pheromone_follow_prob
        self.nest_coords = (width // 2, height // 2)
        
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

        for cell in self.grid:
            cell_agent = EnvCell(self, len(self.agents), cell)
            
            if cell.coordinate == self.nest_coords:
                cell_agent.is_nest = True
            
            self.agents.add(cell_agent)

        # Foood distribution in clusters
        total_cells = width * height
        target_food_cells = int(total_cells * food_area_percentage)
        cells_per_cluster = max(1, target_food_cells // n_food_clusters)
        nest_food_free_radius = 1
        
        for _ in range(n_food_clusters):
            valid_start = False

            while not valid_start:
                cx = self.random.randint(0, width - 1)
                cy = self.random.randint(0, height - 1)
                dist_to_nest = max(abs(cx - self.nest_coords[0]), abs(cy - self.nest_coords[1]))
                
                if dist_to_nest > nest_food_free_radius:
                    valid_start = True
                    
            current_cell = self.grid[cx, cy]
            unique_cells_placed = 0 
            
            while unique_cells_placed < cells_per_cluster:
                env_agent = self._get_env_cell(current_cell)
                curr_x, curr_y = current_cell.coordinate
                dist = max(abs(curr_x - self.nest_coords[0]), abs(curr_y - self.nest_coords[1]))
                
                if env_agent and dist > nest_food_free_radius:
    
                    if env_agent.food_quantity == 0:
                        unique_cells_placed += 1

                    env_agent.food_quantity += food_base_quantity
                
                neighbors = current_cell.get_neighborhood(radius=1, include_center=False)
                
                if len(neighbors) > 0:
                    current_cell = neighbors.select_random_cell()

        # Closest food cell distance from the nest tracker
        food_cells = [c for c in self.agents_by_type.get(EnvCell, []) if c.food_quantity > 0]
        if food_cells:
            self.nearest_food_distance = min(
                max(abs(cell.cell.coordinate[0] - self.nest_coords[0]), abs(cell.cell.coordinate[1] - self.nest_coords[1]))
                for cell in food_cells
            )
        else:
            self.nearest_food_distance = None

        # Initialize Agents at the nest
        nest_cell = self.grid[self.nest_coords[0], self.nest_coords[1]]
        for _ in range(self.num_agents):
            agent = CreatureAgent(self, len(self.agents), nest_cell)
            self.agents.add(agent)

        self.initial_food = sum(c.food_quantity for c in self.agents_by_type.get(EnvCell, []))

        # Data Collector
        self.datacollector = DataCollector(
            model_reporters={
                "Foraging": lambda m: sum(1 for a in m.agents_by_type.get(CreatureAgent, []) if a.state == "FORAGING"),
                "Returning": lambda m: sum(1 for a in m.agents_by_type.get(CreatureAgent, []) if a.state == "RETURNING"),
                "Resting": lambda m: sum(1 for a in m.agents_by_type.get(CreatureAgent, []) if a.state == "RESTING"),
                "Alive": lambda m: len(m.agents_by_type.get(CreatureAgent, [])),
                "Dead (Energy)": lambda m: m.deaths_energy,
                "Dead (Temperature)": lambda m: m.deaths_temperature,
                "Deaths Foraging (Energy)": lambda m: m.deaths_foraging_energy,
                "Deaths Foraging (Temperature)": lambda m: m.deaths_foraging_temperature,
                "Deaths Returning (Energy)": lambda m: m.deaths_returning_energy,
                "Deaths Returning (Temperature)": lambda m: m.deaths_returning_temperature,
                "Deaths Resting (Energy)": lambda m: m.deaths_resting_energy,
                "Deaths Resting (Temperature)": lambda m: m.deaths_resting_temperature,
                "Remaining Food (%)": lambda m: (sum(c.food_quantity for c in m.agents_by_type.get(EnvCell, [])) / max(1, m.initial_food)) * 100
            }
        )
        self.datacollector.collect(self)
        self.running = True

    def _get_env_cell(self, target_cell):
        for obj in target_cell.agents:
            if isinstance(obj, EnvCell):
                return obj
        return None

    def step(self):
        # Move creatures
        if CreatureAgent in self.agents_by_type:
            self.agents_by_type[CreatureAgent].shuffle().do('step')
            
        # Evaporate pheromones on cells
        if EnvCell in self.agents_by_type:
            self.agents_by_type[EnvCell].shuffle().do('step')
            
        self.datacollector.collect(self)

        # End simulation if all agents are dead
        if len(self.agents_by_type.get(CreatureAgent, [])) == 0:
            self.running = False