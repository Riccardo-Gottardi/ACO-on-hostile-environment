from enum import Enum
from mesa.discrete_space import CellAgent, Cell
from mesa import Model
import math

class State(Enum):
    FORAGING = "foraging"
    RETURNING = "returning"
    RESTING = "resting"

class Creature(CellAgent):
    E_RANGE: tuple[int, int] = (0, 100)
    E_BASE_DRAIN_RATE: float = 0.1
    E_MOVEMENT_DRAIN_RATE: float = 0.3
    E_RESTORE_RATE: float = E_BASE_DRAIN_RATE
    E_THRESHOLD: float = 50.0

    T_RANGE: tuple[int, int] = (-10, 100)
    T_SAFE: int = 20
    T_CRIT: int = 55
    T_RISE_RATE: float = 0.5
    T_COOL_RATE: float = T_RISE_RATE

    MAX_STEP_SIZE: int = 3
    RISK_FACTOR: float = 0.3

    def __init__(self, model: Model, cell: Cell):
        super().__init__(model)

        self.cell = cell

        self.energy: float = Creature.E_RANGE[1]
        self.temperature: float = Creature.T_SAFE

        self.state = State.FORAGING

    def step(self):
        if self.is_dead():
            return

        # Creature action on Environment
        match self.state:
            case State.FORAGING:
                self.move()
                if self.is_on_food():
                    self.consume_food()
                if self.is_in_danger():
                    if self.can_return_home():
                        self.state = State.RETURNING

            case State.RETURNING:
                self.move()
                if self.is_in_nest():
                    self.state = State.RESTING
                    
            case State.RESTING:
                if self.is_in_danger():
                    self.state = State.FORAGING
    
        # Environment action on Creature
        match self.state:
            case State.FORAGING:
                self.heat_up()
            
            case State.RETURNING:
                self.heat_up()

            case State.RESTING:
                self.cool_down()

        self.base_tirement()

    def is_dead(self) -> bool:
        return self.energy <= 0 or self.temperature >= Creature.T_CRIT

    def is_in_nest(self) -> bool:
        return self.cell.coordinate == self.model.nest_position # pyright: ignore[reportOptionalMemberAccess]

    def cool_down(self):
        self.temperature = max(Creature.T_SAFE, self.temperature - Creature.T_COOL_RATE)

    def heat_up(self):
        self.temperature += Creature.T_RISE_RATE

    def base_tirement(self):
        self.energy -= Creature.E_BASE_DRAIN_RATE

    def move(self):
        match self.state:
            case State.FORAGING:
                self.move_logic_foraging()
            case State.RETURNING:
                self.move_logic_returning()

    def move_logic_foraging(self):
        neighbors_r1 = self.cell.get_neighborhood(radius=1, include_center=False)
        food_cells = [c for c in neighbors_r1 if self.model.food_grid[c.coordinate] > 0]
        food_cells.sort(key=lambda c: self.model.food_grid[c.coordinate], reverse=True)

        if food_cells:
            self.move_to(food_cells[0])
        else:
            # TODO: Remove random step and move toward higher pheromone concentration also do something about radius 2
            candidate_cells = self.cell.get_neighborhood(radius=2, include_center=True)
            self.move_to(self.random.choice(candidate_cells))

    def move_logic_returning(self):
        distance_to_nest = self.distance_from_nest()
        (nx, ny) = self.model.nest_position
        (cx, cy) = self.cell.coordinate
        distance_vector = (nx - cx, ny - cy)
        norm_distance_vector = (distance_vector[0] / distance_to_nest, distance_vector[1] / distance_to_nest)

        step_size = min(Creature.MAX_STEP_SIZE, distance_to_nest)

        next_cell = (math.ceil(cx + norm_distance_vector[0] * step_size), math.ceil(cy + norm_distance_vector[1] * step_size))

        self.move_to(next_cell)
    

    def move_to(self, new_cell: Cell):
        if new_cell is not self.cell:
            self.movement_tirement()
            self.cell = new_cell

    def movement_tirement(self):
        self.energy -= Creature.E_MOVEMENT_DRAIN_RATE

    def is_on_food(self) -> bool:
        if self.cell is not None:
            return self.model.food_grid[self.cell.coordinate] > 0
        else:
            return False

    def consume_food(self):
        if self.is_on_food():
            self.model.food_grid[self.cell.coordinate] -= 1  # pyright: ignore[reportOptionalMemberAccess]
            self.energy = min(Creature.E_RANGE[1], self.energy + Creature.E_RESTORE_RATE)
    
    def is_in_danger(self) -> bool:
        if self.state == State.FORAGING:
            return self.is_in_foraging_danger()
        elif self.state == State.RESTING:
            return not self.is_in_resting_danger()
        else:
            return False

    def is_in_foraging_danger(self):
        candidates_next_cell = list(
            self.cell.get_neighborhood(radius=Creature.MAX_STEP_SIZE, include_center=True) # pyright: ignore[reportOptionalMemberAccess]
        )

        number_of_dangerous_cells = 0
        for cell in candidates_next_cell:
            pos = cell.coordinate
            if not self.can_return_home((pos[0], pos[1])):
                number_of_dangerous_cells += 1
        
        danger = number_of_dangerous_cells / len(candidates_next_cell)
        return danger > Creature.RISK_FACTOR

    def is_in_resting_danger(self) -> bool:
        thermal_gain_per_step = Creature.T_COOL_RATE / Creature.T_RISE_RATE
        energy_loss_ratio = Creature.E_BASE_DRAIN_RATE / (Creature.E_BASE_DRAIN_RATE + Creature.E_MOVEMENT_DRAIN_RATE)
        
        # Dynamic Scarcity Factor (The "Greed" Variable)
        # If the creature recently found a rich food cluster, 
        # it can afford to rest longer. If food is scarce, it must leave sooner.
        # richness_estimate should be a value from 0.0 to 1.0
        scarcity_buffer = getattr(self, 'richness_estimate', 0.5) 

        worth_it = (thermal_gain_per_step * scarcity_buffer) > energy_loss_ratio

        # Immediate Death Overrides 
        # Never leave if the next step results in T >= T_crit
        if self.temperature + Creature.T_RISE_RATE >= Creature.T_CRIT:
            return True 
        
        if self.temperature <= Creature.T_SAFE:
            return False

        return worth_it

    def can_return_home(self, pos: tuple[int, int] | None = None) -> bool:
        distance_from_home = self.distance_from_nest(pos)

        # TODO: whoud we use the following formula instead?
        # steps_to_go_home = math.ceil(distance_from_home / Creature.MAX_STEP_SIZE)
        # TODO: FIX the following formula to consider dynamically steps maximum size
        step_to_go_home3 = distance_from_home // 3
        step_to_go_home2 = (distance_from_home % 3) // 2
        step_to_go_home1 = distance_from_home - step_to_go_home3 * 3 - step_to_go_home2 * 2
        step_to_go_home = step_to_go_home3 + step_to_go_home2 + step_to_go_home1
        
        temperature_rise_going_home = step_to_go_home * Creature.T_RISE_RATE
        energy_decrease_going_home = step_to_go_home * (Creature.E_BASE_DRAIN_RATE + Creature.E_MOVEMENT_DRAIN_RATE)
        return (self.temperature + temperature_rise_going_home < Creature.T_CRIT and 
                self.energy - energy_decrease_going_home > Creature.E_RANGE[0])

    def distance_from_nest(self, pos: tuple[int, int] | None = None) -> int:
        (nx, ny) = self.model.nest_position
        if pos is None:
            (cx, cy) = self.cell.coordinate
        else:
            (cx, cy) = pos
        return math.sqrt(math.pow(nx - cx, 2) + math.pow(ny - cy, 2))
