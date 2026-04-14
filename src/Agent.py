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

        if self.is_in_nest():
            self.cool_down()
        else:
            self.heat_up()

        if self.is_on_food():
            self.consume_food()

        self.base_tirement()
        self.move()

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
        candidate_cells = list(
            # We explicitly ignore the error about self.cell being potentially None
            # because we are sure that it will not, since it is a parameter of it's 
            # constructor __init__(), that will be set by the Model during Creatures
            # initialisation
            self.cell.get_neighborhood(radius=Creature.MAX_STEP_SIZE, include_center=True) # pyright: ignore[reportOptionalMemberAccess]
        )

        new_cell = self.random.choice(candidate_cells)

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

    def is_in_danger(self):
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

    def can_return_home(self, pos: tuple[int, int] | None = None) -> bool:
        (nx, ny) = self.model.nest_position
        if pos is None:
            (cx, cy) = self.cell.coordinate # pyright: ignore[reportOptionalMemberAccess]
        else:
            (cx, cy) = pos
        
        distance_from_home = math.sqrt(math.pow(nx - cx, 2) + math.pow(ny - cy, 2))

        # TODO: whoud we use the following formula instead?
        # steps_to_go_home = math.ceil(distance_from_home / Creature.MAX_STEP_SIZE)
        step_to_go_home3 = distance_from_home // 3
        step_to_go_home2 = (distance_from_home % 3) // 2
        step_to_go_home1 = distance_from_home - step_to_go_home3 * 3 - step_to_go_home2 * 2
        step_to_go_home = step_to_go_home3 + step_to_go_home2 + step_to_go_home1
        
        temperature_rise_going_home = step_to_go_home * Creature.T_RISE_RATE
        energy_decrease_going_home = step_to_go_home * (Creature.E_BASE_DRAIN_RATE + Creature.E_MOVEMENT_DRAIN_RATE)
        return (self.temperature + temperature_rise_going_home < Creature.T_CRIT and 
                self.energy - energy_decrease_going_home > Creature.E_RANGE[0])
