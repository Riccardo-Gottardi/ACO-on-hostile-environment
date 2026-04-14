from mesa.discrete_space import CellAgent, Cell
from mesa import Model


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

    def __init__(self, model: Model, cell: Cell):
        super().__init__(model)

        self.cell = cell

        self.energy: float = Creature.E_RANGE[1]
        self.temperature: float = Creature.T_SAFE

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
        return self.cell.coordinate == self.model.nest_position

    def cool_down(self):
        self.temperature = max(Creature.T_SAFE, self.temperature - Creature.T_COOL_RATE)

    def heat_up(self):
        self.temperature += Creature.T_RISE_RATE

    def base_tirement(self):
        self.energy -= Creature.E_BASE_DRAIN_RATE

    def move(self):
        candidate_cells = list(
            self.cell.get_neighborhood(radius=Creature.MAX_STEP_SIZE, include_center=True)
        )

        new_cell = self.random.choice(candidate_cells)

        if new_cell is not self.cell:
            self.movement_tirement()
            self.cell = new_cell

    def movement_tirement(self):
        self.energy -= Creature.E_MOVEMENT_DRAIN_RATE

    def is_on_food(self) -> bool:
        return self.model.food_grid[self.cell.coordinate] > 0

    def consume_food(self):
        if self.is_on_food():
            self.model.food_grid[self.cell.coordinate] -= 1
            self.energy = min(Creature.E_RANGE[1], self.energy + Creature.E_RESTORE_RATE)