from mesa.model import Agent, Model


class Creature(Agent):
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

    def __init__(self, position: tuple[int, int], unique_id: int, model: Model):
        super().__init__(model)
        
        self.id = unique_id
        self.energy: float = Creature.E_RANGE[1]
        self.temperature: float = Creature.T_SAFE

    def step(self):
        if self.is_dead():
            return
        
        if self.is_in_nest():
            self.cool_down()
        else:
            self.heat_up()
        
        self.base_tirement()

        self.move()

    def is_dead(self) -> bool:
        if self.energy <= 0 or self.temperature >= Creature.T_CRIT:
            return True
        else:
            return False

    def is_in_nest(self) -> bool:
        if self.pos == self.model.nest_position:
            return True
        else:
            return False

    def cool_down(self):
        self.temperature = max(Creature.T_SAFE, self.temperature - Creature.T_COOL_RATE)

    def heat_up(self):
        self.temperature += Creature.T_RISE_RATE

    def base_tirement(self):
        self.energy -= Creature.E_BASE_DRAIN_RATE

    def move(self):
        candidate_steps = self.model.grid.get_neighborhood(
           self.pos, True, True, Creature.MAX_STEP_SIZE 
        )

        new_position = self.random.choice(candidate_steps)

        if self.pos != new_position:
            self.movement_tirement()
            self.model.grid.move_agent(self, new_position)

    def movement_tirement(self):
        self.energy -= Creature.E_MOVEMENT_DRAIN_RATE