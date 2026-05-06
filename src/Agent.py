import math
from enum import Enum
from mesa.discrete_space import CellAgent

class ReturnReason(Enum):
    RETURN_FOOD = "FOOD"      # carrying food, slow return, dropping pheromone
    RETURN_DANGER = "DANGER"  # no food, panic-fast escape, no pheromone
    RETURN_BOTH = "BOTH"      # carrying food AND slow-return is now unsafe
    NONE = None

class State(Enum):
    FORAGING = "FORAGING"
    RETURNING = "RETURNING"
    RESTING = "RESTING"

class CreatureAgent(CellAgent):
    def __init__(self, model, cell):
        super().__init__(model)
        self.cell = cell

        # State variables
        self.state = State.FORAGING
        self.return_reason = None      # FOOD | DANGER | BOTH | None
        self.E_max = 1000.0
        self.energy = self.E_max
        self.temperature = 30.0
        self.age_steps = 0
        self.lifetime_food_collected = 0

        # Physiology / cost constants
        self.T_safe = 30.0
        self.T_crit = 53.0
        self.cost_E_rest = 1.0         # Per-tick basal metabolism
        self.cost_E_move = 2.0         # Per-cell additional cost when moving
        self.cost_T_env = 0.05         # Per-tick heat gain when outside (independent of distance moved)
        self.cool_rate = 2.0           # Per-tick heat loss when in nest
        self.max_speed = 2
        self.food_energy_gain = 750.0

        # Memory
        self.food_richness_memory = 0.0
        self.has_food = False

    # =========== Helper Functions ===========

    def _chebyshev_distance(self, c1, c2):
        return max(abs(c1[0] - c2[0]), abs(c1[1] - c2[1]))

    def _eucledian_distance(self, c1, c2):
        return math.sqrt((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2)

    def is_dead(self):
        return self.energy <= 0 or self.temperature >= self.T_crit

    def is_in_nest(self):
        return self.cell.coordinate == self.model.nest_coords

    def is_on_food(self):
        neighbors = self.cell.get_neighborhood(radius=1, include_center=True)
        food_layer = self.model.food_layer.data

        return any(food_layer[c.coordinate] > 0 for c in neighbors)

    # ---------- Danger projections ----------

    def _project_danger(self, n_ticks, n_cells):
        """For a hypothetical journey, return (energy_danger, heat_danger).

        Energy cost = basal per tick + movement per cell traversed.
        Heat gain   = per tick outside (does not scale with distance moved).
        """
        energy_needed = n_ticks * self.cost_E_rest + n_cells * self.cost_E_move
        heat_projected = self.temperature + n_ticks * self.cost_T_env

        return (self.energy <= energy_needed, heat_projected >= self.T_crit)

    def _project_danger_slow(self):
        """Slow return: 1 cell/tick. Models the FOOD/BOTH movement profile."""
        dist = self._chebyshev_distance(self.cell.coordinate, self.model.nest_coords)
        n_cells = dist + self.model.safety_buffer_steps
        n_ticks = n_cells

        return self._project_danger(n_ticks, n_cells)

    def _project_danger_fast(self):
        """Fast escape: max_speed cells/tick. Models the DANGER movement profile."""
        dist = self._chebyshev_distance(self.cell.coordinate, self.model.nest_coords)
        n_cells = dist + self.model.safety_buffer_steps
        n_ticks = math.ceil(n_cells / self.max_speed)

        return self._project_danger(n_ticks, n_cells)

    def _resolve_return_reason_after_slow_danger(self):
        """
        FOOD -> DANGER (alternative; uncomment block + comment line above to enable)
        When slow-projection fails, also check fast-projection. If fast escape
        (drop food, no pheromone, max_speed back) is still survivable, do that
        instead. Biases toward saving lives over sharing food info.
        """
        e_fast, h_fast = self._project_danger_fast()

        if not (e_fast or h_fast):
            self.has_food = False
            self.return_reason = RETURN_DANGER
        else:
            self.return_reason = RETURN_BOTH

    # ---------- Cost charging ----------

    def base_tirement(self):
        self.energy -= self.cost_E_rest

    def movement_tirement(self, n_cells):
        """Per-cell movement cost. Basal metabolism is charged separately at top of step()."""
        self.energy -= n_cells * self.cost_E_move

    def cool_down(self):
        self.temperature = max(self.T_safe, self.temperature - self.cool_rate)

    # ---------- Decision helpers ----------

    def should_start_foraging(self):
        if self.temperature <= self.T_safe or self.energy < (self.E_max * self.model.foraging_start_threshold):
            return True
        else:
            return False

    # ---------- Actions ----------

    def consume_food(self):
        neighbors = self.cell.get_neighborhood(radius=1, include_center=True)
        food_layer = self.model.food_layer.data
        food_cells = [c for c in neighbors if food_layer[c.coordinate] > 0]
        if food_cells:
            target_cell = self.model.random.choice(food_cells)
            self.cell = target_cell
            cx, cy = self.cell.coordinate
            food_layer[cx, cy] = max(0.0, food_layer[cx, cy] - 1.0)
            self.energy = min(self.E_max, self.energy + self.food_energy_gain)
            self.has_food = True
            self.food_richness_memory = food_layer[cx, cy]
            self.lifetime_food_collected += 1

    def drop_pheromone(self):
        if self.has_food:
            pheromone_layer = self.model.pheromone_layer.data
            cx, cy = self.cell.coordinate
            pheromone_layer[cx, cy] += (
                self.food_richness_memory * self.model.pheromone_memory_weight
                + self.model.pheromone_base_drop
            )

    # ---------- Movement ----------

    def move(self):
        match self.state:
            case State.FORAGING:
                self.move_logic_foraging()
            case State.RETURNING:
                self.move_logic_returning()

    def move_logic_foraging(self):
        if self.is_on_food():
            return  # standing still on food (the eat-and-transition happens in step())

        old_coord = self.cell.coordinate
        neighbors = self.cell.get_neighborhood(radius=1, include_center=False)
        pheromone_layer = self.model.pheromone_layer.data

        if self.model.random.random() < self.model.pheromone_follow_prob:
            best_cell = max(neighbors, key=lambda c: pheromone_layer[c.coordinate])
            if pheromone_layer[best_cell.coordinate] > 0:
                dx = best_cell.coordinate[0] - old_coord[0]
                dy = best_cell.coordinate[1] - old_coord[1]
                tx = max(0, min(self.model.grid.width - 1, old_coord[0] + dx * 2))
                ty = max(0, min(self.model.grid.height - 1, old_coord[1] + dy * 2))
                self.cell = self.model.grid[tx, ty]
            else:
                self.cell = neighbors.select_random_cell()
        else:
            step_size = self.model.random.randint(1, self.max_speed)
            exploratory_neighbors = self.cell.get_neighborhood(radius=step_size, include_center=False)
            self.cell = exploratory_neighbors.select_random_cell()

        n_cells = self._chebyshev_distance(old_coord, self.cell.coordinate)
        self.movement_tirement(n_cells)

    def move_logic_returning(self):
        if self.is_in_nest():
            return

        old_coord = self.cell.coordinate

        # Speed split:
        # - has_food=True (FOOD/BOTH): slow (1 cell/tick), drop pheromone
        # - has_food=False (DANGER):   fast (max_speed/tick), no pheromone
        if self.has_food:
            self.drop_pheromone()
            radius = 1
        else:
            radius = self.max_speed

        neighbors = self.cell.get_neighborhood(radius=radius, include_center=False)
        best_cell = min(
            neighbors,
            key=lambda c: (self._chebyshev_distance(c.coordinate, self.model.nest_coords) + self._eucledian_distance(c.coordinate, self.model.nest_coords)),
        )
        self.cell = best_cell

        n_cells = self._chebyshev_distance(old_coord, best_cell.coordinate)
        self.movement_tirement(n_cells)

    # ---------- Step ----------

    def step(self):
        # ----- Death handling (preserved from previous logic) -----
        # NOTE: counters break by state and cause. To attribute by return_reason
        # (e.g. deaths_returning_food vs _danger vs _both), add per-reason
        # counters to Model.py and switch on self.return_reason here.
        if self.is_dead():
            if self.state == State.FORAGING:
                self.model.deaths_foraging += 1
                if self.energy <= 0:
                    self.model.deaths_foraging_energy += 1
                else:
                    self.model.deaths_foraging_temperature += 1

            elif self.state == State.RETURNING:
                self.model.deaths_returning += 1
                if self.energy <= 0:
                    self.model.deaths_returning_energy += 1
                else:
                    self.model.deaths_returning_temperature += 1
            elif self.state == State.RESTING:
                self.model.deaths_resting += 1
                if self.energy <= 0:
                    self.model.deaths_resting_energy += 1
                else:
                    self.model.deaths_resting_temperature += 1

            if self.energy <= 0:
                self.model.deaths_energy += 1
            else:
                self.model.deaths_temperature += 1

            self.model.completed_agent_food_collections.append(self.lifetime_food_collected)
            self.remove()
            return

        # ----- Per-tick basal & environmental costs -----
        self.energy -= self.cost_E_rest
        if not self.is_in_nest():
            self.temperature += self.cost_T_env

        # ----- State machine -----
        match self.state:
            case State.FORAGING:
                if self.is_on_food():
                    self.consume_food()  # one eat this tick, then re-evaluate
                    e_danger, h_danger = self._project_danger_slow()

                    if not e_danger or h_danger:
                        # Energy is sufficient OR heat is the binding constraint
                        # (more eating won't cool us down — time to leave).
                        self.return_reason = ReturnReason.RETURN_FOOD
                        if any(self._project_danger_slow()):
                            self._resolve_return_reason_after_slow_danger()
                        self.state = State.RETURNING

                    elif any(self._project_danger_fast()):
                        # Still energy-deficient but we've run out of time even for fast escape.
                        # Must leave now regardless — escalate immediately.
                        self.return_reason = ReturnReason.RETURN_FOOD
                        self._resolve_return_reason_after_slow_danger()  # -> BOTH
                        self.state = State.RETURNING

                    # else: e_danger=True, h_danger=False, fast escape still safe.
                    # Stay put, eat again next tick. move_logic_foraging already no-ops on food.

                elif any(self._project_danger_fast()):
                    self.return_reason = ReturnReason.RETURN_DANGER
                    self.state = State.RETURNING

                else:
                    self.move()

            case State.RETURNING:
                if self.is_in_nest():
                    self.state = State.RESTING
                    self.has_food = False
                    self.return_reason = None
                else:
                    # Eat-while-returning (FOOD reason only).
                    # Conditions:
                    #   - reason is FOOD (DANGER agents don't eat; BOTH can't be saved by eating)
                    #   - currently on a food cell
                    #   - energy is the bottleneck and heat is not (eating only fixes energy)
                    #   - post-eat energy clears the slow-return threshold
                    if self.return_reason == ReturnReason.RETURN_FOOD and self.is_on_food():
                        e_danger, h_danger = self._project_danger_slow()
                        if e_danger and not h_danger:
                            dist = self._chebyshev_distance(self.cell.coordinate, self.model.nest_coords)
                            n_cells = dist + self.model.safety_buffer_steps
                            n_ticks = n_cells
                            energy_needed = n_ticks * self.cost_E_rest + n_cells * self.cost_E_move
                            if (self.energy + self.food_energy_gain) > energy_needed:
                                self.consume_food()

                    # ---- FOOD -> BOTH upgrade ----
                    # If still FOOD after the optional eat, re-check slow-projection.
                    # If now unsafe, escalate via the resolver.
                    if self.return_reason == ReturnReason.RETURN_FOOD and any(self._project_danger_slow()):
                        self._resolve_return_reason_after_slow_danger()

                    self.move()

            case State.RESTING:
                self.cool_down()
                if self.should_start_foraging():
                    self.state = State.FORAGING

        self.age_steps += 1