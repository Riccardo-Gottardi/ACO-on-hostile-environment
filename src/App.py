import solara
from mesa.visualization import SolaraViz, make_space_component, make_plot_component
from mesa.visualization.components import AgentPortrayalStyle

# Import your model and agents
from Model import Environment
from Agent import CreatureAgent
from CellAgent import EnvCell

def agent_portrayal(agent):
    if agent is None:
        return
        
    # 1. Creature Portrayal
    if isinstance(agent, CreatureAgent):
        # Apply your requested colors based on state
        color_map = {
            "RESTING": "blue", 
            "FORAGING": "black", 
            "RETURNING": "red"
        }
        color = color_map.get(agent.state, "grey")
        return AgentPortrayalStyle(size=50, marker="o", zorder=2, color=color)
        
    # 2. Environment Cell Portrayal
    elif isinstance(agent, EnvCell):
        # Priority 1: The Nest
        if agent.is_nest:
            return AgentPortrayalStyle(size=200, marker="s", zorder=1, color="orange")
            
        # Priority 2: Food Clusters
        elif agent.food_quantity > 0:
            return AgentPortrayalStyle(size=200, marker="s", zorder=0, color="green")
            
        # Priority 3: Pheromone Trails (Dynamic Opacity)
        elif agent.pheromone_level > 0.001:
            # Calculate alpha (opacity). Assuming 5.0 is a "very strong" trail.
            # Caps at 1.0 (fully opaque) and floors at 0.1 (faintly visible).
            calculated_alpha = min(1.0, agent.pheromone_level / 5.0)
            calculated_alpha = max(0.1, calculated_alpha)
            
            return AgentPortrayalStyle(
                size=200, 
                marker="s", 
                zorder=-1, 
                color="purple", 
                alpha=calculated_alpha
            )
            
        # Background Grid (Empty cells)
        else:
            return AgentPortrayalStyle(size=200, marker="s", zorder=-2, color="lightgrey", alpha=0.1)

# Sliders and Parameters to control the Environment dynamically
model_params = {
    "width": 60,
    "height": 60,
    "num_agents": {
        "type": "SliderInt",
        "value": 50,
        "label": "Number of Agents",
        "min": 40,
        "max": 60,
        "step": 1
    },
    "pheromone_decay_rate": {
        "type": "SliderFloat",
        "value": 0.02,
        "label": "Pheromone Decay Rate",
        "min": 0.01,
        "max": 0.05,
        "step": 0.01
    },
    "safety_buffer_steps": {
        "type": "SliderInt",
        "value": 2,
        "label": "Safety Buffer Steps",
        "min": 0,
        "max": 5,
        "step": 1
    },
    "foraging_start_threshold": {
        "type": "SliderFloat",
        "value": 0.9,
        "label": "Foraging Start Threshold",
        "min": 0.5,
        "max": 1.0,
        "step": 0.01
    },
    "pheromone_memory_weight": {
        "type": "SliderFloat",
        "value": 0.1,
        "label": "Pheromone Memory Weight",
        "min": 0.0,
        "max": 0.5,
        "step": 0.05
    },
    "pheromone_base_drop": {
        "type": "SliderFloat",
        "value": 1.0,
        "label": "Pheromone Base Drop",
        "min": 0.0,
        "max": 2.0,
        "step": 0.1
    },
    "pheromone_follow_prob": {
        "type": "SliderFloat",
        "value": 0.7,
        "label": "Pheromone Follow Probability",
        "min": 0.0,
        "max": 1.0,
        "step": 0.05
    },
    "food_area_percentage": 0.15,
    "food_base_quantity": 1,
    "n_food_clusters": 12
}

# Initialize the model instance
model = Environment(
    width=60, height=60, num_agents=50, 
    n_food_clusters=12, food_area_percentage=0.15, 
    food_base_quantity=10, pheromone_decay_rate=0.05,
    safety_buffer_steps=2, foraging_start_threshold=0.9,
    pheromone_memory_weight=0.1,
    pheromone_base_drop=1.0, pheromone_follow_prob=0.7
)

# Build the UI components
space_component = make_space_component(agent_portrayal)
# Track the metrics defined in the DataCollector from model.py
state_plot_component = make_plot_component(["Foraging", "Returning", "Resting"])
survival_plot_component = make_plot_component(["Alive", "Dead (Energy)", "Dead (Temperature)"])
death_state_plot_component = make_plot_component([
    "Deaths Foraging (Energy)",
    "Deaths Foraging (Temperature)",
    "Deaths Returning (Energy)",
    "Deaths Returning (Temperature)",
    "Deaths Resting (Energy)",
    "Deaths Resting (Temperature)"
])
food_plot_component = make_plot_component(["Remaining Food (%)"])

page = SolaraViz(
    model,
    components=[space_component, state_plot_component, survival_plot_component, death_state_plot_component, food_plot_component],
    model_params=model_params,
    name="Swarm Intelligence: Foraging Environment"
)