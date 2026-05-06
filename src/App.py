import solara
from mesa.visualization import SolaraViz, SpaceRenderer, make_plot_component
from mesa.visualization.components import AgentPortrayalStyle, PropertyLayerStyle

from Model import Environment
from Agent import CreatureAgent

def agent_portrayal(agent):
    if agent is None:
        return

    if isinstance(agent, CreatureAgent):
        color_map = {
            "RESTING": "blue",
            "FORAGING": "black",
            "RETURNING": "red",
        }
        color = color_map.get(agent.state, "grey")
        return AgentPortrayalStyle(size=50, marker="o", zorder=2, color=color)


def make_property_layer_portrayal(layer):
    if layer.name == "food":
        return PropertyLayerStyle(
            color="green",
            alpha=1.0,
            colorbar=False,
            vmin=0,
            vmax=0.001,  # Solid green for any amount of food
        )

    if layer.name == "pheromone":
        return PropertyLayerStyle(
            colormap="Purples",
            alpha=0.55,
            colorbar=False,
            vmin=0,
            vmax=max(1.0, float(layer.data.max())),
        )

# Sliders and Parameters to control the Environment dynamically
model_params = {
    "width": 60,
    "height": 60,
    "num_agents": {
        "type": "SliderInt",
        "value": 41,
        "label": "Number of Agents",
        "min": 40,
        "max": 60,
        "step": 1
    },
    "pheromone_decay_rate": {
        "type": "SliderFloat",
        "value": 0.065073771,
        "label": "Pheromone Decay Rate",
        "min": 0.01,
        "max": 0.1,
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
        "value": 0.587912346,
        "label": "Foraging Start Threshold",
        "min": 0.5,
        "max": 1.0,
        "step": 0.01
    },
    "pheromone_memory_weight": {
        "type": "SliderFloat",
        "value": 0.140481121,
        "label": "Pheromone Memory Weight",
        "min": 0.0,
        "max": 0.5,
        "step": 0.05
    },
    "pheromone_base_drop": {
        "type": "SliderFloat",
        "value": 1.095370208,
        "label": "Pheromone Base Drop",
        "min": 0.0,
        "max": 2.0,
        "step": 0.1
    },
    "pheromone_follow_prob": {
        "type": "SliderFloat",
        "value": 0.004365772,
        "label": "Pheromone Follow Probability",
        "min": 0.0,
        "max": 1.0,
        "step": 0.05
    },
    "food_area_percentage": 0.15,
    "food_base_quantity": 10,
    "n_food_clusters": 12
}

model = Environment(
    width=60,
    height=60,
    num_agents=41,
    n_food_clusters=12,
    food_area_percentage=0.15,
    food_base_quantity=10,
    pheromone_decay_rate=0.065073771,
    safety_buffer_steps=2,
    foraging_start_threshold=0.587912346,
    pheromone_memory_weight=0.140481121,
    pheromone_base_drop=1.095370208,
    pheromone_follow_prob=0.004365772,
)

renderer = SpaceRenderer(model=model, backend="matplotlib")
renderer.setup_structure(linewidth=1.0, color="black") # Added grid outline
renderer.setup_agents(agent_portrayal)
renderer.setup_propertylayer(make_property_layer_portrayal)
renderer.draw_structure() # Draw the grid
renderer.draw_agents()
renderer.draw_propertylayer()

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

@solara.component
def PlotDashboard(model):
    plots = [
        state_plot_component,
        survival_plot_component,
        death_state_plot_component,
        food_plot_component
    ]
    
    with solara.GridFixed(columns=2): 
        for plot in plots:
            # Mesa's make_plot_component returns (Component, page_number_integer)
            if isinstance(plot, tuple):
                plot_func = plot[0] # Grab just the function
                plot_func(model)    # Call it with the model
            # Fallback just in case it returns a direct component
            else:
                plot(model)

page = SolaraViz(
    model,
    renderer,
    components=[PlotDashboard], 
    model_params=model_params,
    name="Swarm Intelligence: Foraging Environment"
)