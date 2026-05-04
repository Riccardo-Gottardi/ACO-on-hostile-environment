from Model import Environment
from mesa.batchrunner import batch_run
from multiprocessing import freeze_support
import pandas as pd
import os
import warnings

warnings.simplefilter("ignore", category=FutureWarning)
experiment_name = "swarm_foraging_experiment"

MODEL_DEFAULTS = {
    "width": 60,
    "height": 60,
    "num_agents": 50,
    "n_food_clusters": 12,
    "food_area_percentage": 0.15,
    "food_base_quantity": 10,
    "pheromone_decay_rate": 0.02,
    "safety_buffer_steps": 2,
    "foraging_start_threshold": 0.9,
    "pheromone_memory_weight": 0.1,
    "pheromone_base_drop": 1.0,
    "pheromone_follow_prob": 0.7,
}

BEHAVIOR_DEFAULTS = {
    "safety_buffer_steps": 2,
    "foraging_start_threshold": 0.9,
    "pheromone_memory_weight": 0.1,
    "pheromone_base_drop": 1.0,
    "pheromone_follow_prob": 0.7,
}

BEHAVIOR_SWEEPS = {
    "safety_buffer_steps": [1, 2, 3],
    "foraging_start_threshold": [0.85, 0.9, 0.95],
    "pheromone_memory_weight": [0.05, 0.1, 0.15],
    "pheromone_base_drop": [0.5, 1.0, 1.5],
    "pheromone_follow_prob": [0.5, 0.7, 0.9],
}

def get_best_parameters(results_df, param_keys):
    """
    Analyzes batch run results to find the parameters that yield the longest survival.
    """
    # 1. Find the maximum step (survival time) for each unique run
    run_lifespans = results_df.groupby('RunId')['Step'].max().reset_index()
    
    # 2. Re-attach the parameters that were used for each RunId
    run_params = results_df[['RunId'] + param_keys].drop_duplicates()
    lifespan_data = pd.merge(run_lifespans, run_params, on='RunId')
    
    # 3. Calculate average survival time across iterations for each parameter combo
    mean_lifespans = lifespan_data.groupby(param_keys)['Step'].mean().reset_index()
    
    # 4. Extract the row with the absolute highest average survival time
    best_row = mean_lifespans.loc[mean_lifespans['Step'].idxmax()]
    
    return best_row

if __name__ == '__main__':
    freeze_support()
    
    # ==========================================
    # PHASE 1: COARSE SEARCH
    # ==========================================
    print("--- PHASE 1: Starting Coarse Search ---")
    
    coarse_params = {
        **MODEL_DEFAULTS,
        "num_agents": [40, 50, 60],                  # Broad jumps testing extremes and middle
        "pheromone_decay_rate": [0.01, 0.10, 0.20],  # Broad jumps testing low, medium, high
    }

    coarse_results = batch_run(
        Environment,
        parameters=coarse_params,
        rng=[None] * 3,
        max_steps=43200,         
        number_processes=None,   
        display_progress=True,
        data_collection_period=1
    )
    
    coarse_df = pd.DataFrame(coarse_results)
    
    # Determine the best performing parameters from the coarse search
    param_keys_to_optimize = ["num_agents", "pheromone_decay_rate"]
    best_coarse = get_best_parameters(coarse_df, param_keys_to_optimize)
    
    best_agents = int(best_coarse["num_agents"])
    best_decay = float(best_coarse["pheromone_decay_rate"])
    
    print(f"\n✅ Coarse Search Complete!")
    print(f"Best Swarm Size: {best_agents}")
    print(f"Best Decay Rate: {best_decay:.3f}")
    print(f"Average Survival Steps: {best_coarse['Step']:.1f}")
    
    # ==========================================
    # PHASE 2: REFINED SEARCH
    # ==========================================
    print("\n--- PHASE 2: Starting Refined Search ---")
    
    # Create tighter test intervals around the winners of the coarse search.
    # The max/min functions ensure we don't accidentally test outside logical bounds.
    refined_params = {
        **MODEL_DEFAULTS,
        "num_agents": [
            max(40, best_agents - 2), 
            best_agents, 
            min(60, best_agents + 2)
        ],
        "pheromone_decay_rate": [
            max(0.01, best_decay - 0.02), 
            best_decay, 
            best_decay + 0.02
        ],
    }

    refined_results = batch_run(
        Environment,
        parameters=refined_params,
        rng=[None] * 5,          # More iterations for higher statistical confidence
        max_steps=43200,
        number_processes=None,    
        display_progress=True,
        data_collection_period=1
    )
    
    refined_df = pd.DataFrame(refined_results)

    best_refined = get_best_parameters(refined_df, param_keys_to_optimize)
    best_agents = int(best_refined["num_agents"])
    best_decay = float(best_refined["pheromone_decay_rate"])

    print(f"\n✅ Refined Search Complete!")
    print(f"Refined Best Swarm Size: {best_agents}")
    print(f"Refined Best Decay Rate: {best_decay:.3f}")
    print(f"Average Survival Steps: {best_refined['Step']:.1f}")

    # ==========================================
    # PHASE 3: BEHAVIOR HYPERPARAMETER SEARCH
    # ==========================================
    print("\n--- PHASE 3: Starting Behavior Hyperparameter Search ---")

    best_behavior_params = BEHAVIOR_DEFAULTS.copy()
    behavior_results = []

    for param_name, candidate_values in BEHAVIOR_SWEEPS.items():
        print(f"Optimizing {param_name}...")

        sweep_params = {
            **MODEL_DEFAULTS,
            **best_behavior_params,
            "num_agents": best_agents,
            "pheromone_decay_rate": best_decay,
            param_name: candidate_values,
        }

        sweep_results = batch_run(
            Environment,
            parameters=sweep_params,
            rng=[None] * 5,
            max_steps=129600,
            number_processes=None,
            display_progress=True,
            data_collection_period=1
        )

        sweep_df = pd.DataFrame(sweep_results)
        behavior_results.append(sweep_df)

        best_behavior_row = get_best_parameters(sweep_df, [param_name])
        if param_name == "safety_buffer_steps":
            best_behavior_params[param_name] = int(best_behavior_row[param_name])
        else:
            best_behavior_params[param_name] = float(best_behavior_row[param_name])

        print(f"Best {param_name}: {best_behavior_params[param_name]}")

    behavior_df = pd.concat(behavior_results, ignore_index=True)

    print("\n✅ Behavior Hyperparameter Search Complete!")
    print("Best Behavior Hyperparameters:")
    for param_name, value in best_behavior_params.items():
        print(f"{param_name}: {value}")

    # ==========================================
    # PHASE 4: FINAL CONFIRMATION RUN
    # ==========================================
    print("\n--- PHASE 4: Starting Final Confirmation Run ---")

    final_params = {
        **MODEL_DEFAULTS,
        **best_behavior_params,
        "num_agents": best_agents,
        "pheromone_decay_rate": best_decay,
    }

    final_results = batch_run(
        Environment,
        parameters=final_params,
        rng=[None] * 5,
        max_steps=129600,
        number_processes=None,
        display_progress=True,
        data_collection_period=1
    )

    final_df = pd.DataFrame(final_results)
    
    # Generate timestamped CSV file for the final, refined results
    csv_file_name = f'{experiment_name}_refined_{pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")}.csv'
    results_path = os.path.dirname(os.path.abspath(__file__))
    csv_file = os.path.join(results_path, csv_file_name)
    final_df.to_csv(csv_file)
    
    print(f"\n✅ Final optimized batch run complete! Final results saved to:\n{csv_file}")