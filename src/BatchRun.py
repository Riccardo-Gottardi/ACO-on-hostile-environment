"""
Optuna proposes parameter sets; mesa.batch_run evaluates each set across a
fixed list of seeds. The winning configuration is replayed at higher fidelity
for inspection.
"""

import math
import os
import warnings

import mesa
import optuna
import pandas as pd

from Model import Environment

warnings.simplefilter("ignore", category=FutureWarning)

EXPERIMENT_NAME = "swarm_foraging_experiment"
EVALUATION_RUNS = 15          # seeds swept per Optuna trial
FINAL_CONFIRMATION_RUNS = 30    # seeds for the final, high-fidelity batch
TRIAL_MAX_STEPS = 43_200
FINAL_MAX_STEPS = 129_600

FIXED_PARAMS = {
    "width": 60,
    "height": 60,
    "n_food_clusters": 12,
    "food_area_percentage": 0.15,
    "food_base_quantity": 10,
}

HYPOTHESIS_PARAMS = {
    "num_agents": 50,
    "pheromone_decay_rate": 0.03,
    "pheromone_follow_prob": 0.75,
    "foraging_start_threshold": 0.90,
    "safety_buffer_steps": 2,
    "food_richness_memory_regulator": 0.05
}


# =========== Scoring =======================================================-----

def _score_row(row):
    """Map one batch_run result row into an Optimization Score in [0, 1]."""
    survival = max(0.0, min(1.0, row["Survival Ratio"]))
    food = max(0.0, min(1.0, row["Food Collection Ratio"]))
    retrieval = 1.0 - math.exp(-max(0.0, row["Resource Retrieval Rate"]))

    if row["Thermal Efficiency"] > 0:
        thermal = row["Thermal Efficiency"] / (1.0 + row["Thermal Efficiency"]) 
    else:
        thermal = 0.0

    fairness = max(0.0, min(1.0, 1.0 - row["Load Gini"]))

    food_bonus = 0.05 * food
    retrieval_bonus = 0.02 * retrieval
    thermal_bonus = 0.02 * thermal
    fairness_bonus = 0.01 * fairness
    
    total_multiplier = 1.0 + food_bonus + retrieval_bonus + thermal_bonus + fairness_bonus
    
    return survival * total_multiplier


def _add_derived_metrics(df, max_steps):
    """Append the two ratios the model doesn't report on its own."""
    if df.empty:
        return df

    df = df.copy()
    df["Survival Ratio"] = df["Step"] / max_steps
    food_total = df["Total Food Collected"] + df["Remaining Food (Units)"]
    df["Food Collection Ratio"] = df["Total Food Collected"] / food_total.clip(lower=1.0)
    df["Optimization Score"] = df.apply(_score_row, axis=1)

    return df


# =========== Core: run a sweep over seeds =================================---

def _run_seeds(parameters, seeds, max_steps, *, per_step=False):
    """Sweep `seeds` for one configuration via mesa.batch_run.

    per_step=False  → one row per run (final step). Use for trial scoring.
    per_step=True   → one row per (run, step). Use for plotting traces.
    """
    sweep_params = {**parameters, "seed": list(seeds)}
    results = mesa.batch_run(
        Environment,
        parameters=sweep_params,
        iterations=1,
        max_steps=max_steps,
        data_collection_period=1 if per_step else -1,
        number_processes=1,            # parallelize at the trial level instead
        display_progress=False,
    )

    return pd.DataFrame(results)


# =========== Optuna objective ============================================-----

def objective(trial):
    suggested = {
        "num_agents": trial.suggest_int("num_agents", 40, 60),
        "pheromone_decay_rate": trial.suggest_float("pheromone_decay_rate", 0.01, 0.1),
        "safety_buffer_steps": trial.suggest_int("safety_buffer_steps", 1, 5),
        "foraging_start_threshold": trial.suggest_float("foraging_start_threshold", 0.5, 1.0),
        "pheromone_follow_prob": trial.suggest_float("pheromone_follow_prob", 0.0, 1.0),
        "food_richness_memory_regulator": trial.suggest_float("food_richness_memory_regulator", 1e-3, 5e-1, log=True)
    }
    parameters = {**FIXED_PARAMS, **suggested}

    base = trial.number * 1000
    seeds = range(base, base + EVALUATION_RUNS)

    df = _run_seeds(parameters, seeds, max_steps=TRIAL_MAX_STEPS, per_step=False)
    df = _add_derived_metrics(df, max_steps=TRIAL_MAX_STEPS)

    return float(df["Optimization Score"].mean()) if not df.empty else 0.0


# =========== Final confirmation ============================================---

def run_final_confirmation(best_params, runs=FINAL_CONFIRMATION_RUNS, max_steps=FINAL_MAX_STEPS):
    """Replay the best configuration with full per-step traces."""
    parameters = {**FIXED_PARAMS, **best_params}
    seeds = range(runs)

    return _run_seeds(parameters, seeds, max_steps=max_steps, per_step=True)


def run_hypothesized_confirmation(runs=FINAL_CONFIRMATION_RUNS, max_steps=FINAL_MAX_STEPS):
    """Replay the hypothesized configuration with full per-step traces."""
    parameters = {**FIXED_PARAMS, **HYPOTHESIS_PARAMS}
    seeds = range(runs)

    return _run_seeds(parameters, seeds, max_steps=max_steps, per_step=True)


# =========== I/O =======================================================--------

def save_results(df, file_name=None):
    if file_name is None:
        ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        file_name = f"{EXPERIMENT_NAME}_optimized_{ts}.csv"

    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), file_name)
    df.to_csv(path, index=False)

    return path


# =========== Top-level orchestration =================================--------

def run_experiment(n_trials=300):
    study = optuna.create_study(direction="maximize", study_name=EXPERIMENT_NAME)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")

    final_df = run_final_confirmation(study.best_params)
    final_csv = save_results(final_df, file_name=f"{EXPERIMENT_NAME}_optimized_{ts}.csv")

    hypothesized_df = run_hypothesized_confirmation()
    hypothesized_csv = save_results(hypothesized_df, file_name=f"{EXPERIMENT_NAME}_hypothesized_{ts}.csv")

    print(f"Best params: {study.best_params}")
    print(f"Best value:  {study.best_value:.4f}")
    print(f"Saved optimized trace: {final_csv}")
    print(f"Saved hypothesized trace: {hypothesized_csv}")
    
    return {
        "study": study,
        "final_df": final_df,
        "final_csv": final_csv,
        "hypothesized_df": hypothesized_df,
        "hypothesized_csv": hypothesized_csv,
    }


if __name__ == "__main__":
    run_experiment()