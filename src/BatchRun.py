import optuna
import pandas as pd
import os
import warnings
from multiprocessing import freeze_support
import random
import math
from Model import Environment

warnings.simplefilter("ignore", category=FutureWarning)
experiment_name = "swarm_foraging_experiment"

MODEL_DEFAULTS = {
    "width": 60,
    "height": 60,
    "n_food_clusters": 12,
    "food_area_percentage": 0.15,
    "food_base_quantity": 10,
}

EVALUATION_RUNS = 10
FINAL_CONFIRMATION_RUNS = 5

def _build_rng(seed):
    return random.Random(seed) if seed is not None else None

def _normalize_dataframe_index(frame, step_column="Step", agent_column="AgentID"):
    if frame.empty:
        return frame.copy()

    normalized = frame.copy().reset_index()
    rename_map = {}

    if step_column not in normalized.columns:
        if "index" in normalized.columns:
            rename_map["index"] = step_column
        elif "level_0" in normalized.columns:
            rename_map["level_0"] = step_column

    if agent_column not in normalized.columns:
        if "level_1" in normalized.columns:
            rename_map["level_1"] = agent_column

    if rename_map:
        normalized = normalized.rename(columns=rename_map)

    if step_column in normalized.columns:
        normalized[step_column] = normalized[step_column].astype(int)

    return normalized

def _safe_float(value, fallback):
    if pd.isna(value):
        return float(fallback)
    return float(value)

def _simulation_score(summary):
    survival_component = max(0.0, min(1.0, summary["Survival Ratio"]))
    food_component = max(0.0, min(1.0, summary["Food Collection Ratio"]))
    retrieval_component = 1.0 - math.exp(-max(0.0, summary["Resource Retrieval Rate"]))
    thermal_component = summary["Thermal Efficiency"] / (1.0 + summary["Thermal Efficiency"]) if summary["Thermal Efficiency"] > 0 else 0.0
    fairness_component = max(0.0, min(1.0, 1.0 - summary["Load Gini"]))

    return (
        0.30 * survival_component
        + 0.20 * food_component
        + 0.20 * retrieval_component
        + 0.20 * thermal_component
        + 0.10 * fairness_component
    )

def _run_single_simulation(parameters, max_steps, seed):
    model = Environment(**parameters, rng=_build_rng(seed))

    while model.running and model.steps_elapsed < max_steps:
        model.step()

    model_df = _normalize_dataframe_index(model.datacollector.get_model_vars_dataframe(), step_column="Step")
    agent_df = _normalize_dataframe_index(model.datacollector.get_agent_vars_dataframe(), step_column="Step", agent_column="AgentID")

    return model, model_df, agent_df

def _summarize_run(model, model_df, parameters, run_id, seed, max_steps):
    final_row = model_df.iloc[-1] if not model_df.empty else pd.Series(dtype=float)

    completed_steps = int(_safe_float(final_row.get("Step", model.steps_elapsed), model.steps_elapsed))
    total_food_collected = _safe_float(final_row.get("Total Food Collected", model.total_food_collected), model.total_food_collected)
    remaining_food_units = _safe_float(final_row.get("Remaining Food (Units)", model.get_remaining_food_units()), model.get_remaining_food_units())
    resource_retrieval_rate = _safe_float(final_row.get("Resource Retrieval Rate", model.calculate_resource_retrieval_rate()), model.calculate_resource_retrieval_rate())
    thermal_efficiency = _safe_float(final_row.get("Thermal Efficiency", model.calculate_thermal_efficiency()), model.calculate_thermal_efficiency())
    load_gini = _safe_float(final_row.get("Load Gini", model.calculate_load_gini()), model.calculate_load_gini())
    cumulative_thermal_load = _safe_float(final_row.get("Cumulative Thermal Load", model.cumulative_thermal_load), model.cumulative_thermal_load)
    shannon_entropy = _safe_float(final_row.get("Shannon Entropy", model.calculate_spatial_entropy()), model.calculate_spatial_entropy())
    mean_agent_energy = _safe_float(final_row.get("Mean Agent Energy", model.calculate_mean_agent_energy()), model.calculate_mean_agent_energy())
    mean_agent_temperature = _safe_float(final_row.get("Mean Agent Temperature", model.calculate_mean_agent_temperature()), model.calculate_mean_agent_temperature())
    mean_distance_to_nest = _safe_float(final_row.get("Mean Distance to Nest", model.calculate_mean_distance_to_nest()), model.calculate_mean_distance_to_nest())
    mean_lifetime_food_collected = _safe_float(final_row.get("Mean Lifetime Food Collected", model.calculate_mean_lifetime_food_collected()), model.calculate_mean_lifetime_food_collected())

    summary = {
        "RunId": run_id,
        "Seed": seed,
        "Completed Steps": completed_steps,
        "Survival Ratio": completed_steps / max(1, max_steps),
        "Total Food Collected": total_food_collected,
        "Food Collection Ratio": total_food_collected / max(1, model.initial_food),
        "Remaining Food (Units)": remaining_food_units,
        "Remaining Food (%)": _safe_float(final_row.get("Remaining Food (%)", 0.0), 0.0),
        "Resource Retrieval Rate": resource_retrieval_rate,
        "Thermal Efficiency": thermal_efficiency,
        "Load Gini": load_gini,
        "Fairness Score": max(0.0, min(1.0, 1.0 - load_gini)),
        "Cumulative Thermal Load": cumulative_thermal_load,
        "Shannon Entropy": shannon_entropy,
        "Mean Agent Energy": mean_agent_energy,
        "Mean Agent Temperature": mean_agent_temperature,
        "Mean Distance to Nest": mean_distance_to_nest,
        "Mean Lifetime Food Collected": mean_lifetime_food_collected,
    }

    summary["Optimization Score"] = _simulation_score(summary)
    for key, value in parameters.items():
        summary[key] = value

    return summary

def _attach_metadata(frame, metadata):
    if frame.empty:
        return frame.copy()

    enriched = frame.copy()
    for key, value in metadata.items():
        enriched[key] = value

    return enriched

def _simulate_runs(parameters, runs, max_steps, seed_start=0, show_progress=False):
    model_frames = []
    agent_frames = []
    summary_rows = []

    for run_id in range(runs):
        seed = seed_start + run_id
        model, model_df, agent_df = _run_single_simulation(parameters, max_steps=max_steps, seed=seed)
        summary = _summarize_run(model, model_df, parameters, run_id, seed, max_steps)

        run_metadata = {"RunId": run_id, "Seed": seed, **parameters}
        summary_metadata = {**run_metadata, **{key: value for key, value in summary.items() if key not in run_metadata}}

        model_frames.append(_attach_metadata(model_df, summary_metadata))
        if not agent_df.empty:
            agent_frames.append(_attach_metadata(agent_df, run_metadata))
        summary_rows.append(summary)

        if show_progress:
            print(f"Completed simulation run {run_id + 1}/{runs} (seed={seed})")

    final_df = pd.concat(model_frames, ignore_index=True) if model_frames else pd.DataFrame()
    agent_df = pd.concat(agent_frames, ignore_index=True) if agent_frames else pd.DataFrame()
    summary_df = pd.DataFrame(summary_rows)

    if not final_df.empty and {"RunId", "Step"}.issubset(final_df.columns):
        final_df = final_df.sort_values(["RunId", "Step"]).reset_index(drop=True)

    if not agent_df.empty and {"RunId", "Step", "AgentID"}.issubset(agent_df.columns):
        agent_df = agent_df.sort_values(["RunId", "Step", "AgentID"]).reset_index(drop=True)

    if not summary_df.empty and "Optimization Score" in summary_df.columns:
        summary_df = summary_df.sort_values(["Optimization Score", "RunId"], ascending=[False, True]).reset_index(drop=True)

    return final_df, summary_df, agent_df

def objective(trial):
    """Objective function for Optuna Bayesian optimization."""
    params = MODEL_DEFAULTS.copy()
    params["num_agents"] = trial.suggest_int("num_agents", 40, 60)
    params["pheromone_decay_rate"] = trial.suggest_float("pheromone_decay_rate", 0.01, 0.20)
    params["safety_buffer_steps"] = trial.suggest_int("safety_buffer_steps", 1, 3)
    params["foraging_start_threshold"] = trial.suggest_float("foraging_start_threshold", 0.85, 0.95)
    params["pheromone_memory_weight"] = trial.suggest_float("pheromone_memory_weight", 0.05, 0.15)
    params["pheromone_base_drop"] = trial.suggest_float("pheromone_base_drop", 0.5, 1.5)
    params["pheromone_follow_prob"] = trial.suggest_float("pheromone_follow_prob", 0.5, 0.9)

    _, summary_df, _ = _simulate_runs(
        params,
        runs=EVALUATION_RUNS,
        max_steps=43200,
        seed_start=trial.number * 1000,
        show_progress=False,
    )

    return float(summary_df["Optimization Score"].mean()) if not summary_df.empty else 0.0

def run_optimization_search(n_trials=50, show_progress=True):
    """Run Optuna search and return the completed study."""
    study = optuna.create_study(direction="maximize", study_name=experiment_name)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=show_progress)
    return study

def run_final_confirmation(best_params, runs=FINAL_CONFIRMATION_RUNS, max_steps=129600, show_progress=True):
    """Run the final batch using the optimized parameters."""
    final_params = {**MODEL_DEFAULTS, **best_params}
    return _simulate_runs(
        final_params,
        runs=runs,
        max_steps=max_steps,
        seed_start=0,
        show_progress=show_progress,
    )

def save_results(final_df, summary_df=None, agent_df=None, file_name=None):
    """Save batch results to timestamped CSV files."""
    if file_name is None:
        file_name = f"{experiment_name}_optimized_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv"

    results_path = os.path.dirname(os.path.abspath(__file__))
    csv_file = os.path.join(results_path, file_name)
    final_df.to_csv(csv_file, index=False)

    summary_csv_file = None
    if summary_df is not None and not summary_df.empty:
        base_name, extension = os.path.splitext(file_name)
        summary_csv_file = os.path.join(results_path, f"{base_name}_summary{extension}")
        summary_df.to_csv(summary_csv_file, index=False)

    agent_csv_file = None
    if agent_df is not None and not agent_df.empty:
        base_name, extension = os.path.splitext(file_name)
        agent_csv_file = os.path.join(results_path, f"{base_name}_agents{extension}")
        agent_df.to_csv(agent_csv_file, index=False)

    return {
        "csv_file": csv_file,
        "summary_csv_file": summary_csv_file,
        "agent_csv_file": agent_csv_file,
    }

def run_experiment(n_trials=50, final_runs=FINAL_CONFIRMATION_RUNS, final_max_steps=129600):
    """Execute the optimization search and final confirmation batch."""
    freeze_support()

    study = run_optimization_search(n_trials=n_trials, show_progress=True)
    final_df, summary_df, agent_df = run_final_confirmation(
        study.best_params,
        runs=final_runs,
        max_steps=final_max_steps,
        show_progress=True,
    )
    saved_files = save_results(final_df, summary_df=summary_df, agent_df=agent_df)

    return {
        "study": study,
        "final_df": final_df,
        "summary_df": summary_df,
        "agent_df": agent_df,
        **saved_files,
    }
