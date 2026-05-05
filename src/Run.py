import argparse
import os
import subprocess
import sys


def run_app():
    """Launch App.py using Solara if available."""
    app_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "App.py")
    if not os.path.exists(app_path):
        raise FileNotFoundError(f"Unable to find App.py at {app_path}")

    print("Starting the Solara app from App.py...")
    command = [sys.executable, "-m", "solara", "run", app_path]
    try:
        subprocess.run(command, check=True)
    except FileNotFoundError as exc:
        raise RuntimeError(
            "Solara CLI is not available in this Python environment. "
            "Install Solara or run App.py using the Solara launcher directly."
        ) from exc


def run_batch():
    """Execute the batch experiment defined in BatchRun.py."""
    from BatchRun import run_experiment

    print("Running the batch experiment from BatchRun.py...")
    result = run_experiment()
    print(f"Batch experiment complete. Results saved to: {result['csv_file']}")
    return result


def prompt_choice():
    print("Choose what to run:")
    print("  1) App (App.py)")
    print("  2) Batch experiment (BatchRun.py)")
    print("  q) Quit")

    while True:
        choice = input("Enter 1, 2, or q: ").strip().lower()
        if choice in {"1", "app"}:
            return "app"
        if choice in {"2", "batch"}:
            return "batch"
        if choice in {"q", "quit", "exit"}:
            return None
        print("Invalid selection. Please enter 1, 2, or q.")


def main():
    parser = argparse.ArgumentParser(description="Launcher for App.py or BatchRun.py")
    parser.add_argument(
        "--mode",
        choices=["app", "batch"],
        help="Choose whether to run the Solara app or the batch experiment.",
    )
    args = parser.parse_args()

    mode = args.mode or prompt_choice()
    if mode == "app":
        run_app()
    elif mode == "batch":
        run_batch()
    else:
        print("No action selected. Exiting.")


if __name__ == "__main__":
    main()
