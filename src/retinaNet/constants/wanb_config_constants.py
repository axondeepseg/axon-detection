import wandb
api = wandb.Api()

WANDB_ENTITY = "neuropoly-axon-detection"
WANDB_PROJECT = "retinanet-project"

project_runs = api.runs(f"{WANDB_ENTITY}/{WANDB_PROJECT}")

run_count = len(project_runs) + 1
WANDB_RUN_NAME = f"run_{run_count}"