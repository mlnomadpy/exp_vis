import wandb

def init_wandb(project: str, config: dict):
    return wandb.init(project=project, config=config)

def log_metrics(metrics: dict, step: int = None):
    if step is not None:
        wandb.log(metrics, step=step)
    else:
        wandb.log(metrics)

def log_artifact(name: str, type_: str, files: list = None, metadata: dict = None):
    artifact = wandb.Artifact(name, type=type_)
    if files:
        for f in files:
            artifact.add_file(f)
    if metadata:
        artifact.metadata.update(metadata)
    wandb.log_artifact(artifact)
    return artifact 