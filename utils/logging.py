import wandb


def init_logging(cfg):
    if not cfg.no_wandb:
        wandb.init(config=cfg, project=cfg.project_name)
    else:
        raise NotImplementedError


def log_visualizations(cfg, visualizations):
    if not cfg.no_wandb:
        visualizations = NotImplemented  # convert visualizations into wandb types
        wandb.log(visualizations, commit=False)


def log_losses(cfg, losses):
    if not cfg.no_wandb:
        wandb.log(losses, commit=True)
    else:
        raise NotImplementedError


def log_final(cfg, metrics):
    if not cfg.no_wandb:
        wandb.log(metrics, commit=True)
        wandb.finish()
    else:
        raise NotImplementedError