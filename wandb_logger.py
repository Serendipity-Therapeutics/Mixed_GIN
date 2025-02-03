# wandb_logger.py

import wandb

class WandbLogger:

    def __init__(self, project, run_name, config=None):

        self.run = wandb.init(
            project=project,
            name=run_name,
            config=config if config else {},
            reinit=True 
        )

    def log(self, metrics: dict, step=None):
        if step is not None:
            wandb.log(metrics, step=step)
        else:
            wandb.log(metrics)

    def finish(self):
        wandb.finish()