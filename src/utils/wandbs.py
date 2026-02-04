import os
import logging
import wandb
from src.utils.loaders import load_env

logger = logging.getLogger(__name__)


def login_wandb() -> None:
    load_env()
    api_key = os.getenv("WANDB_API_KEY")
    if api_key:
        wandb.login(key=api_key, relogin=True)
        logger.info("W&B login successful.")
    else:
        logger.warning("WANDB_API_KEY not found.")


def init_wandb(project: str, name: str, config: dict = None, tags: list = None):
    login_wandb()
    wandb.init(
        project=project,
        name=name,
        tags=tags or [],
        config=config or {},
        settings=wandb.Settings(console="off")
    )
    logger.info(f"W&B initialized: project={project}, name={name}")


def log_metrics(metrics: dict, step: int = None):
    if wandb.run is None:
        return
    wandb.log(metrics, step=step)
    logger.debug(f"W&B logged metrics: {metrics}")


def log_summary(metrics: dict):
    if wandb.run is None:
        return
    try:
        for key, value in metrics.items():
            wandb.run.summary[key] = value
        logger.info(f"Logged summary to W&B: {metrics}")
    except Exception as e:
        logger.warning(f"Failed to log summary: {e}")


def log_images(images: list, key: str = "predictions"):
    """W&B에 이미지 로깅

    Args:
        images: wandb.Image 리스트 또는 numpy array 리스트
        key: 로깅할 키 이름
    """
    if wandb.run is None:
        return

    wandb_images = []
    for img in images:
        if isinstance(img, wandb.Image):
            wandb_images.append(img)
        else:
            wandb_images.append(wandb.Image(img))

    wandb.log({key: wandb_images})


def finish_wandb():
    if wandb.run is not None:
        wandb.finish()
        logger.info("W&B run finished.")
