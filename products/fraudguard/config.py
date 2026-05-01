from dataclasses import dataclass


@dataclass(frozen=True)
class ProjectConfig:
    name: str = "fraudguard"
    model_name: str = "fraudguard-risk-model"
    environment: str = "local"


def get_project_config() -> ProjectConfig:
    return ProjectConfig()
