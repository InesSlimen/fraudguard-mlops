from products.fraudguard.config import get_project_config


def test_project_config_defaults():
    config = get_project_config()

    assert config.name == "fraudguard"
    assert config.model_name == "fraudguard-risk-model"
    assert config.environment == "local"
