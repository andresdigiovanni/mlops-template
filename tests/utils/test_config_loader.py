import pytest
import yaml

from src.utils import load_config


def test_load_config_valid(tmp_path):
    # Arrange
    config_path = tmp_path / "config.yaml"
    config_data = {
        "data": {"random_state": 42},
        "model": {"type": "lr", "params": {"max_iter": 100}},
    }
    with open(config_path, "w") as f:
        yaml.dump(config_data, f)

    # Act
    config = load_config(config_path)

    # Assert
    assert config["data"]["random_state"] == 42
    assert config["model"]["type"] == "lr"
    assert config["model"]["params"]["max_iter"] == 100


def test_load_config_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_config("non_existing_config.yaml")
