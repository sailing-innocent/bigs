import json
import os


def get_config(name: str):
    config_json_f = os.path.join(os.path.dirname(__file__), name + ".json")
    with open(config_json_f, "r") as f:
        config = json.load(f)
    return config


def get_nvos_json():
    config_json_f = os.path.join(os.path.dirname(__file__), "nvos.json")
    with open(config_json_f, "r") as f:
        config = json.load(f)
    return config


def get_reprod_config():
    config_json_f = os.path.join(os.path.dirname(__file__), "reprod.json")
    with open(config_json_f, "r") as f:
        config = json.load(f)
    return config


def get_gsd_json():
    config_json_f = os.path.join(os.path.dirname(__file__), "gsd.json")
    with open(config_json_f, "r") as f:
        config = json.load(f)
    return config


def get_teaser_json():
    config_json_f = os.path.join(os.path.dirname(__file__), "teaser.json")
    with open(config_json_f, "r") as f:
        config = json.load(f)
    return config


def get_langsplat_json():
    config_json_f = os.path.join(os.path.dirname(__file__), "langsplat.json")
    with open(config_json_f, "r") as f:
        config = json.load(f)
    return config


def get_style_json():
    config_json_f = os.path.join(os.path.dirname(__file__), "style.json")
    with open(config_json_f, "r") as f:
        config = json.load(f)
    return config
