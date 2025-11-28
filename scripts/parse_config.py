import sys
from datetime import datetime
from pathlib import Path
from omegaconf import OmegaConf

OmegaConf.register_new_resolver("eval", eval)

file_mapping = {}
for path in Path("config").glob("**/*.yaml"):
    folder = path.parent.name
    file_mapping[path.stem] = folder


def parse_yaml(data, prefix=""):
    command_string = ""

    for key, value in sorted(data.items(), key=lambda item: item[0]):
        if key == "dataparser":
            continue
        key = key.replace("_", "-")
        if isinstance(value, dict):
            command_string += parse_yaml(value, prefix + f"{key}.")
        elif isinstance(value, list):
            if all([isinstance(item, (int, float, str)) for item in value]):
                command_string += f"--{prefix}{key} {' '.join([str(item) for item in value])} "
            else:
                for i, item in enumerate(value):
                    if isinstance(item, dict):
                        command_string += parse_yaml(item, prefix + f"{key}.{i}.")
        else:
            command_string += f"--{prefix}{key} {value} "
    return command_string


if __name__ == "__main__":
    data_path, config_path, scene_name = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
    deform_ckpt = sys.argv[4] if len(sys.argv) > 4 else None
    base_config_path = config_path / file_mapping[scene_name] / "base.yaml"
    scene_config_path = config_path / file_mapping[scene_name] / f"{scene_name}.yaml"
    base_data = OmegaConf.load(base_config_path)
    OmegaConf.resolve(base_data)
    scene_data = OmegaConf.load(scene_config_path)
    OmegaConf.resolve(scene_data)
    data = OmegaConf.merge(base_data, scene_data)
    data = OmegaConf.to_container(data)
    data.pop("spatial_lr_scale")

    if file_mapping[scene_name] == "conerf":
        data_path = data_path.parent / "CoNeRF/captures-camera-ready"
        data_config = "--data " + str(data_path / scene_name) + " "
    else:
        data_config = "--data " + str(data_path / file_mapping[scene_name].capitalize() / scene_name) + " "
    # data_config = "--data " + str(data_path / file_mapping[scene_name] / scene_name) + " "
    timestamp = "--timestamp " + datetime.now().strftime("%Y-%m-%d_%H%M%S.%f") + " "
    trainer_config = parse_yaml(data)
    trainer_config += f"--pipeline.load-deformable-checkpoint {deform_ckpt} " if deform_ckpt is not None else ""
    dataparser_config = f"freegaussian-{file_mapping[scene_name]}-data "
    dataparser_config += parse_yaml(data["dataparser"] if "dataparser" in data else {})
    print(data_config + timestamp + trainer_config + dataparser_config)
