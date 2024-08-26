from pathlib import Path
from typing import Dict, Tuple, Union

import cv2
import numpy as np

from decode import STATE_DTYPES

STRING_KEYS = [
    "tyre_compound",
    "last_time",
    "best_time",
    "split",
    "current_time",
]


def load_game_state(filepath: str) -> Dict:
    """
    Loads recorded game state np.arrays as a dictionary of observations
        see src.game_capture.state.shared_memory.ac for a list of keys
    """
    with open(filepath, "rb") as file:
        data = file.read()
    return state_bytes_to_dict(data)


def state_bytes_to_dict(data: bytes) -> Dict:
    """
    Converts a game state np.arrays to a dictionary of observations
        see src.game_capture.state.shared_memory for a list of keys
    """
    state_array = np.frombuffer(data, STATE_DTYPES)
    state_dict = {key[0]: value for key, value in zip(STATE_DTYPES, state_array[0])}

    for key in STRING_KEYS:
        state_dict[key] = state_dict[key].tobytes().decode("utf-16").rstrip("\x00")
    return state_dict


def load_image(filepath: Union[Path, str]) -> np.array:
    """
    Loads an image from file.
    :param filepath: Path to image file to be loaded.
    :type filepath: Union[Path,str]
    :return: Image loaded as a numpy array.
    :rtype: np.array
    """
    if isinstance(filepath, Path):
        filepath = str(filepath)
    return cv2.imread(filepath)


def resize_image(image: np.array, dim) -> np.array:
    if dim is None:
        return image
    return cv2.resize(image, dim, interpolation=cv2.INTER_AREA)


def convert_to_zarr(
    input_directory_path: str,
    output_zarr_path: str,
    state_keys: Dict,
    action_keys: Dict,
    meta_keys: Dict,
    image_resize_dim=None,
):
    # TODO: This loads the entire dataset into memory; need to do it in chunks
    import zarr
    import glob

    # Extract ep_ids and ensure same order for data_paths:
    step_ids = sorted(
        int(path.split("/")[-1].split(".")[0])
        for path in glob.glob(f"{input_directory_path}/*.jpeg")
    )
    img_paths = [f"{input_directory_path}/{ep_id}.jpeg" for ep_id in step_ids]
    data_paths = [f"{input_directory_path}/{ep_id}.bin" for ep_id in step_ids]

    zarr_file = zarr.open(output_zarr_path, mode="w")
    meta_group = zarr_file.create_group("meta", overwrite=True)
    data_group = zarr_file.create_group("data", overwrite=True)

    data_group["img"] = [
        resize_image(load_image(path), image_resize_dim) for path in img_paths
    ]

    all_data = [load_game_state(path) for path in data_paths]
    data_group["state"] = np.array([[d[k] for k in state_keys] for d in all_data])
    data_group["action"] = np.array([[d[k] for k in action_keys] for d in all_data])

    meta_group["step_id"] = step_ids
    # Convert dicts to keys, maintaining order just in case:
    meta_data = np.array([[d[k] for k in meta_keys] for d in all_data])
    for i, key in enumerate(meta_keys):
        meta_group[key] = meta_data[:, i]

    episode_ends = np.nonzero(np.diff(meta_group["completed_laps"]))
    episode_ends = np.append(episode_ends, len(step_ids) - 1)
    meta_group["episode_ends"] = episode_ends


if __name__ == "__main__":
    meta_keys = [
        "velocity_x",
        "velocity_y",
        "velocity_z",
        "completed_laps",
    ]

    state_keys = [
        "rpm",
        "speed_kmh",
    ]

    action_keys = [
        "steering_angle",
        "throttle",
        "brake",
        "gear",
    ]

    path = "racing/monza-imitation-learning-data"
    # for train_val in ("val",):
    for train_val in ("train", "val"):
        convert_to_zarr(
            f"{path}/{train_val}",
            f"racing_data_{train_val}.zarr",
            state_keys,
            action_keys,
            meta_keys,
            image_resize_dim=(320, 180),
        )
