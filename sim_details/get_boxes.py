import requests
from typing import Any
import numpy as np
import json
import os
import argparse
from pathlib import Path

API_URL = "https://dexterity.ai/challenge/api"
API_KEY = "dk_dfc2fe80efb446bdc6bf0f8ef2c5cc48c45951d7cc276489a7ae18e687ef2ea5"
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_PATH = SCRIPT_DIR / "box_dimensions.json"
TRUCK = {'depth' : 2.0, 'width' : 2.6, 'height' : 2.75} # truck dimensions in meters so volume = 14.3 m^3

def json_box_dimensions(
    dimensions: dict[int, dict[str, Any]],
    path: str | Path = DEFAULT_OUTPUT_PATH,
) -> None:
    """Append box records to ``path`` as one JSON array (``json.load`` compatible).

    Reads the existing array if present, extends it with this fetch, and writes
    the full array back so repeated runs keep valid JSON.
    """
    path = Path(path)
    if not path.is_absolute():
        path = SCRIPT_DIR / path

    new_records = [dimensions[i] for i in sorted(dimensions.keys())]

    existing: list[dict[str, Any]] = []
    if os.path.exists(path) and os.path.getsize(path) > 0:
        with open(path, encoding="utf-8") as f:
            try:
                loaded = json.load(f)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"{path} is not valid JSON; remove it or fix it before re-running."
                ) from e
        if isinstance(loaded, list):
            existing = loaded
        else:
            existing = [loaded]

    existing.extend(new_records)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2)
        f.write("\n")




def prefetch_dimensions_after_start(
    n_boxes: int,
    base_url: str,
    game_start: dict[str, Any],
) -> dict[int, dict[str, Any]]:
    """Walk the dev-mode ``/place`` endpoint to collect ``n_boxes`` specs.

    Shared by ``fetch_box_sequence`` and ``MujocoTruckEnv._prefetch_box_sequence``.
    """
    dimensions: dict[int, dict[str, Any]] = {}
    x_cursor = 0.0
    y_cursor = 0.0
    z_cursor = 0.0
    game_id = game_start["game_id"]
    current_box = game_start.get("current_box")
    counter = 0

    while current_box is not None and counter < n_boxes:
        dims = current_box["dimensions"]

        if y_cursor + dims[1] > TRUCK["width"]:
            y_cursor = 0.0
            x_cursor += dims[0]
        if x_cursor + dims[0] > TRUCK["depth"]:
            x_cursor = 0.0
            z_cursor += dims[2]
        if z_cursor + dims[2] > TRUCK["height"]:
            break

        placed_box = current_box
        place = requests.post(
            f"{base_url}/place",
            json={
                "game_id": game_id,
                "box_id": current_box["id"],
                "position": [
                    x_cursor + dims[0] / 2.0,
                    y_cursor + dims[1] / 2.0,
                    dims[2] / 2.0,
                ],
                "orientation_wxyz": [1, 0, 0, 0],
            },
        ).json()
        if "detail" in place:
            raise ValueError(f"Place API validation failed: {place['detail']}")

        current_box = place.get("current_box")
        weight = placed_box.get("weight") or 1.0
        dimensions[counter] = {
            "id": placed_box.get("id"),
            "dimensions": list(dims),
            "weight": weight,
        }
        counter += 1

        y_cursor += dims[1]
        if y_cursor >= TRUCK["width"] - dims[1]:
            y_cursor = 0.0
            x_cursor += dims[0]
            if x_cursor >= TRUCK["depth"] - dims[0]:
                x_cursor = 0.0
                z_cursor += dims[2]
                if z_cursor >= TRUCK["height"] - dims[2]:
                    z_cursor = 0.0

    if counter != n_boxes:
        print(f"Warning: only {counter} boxes were fetched out of {n_boxes}")
    return dimensions


def fetch_box_sequence(
    n_boxes: int = 20,
    mode: str = "dev",
    api_key: str = API_KEY,
    base_url: str = API_URL,
) -> dict[int, dict[str, Any]]:
    """One ``/start`` plus ``n_boxes`` ``/place`` calls. Returns dimensions dict."""
    start_resp = requests.post(
        f"{base_url}/start",
        json={"api_key": api_key, "mode": mode},
    ).json()
    if "detail" in start_resp:
        raise ValueError(f"Start API failed: {start_resp['detail']}")
    return prefetch_dimensions_after_start(n_boxes, base_url, start_resp)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_boxes', type=int, default=100)
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT_PATH),
        help="Path to write the box dimension pool JSON.",
    )
    args = parser.parse_args()
    dimensions = fetch_box_sequence(n_boxes=args.n_boxes)
    json_box_dimensions(dimensions, path=args.output)





if __name__== "__main__":
    main()

