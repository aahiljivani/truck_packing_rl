"""MuJoCo-based Gymnasium environment for the Dexterity truck-packing MDP.

See `Dexterity Truck Packing MDP and MuJoCo Gym Env Design.md` for the full
specification and `test.py` for the reference API-only simulation loop this
env mirrors locally (without calling `/place` during `step`).

Named ``mjc.py`` (not ``mujoco.py``) to avoid shadowing the installed
``mujoco`` C-bindings package when Python is run from this directory.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import gymnasium as gym
import mujoco
import mujoco.viewer
import requests
from gymnasium import spaces
import json
import os
import random

API_URL = "https://dexterity.ai/challenge/api"
API_KEY = "dk_dfc2fe80efb446bdc6bf0f8ef2c5cc48c45951d7cc276489a7ae18e687ef2ea5"
PROJECT_ROOT = Path(__file__).resolve().parent
SIM_DETAILS_DIR = PROJECT_ROOT / "sim_details"
DEFAULT_BOX_DIMENSIONS_PATH = SIM_DETAILS_DIR / "box_dimensions.json"
DEFAULT_SIM_XML_PATH = SIM_DETAILS_DIR / "sim.xml"

TRUCK: dict[str, float] = {"depth": 2.0, "width": 2.6, "height": 2.75}

TRUCK_OFFSET: np.ndarray = np.array([1.0, 1.3, 0.0], dtype=np.float64)

# Observation normalization scales. Box dims in ``sim_details/box_dimensions.json``
# are already in [0, 1], so only weight and the reached-extent scalars need
# explicit scaling. ``WEIGHT_SCALE`` is a kg cap used to map ``w_b`` into [0, 1];
# raise it if observed weights routinely exceed this value.
WEIGHT_SCALE: float = 50.0

IDENTITY_DIMS_MAP: dict[tuple[float, float, float, float], Any] = {
    (1.0, 0.0, 0.0, 0.0):     lambda l, w, h: (l, w, h),
    (0.707, 0.0, 0.0, 0.707): lambda l, w, h: (w, l, h),
    (0.707, 0.0, 0.707, 0.0): lambda l, w, h: (h, w, l),
    (0.707, 0.707, 0.0, 0.0): lambda l, w, h: (l, h, w),
}


def json_box_sequence(
    n_boxes: int = 20,
    mode: str = "dev",
    api_key: str = API_KEY,
    base_url: str = API_URL,
    json_path: str | Path = DEFAULT_BOX_DIMENSIONS_PATH,
) -> dict[int, dict[str, Any]]:
    """Sample a box sequence from ``box_dimensions.json``.

    ``mode``/``api_key``/``base_url`` are kept for call-site compatibility
    with the old API-prefetch helper used by ``sac.py``.
    """
    _ = (mode, api_key, base_url)
    path = Path(json_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.exists():
        raise ValueError(
            f"{path} does not exist. Run sim_details/get_boxes.py to generate it."
        )

    with open(path, "r", encoding="utf-8") as f:
        boxes = json.load(f)

    if n_boxes > len(boxes):
        raise ValueError(
            "n_boxes is greater than the number of boxes in sim_details/box_dimensions.json. "
            "Run sim_details/get_boxes.py to generate a larger pool."
        )

    sampled_idx = np.random.randint(0, len(boxes), size=n_boxes)
    return {i: boxes[int(idx)].copy() for i, idx in enumerate(sampled_idx)}


# def prefetch_dimensions_after_start(
#     n_boxes: int,
#     base_url: str,
#     game_start: dict[str, Any],
# ) -> dict[int, dict[str, Any]]:
#     """Walk the dev-mode ``/place`` endpoint to collect ``n_boxes`` specs.

#     Shared by ``fetch_box_sequence`` and ``MujocoTruckEnv._prefetch_box_sequence``.
#     """
#     dimensions: dict[int, dict[str, Any]] = {}
#     x_cursor = 0.0
#     y_cursor = 0.0
#     z_cursor = 0.0
#     game_id = game_start["game_id"]
#     current_box = game_start.get("current_box")
#     counter = 0

#     while current_box is not None and counter < n_boxes:
#         dims = current_box["dimensions"]

#         if y_cursor + dims[1] > TRUCK["width"]:
#             y_cursor = 0.0
#             x_cursor += dims[0]
#         if x_cursor + dims[0] > TRUCK["depth"]:
#             x_cursor = 0.0
#             z_cursor += dims[2]
#         if z_cursor + dims[2] > TRUCK["height"]:
#             break

#         placed_box = current_box
#         place = requests.post(
#             f"{base_url}/place",
#             json={
#                 "game_id": game_id,
#                 "box_id": current_box["id"],
#                 "position": [
#                     x_cursor + dims[0] / 2.0,
#                     y_cursor + dims[1] / 2.0,
#                     dims[2] / 2.0,
#                 ],
#                 "orientation_wxyz": [1, 0, 0, 0],
#             },
#         ).json()
#         if "detail" in place:
#             raise ValueError(f"Place API validation failed: {place['detail']}")

#         current_box = place.get("current_box")
#         weight = placed_box.get("weight") or 1.0
#         dimensions[counter] = {
#             "id": placed_box.get("id"),
#             "dimensions": list(dims),
#             "weight": weight,
#         }
#         counter += 1

#         y_cursor += dims[1]
#         if y_cursor >= TRUCK["width"] - dims[1]:
#             y_cursor = 0.0
#             x_cursor += dims[0]
#             if x_cursor >= TRUCK["depth"] - dims[0]:
#                 x_cursor = 0.0
#                 z_cursor += dims[2]
#                 if z_cursor >= TRUCK["height"] - dims[2]:
#                     z_cursor = 0.0

#     if counter != n_boxes:
#         print(f"Warning: only {counter} boxes were fetched out of {n_boxes}")
#     return dimensions


# def fetch_box_sequence(
#     n_boxes: int = 20,
#     mode: str = "dev",
#     api_key: str = API_KEY,
#     base_url: str = API_URL,
# ) -> dict[int, dict[str, Any]]:
#     """One ``/start`` plus ``n_boxes`` ``/place`` calls. Returns dimensions dict."""
#     start_resp = requests.post(
#         f"{base_url}/start",
#         json={"api_key": api_key, "mode": mode},
#     ).json()
#     if "detail" in start_resp:
#         raise ValueError(f"Start API failed: {start_resp['detail']}")
#     return prefetch_dimensions_after_start(n_boxes, base_url, start_resp)


class MujocoTruckEnv(gym.Env):
    """Gymnasium env that locally simulates truck packing with MuJoCo.

    Observation (shape ``(9 + 3 * grid_x * grid_y,)``, float32, all values in
    ``[0, 1]``; default 57 for a ``4×4`` grid):

        [l, w, h, w_b_norm, max_x_norm, max_y_norm, max_z_norm, density ρ, void v,
         grid_stability flattened row-major (xi fastest),
         grid_count_norm flattened row-major,
         grid_height_norm flattened row-major]

    Scalar normalization: box dims ``l, w, h`` are assumed already in ``[0, 1]``
    (the pooled dimensions in ``sim_details/box_dimensions.json`` satisfy this);
    ``w_b`` is divided by ``WEIGHT_SCALE`` and clipped to ``[0, 1]``; ``max_x``,
    ``max_y``, ``max_z`` are divided by ``TRUCK["depth"]``, ``TRUCK["width"]``,
    ``TRUCK["height"]`` respectively. Raw-unit values are still emitted in
    ``info`` for diagnostics.

    The height grid uses orientation-aware footprint rasterization: each placed
    box contributes its top ``center_z + h_o / 2`` to every cell its oriented
    XY footprint overlaps, normalized by ``TRUCK["height"]``.

    Action (shape (4,), float32):
        [x, y, z, s] in API frame (corner-origin). ``x, y, z`` are the box
        center; ``s`` is a scalar in [-1, 1] discretized into 4 quaternions
        via ``quaternion(s)`` as in ``test.py``.
    """

    metadata = {"render_modes": ["human"], "render_fps": 30}

    TRUCK = TRUCK
    TRUCK_OFFSET = TRUCK_OFFSET
    IDENTITY_DIMS_MAP = IDENTITY_DIMS_MAP

    def __init__(
        self,
        mode: str = "dev",
        api_key: str = API_KEY,
        base_url: str = API_URL,
        n_boxes: int = 100,
        settle_steps: int = 100,
        settle_seconds_max: float = 3.0,
        adaptive_settle: bool = True,
        settle_vel_eps: float = 0.02,
        settle_angvel_eps: float = 0.05,
        settle_consecutive_steps: int = 15,
        max_steps: int = 916,
        alpha: float = 1.0,
        beta: float = 0.0,
        lambda_unstable: float = 0.0,
        stop_density: float = 0.8,
        overlap_tol: float = 1e-4,
        overlap_precheck_penalty: float = 0.0,
        overlap_penalty_scale: float = 0.0,
        settle_drift_margin: float = 0.03,
        x_depth_penalty_scale: float = 0.0,
        drift_penalty_scale: float = 0.0,
        out_of_container_penalty: float = 200.0,
        out_of_container_tol: float = 0.01,
        terminate_on_out_of_container: bool = False,
        grid_x: int = 4,
        grid_y: int = 4,
        grid_count_buffer: float = 1.5,
        grid_disp_thresh_m: float = 0.10,
        sim_xml_path: str | Path = DEFAULT_SIM_XML_PATH,
        render_mode: str | None = None,
        dimensions_pool: list[dict[int, dict[str, Any]]] | None = None,
    ) -> None:
        super().__init__()

        self.mode = mode
        self.api_key = api_key
        self.base_url = base_url
        self.n_boxes = int(n_boxes)
        self.settle_steps = int(settle_steps)
        self.settle_seconds_max = float(settle_seconds_max)
        self.adaptive_settle = bool(adaptive_settle)
        self.settle_vel_eps = float(settle_vel_eps)
        self.settle_angvel_eps = float(settle_angvel_eps)
        self.settle_consecutive_steps = max(1, int(settle_consecutive_steps))
        self.max_steps = int(max_steps) if max_steps is not None else self.n_boxes
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.lambda_unstable = float(lambda_unstable)
        self.stop_density = float(stop_density)
        self.overlap_tol = max(0.0, float(overlap_tol))
        self.overlap_precheck_penalty = float(overlap_precheck_penalty)
        self.overlap_penalty_scale = float(overlap_penalty_scale)
        self.settle_drift_margin = max(0.0, float(settle_drift_margin))
        self.x_depth_penalty_scale = max(0.0, float(x_depth_penalty_scale))
        self.drift_penalty_scale = max(0.0, float(drift_penalty_scale))
        self.out_of_container_penalty = abs(float(out_of_container_penalty))
        self.out_of_container_tol = max(0.0, float(out_of_container_tol))
        self.terminate_on_out_of_container = bool(terminate_on_out_of_container)
        self.grid_x = max(1, int(grid_x))
        self.grid_y = max(1, int(grid_y))
        self.grid_count_buffer = float(grid_count_buffer)
        self.grid_disp_thresh_m = float(grid_disp_thresh_m)
        self.grid_n_max: int = 1
        self.sim_xml_path = Path(sim_xml_path)
        self.render_mode = render_mode
        self._dimensions_pool = dimensions_pool

        obs_dim = 9 + 3 * self.grid_x * self.grid_y
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(obs_dim,),
            dtype=np.float32,
        )

        self.action_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, -1.0], dtype=np.float32),
            high=np.array(
                [TRUCK["depth"], TRUCK["width"], TRUCK["height"], 1.0],
                dtype=np.float32,
            ),
            shape=(4,),
            dtype=np.float32,
        )

        self.model: mujoco.MjModel | None = None
        self.data: mujoco.MjData | None = None
        self.dimensions: dict[int, dict[str, Any]] = {}
        self.game_id: str | None = None

        self.box_qposadr: list[int] = []
        self.box_qveladr: list[int] = []
        self.box_geom_id: list[int] = []
        self.box_body_id: list[int] = []
        self.preview_geom_id: list[int] = []
        self.mocap_ids: list[int] = []
        self.z_floor: list[float] = []
        self.inactive_xyz: list[np.ndarray] = []

        self.current_step: int = 0
        self.placed_indices: list[int] = []
        self.placement_positions: dict[int, np.ndarray] = {}
        self.placed_boxes: list[dict[str, Any]] = []
        self.prev_density: float = 0.0
        self.prev_void: float = 0.0
        self.max_x_reached: float = 0.0
        self.max_y_reached: float = 0.0
        self.max_z_reached: float = 0.0

        self._viewer: Any | None = None
        self._birdseye_transition_id: int = -1
        self._birdseye_cache_tid: int = -2
        self._birdseye_cache: tuple[
            np.ndarray, np.ndarray, np.ndarray, float, float, float, float
        ] | None = None
        # **DO NOT DELETE THIS COMMENT. THESE ARE DEVELOPER NOTES.** I read up to here and I know what is going on.

    def _json_box_sequence(self) -> dict[int, dict[str, Any]]:
        """Sample ``self.n_boxes`` boxes from ``sim_details/box_dimensions.json``.

        Reads a JSON array generated by ``get_boxes.py`` and returns a dict
        keyed by local box index for downstream MuJoCo model construction.
        Sampling is stochastic and allows duplicates.
        """
        json_path = DEFAULT_BOX_DIMENSIONS_PATH
        if not json_path.exists():
            raise ValueError(
                f"{json_path} does not exist. Run sim_details/get_boxes.py to generate it."
            )

        with open(json_path, "r", encoding="utf-8") as f:
            boxes = json.load(f)

        if self.n_boxes > len(boxes):
            raise ValueError(
                "n_boxes is greater than the number of boxes in sim_details/box_dimensions.json. "
                "Run sim_details/get_boxes.py to generate a larger pool."
            )

        sampled_idx = self.np_random.integers(
            0, len(boxes), size=self.n_boxes, endpoint=False
        )
        dimensions = {i: boxes[int(idx)].copy() for i, idx in enumerate(sampled_idx)}
        return dimensions
        

    def _load_boxes(self, dimensions: dict[int, dict[str, Any]]) -> str:
        """Build the per-box XML fragment: a mocap preview + a freejoint body."""
        chunks: list[str] = []
        for i in range(len(dimensions)):
            box = dimensions[i]
            dims = box["dimensions"]
            hx, hy, hz = dims[0] / 2.0, dims[1] / 2.0, dims[2] / 2.0 # half the dimensions of the box
            chunks.append(
                f"""
            <body name="box_{i}_preview" mocap="true" pos="0 0 -500">
                <geom type="box" size="{hx} {hy} {hz}"
                contype="0" conaffinity="0"
                rgba="0.76 0.55 0.30 1"/>
            </body>
            <body name="box_{i}" pos="0 0 -500">
                <freejoint/>
                <geom type="box" size="{hx} {hy} {hz}"
                mass="{box['weight']}"
                friction="0.8 0.005 0.0001"
                rgba="0.76 0.55 0.30 1"/>
            </body>"""
            )
        return "".join(chunks)

    def _load_augmented_xml(self) -> str:
        """Inject per-box bodies in front of the truck XML's ``</worldbody>``."""
        xml_path = self.sim_xml_path
        if not xml_path.is_absolute():
            candidate = PROJECT_ROOT / xml_path
            xml_path = candidate if candidate.exists() else xml_path
        src = xml_path.read_text()
        marker = "</worldbody>"
        if marker not in src:
            raise ValueError(
                f"`{xml_path}` does not contain a </worldbody> tag."
            )
        return src.replace(
            marker, self._load_boxes(self.dimensions) + "\n    " + marker
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Bootstrap a new episode: /start, pre-fetch boxes, build MuJoCo state."""
        super().reset(seed=seed)

        if self._dimensions_pool is not None:
            if not self._dimensions_pool:
                raise ValueError("dimensions_pool is empty")
            idx = int(self.np_random.integers(0, len(self._dimensions_pool)))
            self.dimensions = self._dimensions_pool[idx]
            self.game_id = f"pool:{idx}"
        else:
            start_resp = requests.post(
                f"{self.base_url}/start",
                json={"api_key": self.api_key, "mode": self.mode},
            ).json()
            if "detail" in start_resp:
                raise ValueError(f"Start API failed: {start_resp['detail']}")
            self.game_id = start_resp.get("game_id")
            self.dimensions = self._json_box_sequence()

        boxes_in_episode = len(self.dimensions)
        self.grid_n_max = max(
            1,
            int(
                np.ceil(
                    boxes_in_episode
                    / float(self.grid_x * self.grid_y)
                    * self.grid_count_buffer
                )
            ),
        )

        if self.model is None:
            # First reset: build MuJoCo model/data and cache address lookups.
            # Subsequent resets mutate the existing model in place so the
            # passive viewer (bound to this model/data) stays valid.
            aug_xml = self._load_augmented_xml()
            self.model = mujoco.MjModel.from_xml_string(aug_xml)
            self.data = mujoco.MjData(self.model)

            inactive_z_base = -400.0
            inactive_dz = 0.02

            self.box_qposadr = []
            self.box_qveladr = []
            self.box_geom_id = []
            self.box_body_id = []
            self.preview_geom_id = []
            self.mocap_ids = []
            self.inactive_xyz = []
            self.z_floor = [0.0] * self.n_boxes

            for i in range(self.n_boxes):
                bid = mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_BODY, f"box_{i}"
                )
                if bid < 0:
                    raise RuntimeError(f"Body box_{i} not found.")
                mid_body = mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_BODY, f"box_{i}_preview"
                )
                if mid_body < 0:
                    raise RuntimeError(f"Body box_{i}_preview not found.")
                mocap_id = int(self.model.body_mocapid[mid_body])
                if mocap_id < 0:
                    raise RuntimeError(f"Body box_{i}_preview is not a mocap body.")

                jid = self.model.body_jntadr[bid]
                gid = self.model.body_geomadr[bid]
                pgid = self.model.body_geomadr[mid_body]
                self.box_qposadr.append(int(self.model.jnt_qposadr[jid]))
                self.box_qveladr.append(int(self.model.jnt_dofadr[jid]))
                self.box_geom_id.append(int(gid))
                self.box_body_id.append(int(bid))
                self.preview_geom_id.append(int(pgid))
                self.mocap_ids.append(mocap_id)
                self.inactive_xyz.append(
                    np.array(
                        [0.0, 0.0, inactive_z_base + i * inactive_dz],
                        dtype=float,
                    )
                )

        # Update per-episode box geometry/mass in place so the viewer's
        # bound (model, data) pair does not need to be re-created.
        for i in range(self.n_boxes):
            dims = self.dimensions[i]["dimensions"]
            hx = float(dims[0]) / 2.0
            hy = float(dims[1]) / 2.0
            hz = float(dims[2]) / 2.0
            weight = float(self.dimensions[i]["weight"])

            self.model.geom_size[self.box_geom_id[i]] = [hx, hy, hz]
            self.model.geom_size[self.preview_geom_id[i]] = [hx, hy, hz]

            bid = self.box_body_id[i]
            self.model.body_mass[bid] = weight
            # Uniform-density box inertia about principal axes (half-extents hx,hy,hz).
            self.model.body_inertia[bid] = [
                weight / 3.0 * (hy * hy + hz * hz),
                weight / 3.0 * (hx * hx + hz * hz),
                weight / 3.0 * (hx * hx + hy * hy),
            ]
            self.z_floor[i] = hz

        # Park every box inactive: disable contacts, seat qpos under the floor,
        # zero velocities, and hide the mocap preview below the floor too.
        identity_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
        for i in range(self.n_boxes):
            gid = self.box_geom_id[i]
            self.model.geom_contype[gid] = 0
            self.model.geom_conaffinity[gid] = 0

            qpa = self.box_qposadr[i]
            qva = self.box_qveladr[i]
            self.data.qpos[qpa : qpa + 3] = self.inactive_xyz[i]
            self.data.qpos[qpa + 3 : qpa + 7] = identity_quat
            self.data.qvel[qva : qva + 6] = 0.0

            mocap_id = self.mocap_ids[i]
            self.data.mocap_pos[mocap_id] = self.inactive_xyz[i]
            self.data.mocap_quat[mocap_id] = identity_quat

        mujoco.mj_forward(self.model, self.data)

        self.current_step = 0
        self.placed_indices = []
        self.placement_positions = {}
        self.placed_boxes = []
        self.prev_density = 0.0
        self.prev_void = 0.0
        self.max_x_reached = 0.0
        self.max_y_reached = 0.0
        self.max_z_reached = 0.0
        self._birdseye_transition_id = 0
        self._birdseye_cache_tid = -2
        self._birdseye_cache = None

        if self.render_mode == "human":
            if self._viewer is None:
                self._viewer = mujoco.viewer.launch_passive(
                    self.model, self.data
                )
            if self._viewer is not None and self._viewer.is_running():
                self._viewer.sync()

        return self._get_obs(), self._get_info()

    def _get_obs(self) -> np.ndarray:
        """Assemble the observation vector (normalized scalars + bird's-eye grids).

        All scalars are mapped into ``[0, 1]``: box dims ``l, w, h`` are assumed
        already normalized by the pool; ``w_b`` is divided by ``WEIGHT_SCALE``
        and clipped; ``max_*`` extents are divided by the corresponding
        ``TRUCK`` axis; density and void are already ``[0, 1]``. Grids
        (stability, count, height) are already ``[0, 1]``.
        """
        if self.current_step < len(self.dimensions):
            l, w, h = self.dimensions[self.current_step]["dimensions"]
            w_b = float(self.dimensions[self.current_step]["weight"])
        else:
            l = w = h = 0.0
            w_b = 0.0
        rho = float(self._compute_density())
        v = float(self._compute_void_ratio())
        stab, cnorm, hnorm, _, _, _, _ = self._compute_birdseye_grids()

        w_b_norm = min(float(w_b) / WEIGHT_SCALE, 1.0)
        max_x_norm = float(self.max_x_reached) / TRUCK["depth"]
        max_y_norm = float(self.max_y_reached) / TRUCK["width"]
        max_z_norm = float(self.max_z_reached) / TRUCK["height"]

        return np.concatenate(
            [
                np.array(
                    [
                        float(l),
                        float(w),
                        float(h),
                        w_b_norm,
                        max_x_norm,
                        max_y_norm,
                        max_z_norm,
                        rho,
                        v,
                    ],
                    dtype=np.float32,
                ),
                stab,
                cnorm,
                hnorm,
            ],
            axis=0,
        )

    def _get_info(self) -> dict[str, Any]:
        """Diagnostic info dict used by reset() and step().

        Keys match the "Logging and Diagnostics" section of the design doc:
        ``game_id``, ``boxes_remaining``, ``reason`` /
        ``termination_reason``, max extents, ``density``, ``void``,
        ``n_displaced``, ``avg_disp``, and ``clipped``. Step-specific values
        (reason, clipped, n_displaced, avg_disp, curr density/void) are
        overwritten in ``step`` after the physics settle.
        """
        return {
            "game_id": self.game_id,
            "boxes_remaining": self.n_boxes - self.current_step,
            "reason": None,
            "termination_reason": None,
            "max_x_reached": float(self.max_x_reached),
            "max_y_reached": float(self.max_y_reached),
            "max_z_reached": float(self.max_z_reached),
            "density": float(self.prev_density),
            "void": float(self.prev_void),
            "n_displaced": 0,
            "avg_disp": 0.0,
            "clipped": False,
        }

    def _quaternion(self, s: float) -> np.ndarray:
        """Discretize a scalar in [-1, 1] into one of 4 axis-aligned quaternions.

        Mirrors ``Game.quaternion`` in ``test.py`` (lines 162-173); bin
        boundaries are unchanged.
        """
        s = float(s)
        if -1.0 <= s <= -0.49:
            return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
        elif -0.49 < s <= 0.01:
            return np.array([0.707, 0.0, 0.0, 0.707], dtype=float)
        elif 0.01 < s <= 0.5:
            return np.array([0.707, 0.0, 0.707, 0.0], dtype=float)
        elif 0.5 < s <= 1.0:
            return np.array([0.707, 0.707, 0.0, 0.0], dtype=float)
        else:
            raise ValueError(f"Invalid orientation scalar: {s}")

    def _get_box_position(self, i: int) -> np.ndarray:
        """Return the MuJoCo-frame position of box ``i`` from ``data.qpos``."""
        qpa = self.box_qposadr[i]
        return self.data.qpos[qpa : qpa + 3].copy()

    def _check_stability(self) -> tuple[int, float]:
        """Count previously-placed boxes displaced > 10 cm from their settle pose."""
        displaced: list[float] = []
        for j in self.placed_indices[:-1]:
            current_pos = self._get_box_position(j)
            recorded_pos = self.placement_positions[j]
            dist = float(np.linalg.norm(current_pos - recorded_pos))
            if dist > 0.10:
                displaced.append(dist)
        n = len(displaced)
        avg = float(np.mean(displaced)) if displaced else 0.0
        return n, avg

    def _update_max_extents(self, l: float, w: float, h: float, quat: np.ndarray) -> None:
        """Fold a placed box's oriented dims into the running max-extents trackers."""
        key = tuple(float(x) for x in quat)
        l_o, w_o, h_o = IDENTITY_DIMS_MAP[key](float(l), float(w), float(h))
        if l_o > self.max_x_reached:
            self.max_x_reached = float(l_o)
        if w_o > self.max_y_reached:
            self.max_y_reached = float(w_o)
        if h_o > self.max_z_reached:
            self.max_z_reached = float(h_o)

    def _compute_density(self) -> float:
        """Occupied volume / (max_x * truck_width * truck_height). Matches game."""
        if len(self.placed_indices) == 0:
            return 0.0
        sum_vol = 0.0
        for b in self.placed_boxes:
            dx, dy, dz = b["dimensions"]
            sum_vol += float(dx) * float(dy) * float(dz)
        denom = float(self.max_x_reached) * TRUCK["width"] * TRUCK["height"]
        if denom <= 0.0:
            return 0.0
        return float(sum_vol / denom)

    def _compute_void_ratio(self) -> float:
        """1 - occupied / AABB(placed_boxes), clipped to [0, 1]."""
        if len(self.placed_indices) == 0:
            return 0.0
        if len(self.placed_boxes) < 2:
            return 0.0
        mins: list[np.ndarray] = []
        maxs: list[np.ndarray] = []
        occupied = 0.0
        for b in self.placed_boxes:
            pos = np.asarray(b["position"], dtype=float)
            quat = b["orientation_wxyz"]
            key = tuple(float(x) for x in quat)
            l, w, h = b["dimensions"]
            l_o, w_o, h_o = IDENTITY_DIMS_MAP[key](float(l), float(w), float(h))
            half = np.array([l_o / 2.0, w_o / 2.0, h_o / 2.0], dtype=float)
            mins.append(pos - half)
            maxs.append(pos + half)
            occupied += l_o * w_o * h_o
        bb_min = np.min(np.stack(mins, axis=0), axis=0)
        bb_max = np.max(np.stack(maxs, axis=0), axis=0)
        bounding = float(np.prod(bb_max - bb_min))
        if bounding <= 0.0:
            return 0.0
        v = 1.0 - occupied / bounding
        return float(np.clip(v, 0.0, 1.0))

    def _compute_birdseye_grids(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float, float, float]:
        """Per-cell stability, counts, and max top height (bird's-eye).

        Stability and count use the box center for cell assignment (matching the
        original convention). Height uses footprint rasterization: each placed
        box's oriented XY footprint ``(l_o, w_o)`` contributes its top
        ``center_z + h_o / 2`` to every cell it overlaps, normalized by
        ``TRUCK["height"]``. Oriented dims come from
        ``IDENTITY_DIMS_MAP[orientation_wxyz]``.

        Row-major flattening: ``k = xi * grid_y + yi``.

        Returns:
            (stability, count_norm, height_norm, stab_mean, cell_disp_max,
             counts_sum, height_max)
        """
        if (
            self._birdseye_cache is not None
            and self._birdseye_cache_tid == self._birdseye_transition_id
        ):
            return self._birdseye_cache

        gx, gy = self.grid_x, self.grid_y
        n_cells = gx * gy
        sum_d = np.zeros((gx, gy), dtype=np.float64)
        cnt_d = np.zeros((gx, gy), dtype=np.int32)
        counts = np.zeros((gx, gy), dtype=np.int32)
        height_mat = np.zeros((gx, gy), dtype=np.float64)

        if self.model is None or self.data is None or len(self.placed_indices) == 0:
            stability = np.ones(n_cells, dtype=np.float32)
            count_norm = np.zeros(n_cells, dtype=np.float32)
            height_norm = np.zeros(n_cells, dtype=np.float32)
            out = (stability, count_norm, height_norm, 1.0, 0.0, 0.0, 0.0)
            self._birdseye_cache = out
            self._birdseye_cache_tid = self._birdseye_transition_id
            return out

        cell_dx = float(TRUCK["depth"]) / float(gx)
        cell_dy = float(TRUCK["width"]) / float(gy)
        truck_h = float(TRUCK["height"])
        n_max = max(1, int(self.grid_n_max))

        for j, b in zip(self.placed_indices, self.placed_boxes):
            if j not in self.placement_positions:
                continue
            cur_mj = self._get_box_position(j)
            rec_mj = self.placement_positions[j]
            dist = float(np.linalg.norm(cur_mj - rec_mj))

            pos_api = cur_mj + TRUCK_OFFSET
            cx_api = float(pos_api[0])
            cy_api = float(pos_api[1])
            cz_api = float(pos_api[2])

            # Center-based bucketing for stability and count (unchanged).
            xi = int(np.clip(cx_api / cell_dx, 0, gx - 1))
            yi = int(np.clip(cy_api / cell_dy, 0, gy - 1))
            sum_d[xi, yi] += dist
            cnt_d[xi, yi] += 1
            counts[xi, yi] += 1

            # Orientation-aware oriented dims for footprint rasterization.
            quat_key = tuple(float(x) for x in b["orientation_wxyz"])
            l, w, h = (float(x) for x in b["dimensions"])
            mapper = IDENTITY_DIMS_MAP.get(quat_key)
            if mapper is None:
                l_o, w_o, h_o = l, w, h
            else:
                l_o, w_o, h_o = mapper(l, w, h)

            half_lo = l_o / 2.0
            half_wo = w_o / 2.0
            half_ho = h_o / 2.0
            top = cz_api + half_ho

            xi_lo = int(np.clip(np.floor((cx_api - half_lo) / cell_dx), 0, gx - 1))
            xi_hi = int(np.clip(np.floor((cx_api + half_lo) / cell_dx), 0, gx - 1))
            yi_lo = int(np.clip(np.floor((cy_api - half_wo) / cell_dy), 0, gy - 1))
            yi_hi = int(np.clip(np.floor((cy_api + half_wo) / cell_dy), 0, gy - 1))
            sub = height_mat[xi_lo : xi_hi + 1, yi_lo : yi_hi + 1]
            np.maximum(sub, top, out=sub)

        mean_d = np.zeros((gx, gy), dtype=np.float64)
        mask = cnt_d > 0
        mean_d[mask] = sum_d[mask] / cnt_d[mask].astype(np.float64)
        d_thresh = max(1e-9, float(self.grid_disp_thresh_m))
        stab_mat = np.clip(1.0 - mean_d / d_thresh, 0.0, 1.0).astype(np.float32)
        count_mat = np.clip(counts.astype(np.float64) / float(n_max), 0.0, 1.0).astype(
            np.float32
        )
        height_mat_norm = np.clip(height_mat / max(truck_h, 1e-9), 0.0, 1.0).astype(
            np.float32
        )

        # Row-major: xi varies fastest across the flattened vector.
        stability = stab_mat.reshape(-1, order="C")
        count_norm = count_mat.reshape(-1, order="C")
        height_norm = height_mat_norm.reshape(-1, order="C")

        stab_mean = float(np.mean(stability)) if stability.size else 1.0
        cell_disp_max = float(np.max(mean_d)) if mean_d.size else 0.0
        counts_sum = float(np.sum(counts))
        height_max = float(np.max(height_mat)) if height_mat.size else 0.0
        out = (
            stability,
            count_norm,
            height_norm,
            stab_mean,
            cell_disp_max,
            counts_sum,
            height_max,
        )
        self._birdseye_cache = out
        self._birdseye_cache_tid = self._birdseye_transition_id
        return out

    def _compute_compactness_proxy(self) -> float:
        """Occupied volume / current reached prism volume, clipped to [0, 1]."""
        if not self.placed_boxes:
            return 0.0
        occupied = 0.0
        for b in self.placed_boxes:
            dx, dy, dz = b["dimensions"]
            occupied += float(dx) * float(dy) * float(dz)
        prism = (
            float(self.max_x_reached)
            * float(self.max_y_reached)
            * float(self.max_z_reached)
        )
        if prism <= 0.0:
            return 0.0
        return float(np.clip(occupied / prism, 0.0, 1.0))

    @staticmethod
    def _aabb_overlap_volume(
        center_a: np.ndarray,
        half_a: np.ndarray,
        center_b: np.ndarray,
        half_b: np.ndarray,
    ) -> float:
        """Return AABB intersection volume for two axis-aligned boxes."""
        delta = np.abs(np.asarray(center_a, dtype=float) - np.asarray(center_b, dtype=float))
        overlap = (np.asarray(half_a, dtype=float) + np.asarray(half_b, dtype=float)) - delta
        overlap = np.clip(overlap, 0.0, None)
        return float(np.prod(overlap))

    def _compute_overlap_volume_with_placed(
        self,
        center_api: np.ndarray,
        dims_oriented: tuple[float, float, float],
        *,
        include_current: bool = False,
    ) -> float:
        """Compute overlap volume of a candidate box against placed boxes."""
        cand_half = np.array(dims_oriented, dtype=float) / 2.0
        total_overlap = 0.0
        n_placed = len(self.placed_boxes)
        if n_placed == 0:
            return 0.0

        limit = n_placed if include_current else max(0, n_placed - 1)
        for b in self.placed_boxes[:limit]:
            pos = np.asarray(b["position"], dtype=float)
            quat = b["orientation_wxyz"]
            key = tuple(float(x) for x in quat)
            l, w, h = b["dimensions"]
            l_o, w_o, h_o = IDENTITY_DIMS_MAP[key](float(l), float(w), float(h))
            other_half = np.array([l_o, w_o, h_o], dtype=float) / 2.0
            total_overlap += self._aabb_overlap_volume(center_api, cand_half, pos, other_half)
        return float(total_overlap)

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Place the current box in local simulation and return an RL transition.

        Implements the two-stage OOB handling (clip per axis, then bail only
        if some axis interval is empty), physics settle, stability /
        density_threshold / complete / time_limit termination ladder, and the
        shaped reward described in the design doc.
        """
        if self.model is None or self.data is None:
            raise RuntimeError("Env not initialized; call reset() before step().")

        i = self.current_step
        if i >= self.n_boxes:
            raise RuntimeError(
                f"step() called after all {self.n_boxes} boxes were placed."
            )

        action_arr = np.asarray(action, dtype=float).reshape(-1)
        if action_arr.shape[0] < 4:
            raise ValueError(
                f"action must have at least 4 components, got shape {action_arr.shape}"
            )

        box = self.dimensions[i]
        l, w, h = (float(x) for x in box["dimensions"])

        s_scalar = float(np.clip(action_arr[3], -1.0, 1.0))
        quat = self._quaternion(s_scalar)
        key = tuple(float(x) for x in quat)
        l_o, w_o, h_o = IDENTITY_DIMS_MAP[key](l, w, h)

        lo_x, hi_x = l_o / 2.0, TRUCK["depth"] - l_o / 2.0
        lo_y, hi_y = w_o / 2.0, TRUCK["width"] - w_o / 2.0
        lo_z, hi_z = h_o / 2.0, TRUCK["height"] - h_o / 2.0

        if lo_x > hi_x or lo_y > hi_y or lo_z > hi_z:
            info = self._get_info()
            info["clipped"] = False
            info["reason"] = "out_of_bounds"
            info["termination_reason"] = "out_of_bounds"
            return self._get_obs(), -50.0, True, False, info

        requested_pos = np.array(
            [float(action_arr[0]), float(action_arr[1]), float(action_arr[2])],
            dtype=float,
        )
        pos = np.array(
            [
                np.clip(requested_pos[0], lo_x, hi_x),
                np.clip(requested_pos[1], lo_y, hi_y),
                np.clip(requested_pos[2], lo_z, hi_z),
            ],
            dtype=float,
        )
        clipped = bool(np.any(pos != requested_pos))

        mj_xyz = pos - TRUCK_OFFSET
        precheck_overlap = self._compute_overlap_volume_with_placed(
            pos,
            (l_o, w_o, h_o),
            include_current=True,
        )
        precheck_penalty = 0.0
        if precheck_overlap > self.overlap_tol:
            precheck_penalty = abs(self.overlap_precheck_penalty)

        qpa = self.box_qposadr[i]
        qva = self.box_qveladr[i]
        self.data.qpos[qpa : qpa + 3] = mj_xyz
        self.data.qpos[qpa + 3 : qpa + 7] = quat
        self.data.qvel[qva : qva + 6] = 0.0

        gid = self.box_geom_id[i]
        self.model.geom_contype[gid] = 1
        self.model.geom_conaffinity[gid] = 1

        mocap_id = self.mocap_ids[i]
        self.data.mocap_pos[mocap_id] = self.inactive_xyz[i]
        self.data.mocap_quat[mocap_id] = np.array(
            [1.0, 0.0, 0.0, 0.0], dtype=float
        )

        model_dt = float(self.model.opt.timestep)
        if model_dt <= 0.0:
            max_settle_steps = max(1, self.settle_steps)
        else:
            max_settle_steps = max(
                1, int(np.ceil(self.settle_seconds_max / model_dt))
            )

        stable_steps = 0
        settle_steps_used = 0
        for _ in range(max_settle_steps):
            mujoco.mj_step(self.model, self.data)
            settle_steps_used += 1
            if self.adaptive_settle:
                lin_vel = self.data.qvel[qva : qva + 3]
                ang_vel = self.data.qvel[qva + 3 : qva + 6]
                if (
                    float(np.linalg.norm(lin_vel)) <= self.settle_vel_eps
                    and float(np.linalg.norm(ang_vel)) <= self.settle_angvel_eps
                ):
                    stable_steps += 1
                    if stable_steps >= self.settle_consecutive_steps:
                        break
                else:
                    stable_steps = 0
            if self._viewer is not None and self._viewer.is_running():
                self._viewer.sync()
        mujoco.mj_forward(self.model, self.data)
        if self._viewer is not None and self._viewer.is_running():
            self._viewer.sync()

        settled_mj = self._get_box_position(i)
        self.placement_positions[i] = settled_mj.copy()
        self.placed_indices.append(i)
        settled_api = settled_mj + TRUCK_OFFSET
        self.placed_boxes.append(
            {
                "id": box.get("id"),
                "position": [
                    float(settled_api[0]),
                    float(settled_api[1]),
                    float(settled_api[2]),
                ],
                "orientation_wxyz": quat,
                "dimensions": [l, w, h],
            }
        )
        self._birdseye_transition_id += 1

        n_displaced, avg_disp = self._check_stability()

        self._update_max_extents(l, w, h, quat)
        curr_density = self._compute_density()
        curr_void = self._compute_void_ratio()
        curr_compactness = self._compute_compactness_proxy()
        post_settle_overlap = self._compute_overlap_volume_with_placed(
            settled_api,
            (l_o, w_o, h_o),
            include_current=False,
        )
        settle_drift = float(np.linalg.norm(settled_api - pos))

        settled_half = np.array([l_o / 2.0, w_o / 2.0, h_o / 2.0], dtype=float)
        settled_min = settled_api - settled_half
        settled_max = settled_api + settled_half
        tol = self.out_of_container_tol
        out_of_container = bool(
            settled_min[0] < -tol
            or settled_min[1] < -tol
            or settled_min[2] < -tol
            or settled_max[0] > TRUCK["depth"] + tol
            or settled_max[1] > TRUCK["width"] + tol
            or settled_max[2] > TRUCK["height"] + tol
        )

        density_gain = float(curr_density - self.prev_density)
        # Reward is density gain only. Stability, overlap, drift, and depth
        # shaping are all handled implicitly: unstable stacks terminate the
        # episode (no more density rewards), and the density denominator
        # (max_x_reached * truck_width * truck_height) already penalizes
        # pushing boxes deeper into the truck or letting them drift.
        density_term = self.alpha * 100.0 * density_gain
        x_depth_normalized = float(
            np.clip(settled_api[0] / max(1e-9, float(TRUCK["depth"])), 0.0, 1.0)
        )
        x_depth_penalty = 0.0
        stability_term = 0.0
        void_compact_term = 0.0
        overlap_penalty = 0.0
        drift_exceeds_margin = False
        drift_penalty = 0.0
        step_reward = density_term
        reward = float(step_reward)

        terminated = False
        truncated = False
        termination_reason: str | None = None
        # Precedence: invalid/hard outcomes -> density/complete success -> time limit.
        if out_of_container and self.terminate_on_out_of_container:
            reward -= self.out_of_container_penalty
            terminated = True
            termination_reason = "out_of_container"
        elif n_displaced >= 3:
            reward -= 10.0
            terminated = True
            termination_reason = "unstable"
        elif curr_density >= self.stop_density:
            reward += curr_density * (1.0 - curr_void) * 500.0
            terminated = True
            termination_reason = "density_threshold"
        elif self.current_step + 1 >= self.n_boxes:
            reward += curr_density * (1.0 - curr_void) * 500.0
            terminated = True
            termination_reason = "complete"

        self.prev_density = float(curr_density)
        self.prev_void = float(curr_void)

        if not terminated:
            self.current_step += 1
            if self.current_step >= self.max_steps:
                truncated = True
                termination_reason = "time_limit"

        info = self._get_info()
        info["clipped"] = clipped
        info["n_displaced"] = int(n_displaced)
        info["avg_disp"] = float(avg_disp)
        info["density"] = float(curr_density)
        info["void"] = float(curr_void)
        info["compactness"] = float(curr_compactness)
        info["reward_density_term"] = float(density_term)
        info["reward_x_depth_penalty"] = float(-x_depth_penalty)
        info["x_depth_normalized"] = float(x_depth_normalized)
        info["reward_stability_term"] = float(stability_term)
        info["reward_void_compact_term"] = float(void_compact_term)
        info["max_x_reached"] = float(self.max_x_reached)
        info["max_y_reached"] = float(self.max_y_reached)
        info["max_z_reached"] = float(self.max_z_reached)
        info["boxes_remaining"] = self.n_boxes - self.current_step
        info["settle_steps_used"] = int(settle_steps_used)
        info["settle_steps_max"] = int(max_settle_steps)
        info["overlap_precheck_volume"] = float(precheck_overlap)
        info["overlap_precheck_penalty"] = float(precheck_penalty)
        info["overlap_post_settle_volume"] = float(post_settle_overlap)
        info["overlap_penalty"] = float(overlap_penalty)
        info["settle_drift"] = float(settle_drift)
        info["settle_drift_margin"] = float(self.settle_drift_margin)
        info["drift_exceeds_margin"] = drift_exceeds_margin
        info["drift_penalty"] = float(drift_penalty)
        info["out_of_container"] = out_of_container
        (
            _stab,
            _cnorm,
            _hnorm,
            grid_stab_mean,
            grid_cell_disp_max,
            grid_counts_sum,
            grid_height_max,
        ) = self._compute_birdseye_grids()
        info["grid_stability_mean"] = float(grid_stab_mean)
        info["grid_cell_disp_max"] = float(grid_cell_disp_max)
        info["grid_counts_sum"] = float(grid_counts_sum)
        info["grid_n_max"] = int(self.grid_n_max)
        info["grid_height_max"] = float(grid_height_max)
        if precheck_penalty > 0.0 and termination_reason is None:
            info["reason"] = "overlap_precheck_penalty"
        if termination_reason is not None:
            info["reason"] = termination_reason
            info["termination_reason"] = termination_reason

        return self._get_obs(), reward, terminated, truncated, info

    def render(self) -> None:
        """Sync the passive MuJoCo viewer when ``render_mode='human'``.

        Mirrors the ``viewer.sync()`` call pattern used in ``test.py`` line
        366. For any other ``render_mode`` this is a no-op.
        """
        if self.render_mode != "human":
            return
        if self._viewer is not None and self._viewer.is_running():
            self._viewer.sync()

    def close(self) -> None:
        """Close the passive viewer if one was launched."""
        if self._viewer is not None:
            try:
                self._viewer.close()
            except Exception:
                pass
            self._viewer = None
# DO NOT DELETE THIS COMMENT. THESE ARE DEVELOPER NOTES.** I read up to here and I know what is going on.


def _run_smoke_test(
    check_env: bool = False,
    render_mode: str | None = None,
) -> None:
    """Random-action rollout used as a quick sanity check for the env.

    Constructs a small ``MujocoTruckEnv`` in dev mode and steps it with
    uniformly sampled actions until the episode ends, printing the
    per-step reward, density, void ratio, and termination reason. Pass
    ``render_mode='human'`` to attach the MuJoCo passive viewer (same
    viewer that ``test.py`` uses).
    """
    env = MujocoTruckEnv(mode="dev", n_boxes=20, render_mode=render_mode)

    if check_env:
        from gymnasium.utils.env_checker import check_env as _check_env

        _check_env(env.unwrapped)
        print("[smoke] gymnasium env_checker passed.")

    try:
        obs, info = env.reset(seed=0)
        print(
            f"[smoke] reset: game_id={info.get('game_id')} "
            f"boxes_remaining={info.get('boxes_remaining')} obs_shape={obs.shape}"
        )

        rng = np.random.default_rng(0)
        total_reward = 0.0
        terminated = False
        truncated = False
        for t in range(env.n_boxes):
            if (
                render_mode == "human"
                and env._viewer is not None
                and not env._viewer.is_running()
            ):
                print("[smoke] viewer closed by user; stopping rollout.")
                break
            action = rng.uniform(
                low=env.action_space.low,
                high=env.action_space.high,
            ).astype(np.float32)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            print(
                f"[smoke] step={t:03d} reward={reward:+.4f} "
                f"density={info.get('density', 0.0):.4f} "
                f"void={info.get('void', 0.0):.4f} "
                f"reason={info.get('reason')!r} "
                f"clipped={info.get('clipped')} "
                f"n_displaced={info.get('n_displaced')}"
            )
            if terminated or truncated:
                break

        print(
            f"[smoke] done: total_reward={total_reward:+.4f} "
            f"final_reason={info.get('reason')!r} "
            f"terminated={terminated} truncated={truncated}"
        )
    finally:
        env.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Run gymnasium.utils.env_checker.check_env before the rollout.",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Launch the MuJoCo passive viewer (equivalent to render_mode='human').",
    )
    args = parser.parse_args()

    _run_smoke_test(
        check_env=args.check,
        render_mode="human" if args.render else None,
    )
