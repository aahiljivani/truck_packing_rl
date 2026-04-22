# api key: dk_dfc2fe80efb446bdc6bf0f8ef2c5cc48c45951d7cc276489a7ae18e687ef2ea5
# og_api_key = "dk_941a358f18bf036efe699cde6b2f7cdb93b117975288a27011fa9ee17b06c084"
import requests
import numpy as np
import gymnasium as gym
from gymnasium import spaces


def _api_response_failed(resp: dict) -> bool:
    if not isinstance(resp, dict):
        return True
    if "detail" in resp:
        return True
    st = resp.get("status")
    # Official flow: "ok" while playing; game-over payloads use top-level "terminated" with full fields.
    if st == "terminated":
        return False
    if st is not None and st != "ok":
        return True
    return False


class Gym(gym.Env):

    def __init__(
        self,
        mode="dev",
        BASE="https://dexterity.ai/challenge/api",
        api_key="dk_941a358f18bf036efe699cde6b2f7cdb93b117975288a27011fa9ee17b06c084",
    ):
        super().__init__()
        self.mode = mode
        self.BASE = BASE
        self.api_key = api_key
        self.total_boxes = 916
        self.max_placed_slots = 50
        self.pos_space = 3
        self.ori_space = 4
        self.dim = 3
        self.feat_per_placed = self.pos_space + self.ori_space + self.dim
        self.obs_dim = 4 + self.max_placed_slots * self.feat_per_placed + 1
        # Baseline before any /place; /start often omits density — reward deltas use this seed.
        self.density_list = [0.0]
        self.x_reached = list()
        # Running max of first box dimension (avoids O(n) max() in reward every step as x_reached grows).
        self._xmax_reached = 0.0
        self.failed_step_reward = -1.0

        low = np.full(7, 0.0, dtype=np.float32)
        high = np.array([2.0, 2.6, 2.75, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.obs_dim,),
            dtype=np.float32,
        )

        self._last_api_error = None
        self._last_placed_box = None
        self._last_place_request = None
        self.last_place_response = None
        self.game_id = None
        self.placed_boxes = []
        self.game_status = None
        self.termination_reason = None
        self.decision_latency_ms = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._last_api_error = None
        self._last_placed_box = None
        self._last_place_request = None
        self.termination_reason = None
        self.decision_latency_ms = None
        mode = self.mode
        if options is not None and "mode" in options:
            mode = options["mode"]

        try:
            r = requests.post(
                f"{self.BASE}/start",
                json={"api_key": self.api_key, "mode": mode},
                timeout=60,
            )
            start = r.json()
        except requests.RequestException as e:
            start = {"detail": str(e), "status": "error"}
        except ValueError:
            start = {"detail": "invalid json from /start", "status": "error"}

        self.start_response = start

        if _api_response_failed(start) or not start.get("game_id"):
            self.game_id = None
            # Match API keys used in step() (depth/width/height), not "length".
            self.truck = start.get("truck") or {"depth": 2.0, "width": 2.6, "height": 2.75}
            self.current_box = start.get("current_box")
            self.placed_boxes = list(start.get("placed_boxes") or [])
            self.last_place_response = None
            self.boxes_remaining = start.get("boxes_remaining")
            self.game_status = start.get("game_status")
            self.density_list = [0.0]
            self.x_reached = []
            self._xmax_reached = 0.0
            if self.current_box is not None:
                d0 = float(self.current_box["dimensions"][0])
                self.x_reached.append(d0)
                self._xmax_reached = d0
            self.density = self.density_list[-1]
            self._last_api_error = start
            info = {
                "api_failed": True,
                "error": start.get("detail", start),
            }
            return self._build_obs(), info

        self.game_id = start.get("game_id")
        self.truck = start.get("truck")
        self.current_box = start.get("current_box")
        self.last_place_response = None
        self.boxes_remaining = start.get("boxes_remaining")
        self.game_status = start.get("game_status")
        self.placed_boxes = list(start.get("placed_boxes") or [])
        self.density_list = [0.0]
        self.x_reached = []
        self._xmax_reached = 0.0
        if self.current_box is not None:
            d0 = float(self.current_box["dimensions"][0])
            self.x_reached.append(d0)
            self._xmax_reached = d0
        self.density = self.density_list[-1]
        if start.get("density") is not None:
            self.density = float(start["density"])
            self.density_list[-1] = self.density

        return self._build_obs(), self._info_dict()

    def step(self, action):
        # asset that action should be a numpy array of shape (7,)
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.shape[0] != 7:
            raise ValueError(f"action must have shape (7,), got {action.shape}")

        if not self.game_id:
            if self._last_api_error is not None:
                return (
                    self._build_obs(),
                    self.failed_step_reward,
                    False,
                    True,
                    self._info_dict(
                        api_error=self._last_api_error,
                        terminated=False,
                        truncated=True,
                    ),
                )
            raise RuntimeError("call reset() first")

        if self.current_box is None:
            terminated = self.terminated()
            truncated = self.truncated()
            return (
                self._build_obs(),
                0.0,
                terminated,
                truncated,
                self._info_dict(terminated=terminated, truncated=truncated),
            )

        prev_box = self.current_box

        # lets now clip the actions so that the box is not placed outside the truck.
        # if the action[:3] + dimensions of the box is greater than or equal to dimensions of the truck,
        # then clip the action[:3] to the dimensions of the truck - dimensions of the box.
        pos = action[:3].tolist()
        truck_dims = [self.truck['depth'], self.truck['width'], self.truck['height']]
        for i, dim in enumerate(truck_dims):
            if pos[i] + self.current_box["dimensions"][i]/2 > dim:
                pos[i] = dim - self.current_box["dimensions"][i]/2
            if pos[i] - self.current_box["dimensions"][i]/2 < 0:
                pos[i] = self.current_box["dimensions"][i]/2
        action[:3] = pos

# # now we can constrain orientation to be a quaternion, in that the sum of the squares of the quaternion should be 1.
#         ori = action[3:].tolist()
#         if ori[0]**2 + ori[1]**2 + ori[2]**2 + ori[3]**2 != 1:
#     # resample the quaternion so that the sum of the squares of the quaternion is 1.
#          ori = Quaternion(ori).normalised.elements.tolist()
#          action[3:] = ori
        ori = [1,0,0,0]
        action[3:] = ori

        if self.mode == "compete":
            self._last_place_request = {
                "game_id": self.game_id,
                "box_id": self.current_box["id"],
                "position": list(pos),
                "orientation_wxyz": list(ori),
            }

        try:
            r = requests.post(
                f"{self.BASE}/place",
                json={
                    "game_id": self.game_id,
                    "box_id": self.current_box["id"],
                    "position": pos,
                    "orientation_wxyz": ori,
                },
                timeout=60,
            )
            place = r.json()
        except requests.RequestException as e:
            place = {"detail": str(e), "status": "error"}
        except ValueError:
            place = {"detail": "invalid json from /place", "status": "error"}

        self.last_place_response = place
        # notes.txt: every /place JSON includes termination_reason (often None while in_progress).
        if isinstance(place, dict):
            self.termination_reason = place.get("termination_reason")
            self.decision_latency_ms = place.get("decision_latency_ms")

        failed = _api_response_failed(place)
        if failed:
            self._last_api_error = place
            self.status = place.get("status")
            obs = self._build_obs()
            info = self._info_dict(
                api_error=place, terminated=False, truncated=True
            )
            return obs, self.failed_step_reward, False, True, info

        self._last_api_error = None
        self.status = place.get("status")
        self.placed_boxes = place.get("placed_boxes") or []
        self.current_box = place.get("current_box")
        self.boxes_remaining = place.get("boxes_remaining")
        self.game_status = place.get("game_status")
        self.density = place.get("density")
        if self.density is not None:
            self.density_list.append(float(self.density))
        else:
            self.density_list.append(self.density_list[-1])

        if self.current_box is not None:
            v = float(self.current_box["dimensions"][0])
            self.x_reached.append(v)
            if v > self._xmax_reached:
                self._xmax_reached = v
        elif prev_box is not None:
            v = float(prev_box["dimensions"][0])
            self.x_reached.append(v)
            if v > self._xmax_reached:
                self._xmax_reached = v

        self._last_placed_box = prev_box
        obs = self._build_obs()
        reward = self._compute_reward(prev_box)
        terminated = self.terminated()
        truncated = self.truncated()
        info = self._info_dict(terminated=terminated, truncated=truncated)
        return obs, reward, terminated, truncated, info

    def _build_obs(self):
        # Fixed (obs_dim,) vector; safe for empty placed_boxes, None current_box, bad dict fields.
        obs = np.zeros(self.obs_dim, dtype=np.float32)
        cb = self.current_box
        if isinstance(cb, dict):
            try:
                dims = cb.get("dimensions")
                if dims is not None:
                    d = np.asarray(dims, dtype=np.float32).reshape(-1)
                    if d.size >= 3:
                        obs[0:3] = d[:3]
                w = cb.get("weight", 0.0)
                obs[3] = float(w) if w is not None else 0.0
            except (TypeError, ValueError):
                pass

        boxes = self.placed_boxes if self.placed_boxes is not None else []
        feat = self.feat_per_placed
        for i, pb in enumerate(boxes[: self.max_placed_slots]):
            if not isinstance(pb, dict):
                continue
            base = 4 + i * feat
            try:
                pos = pb.get("position")
                if pos is not None:
                    p = np.asarray(pos, dtype=np.float32).reshape(-1)
                    if p.size >= 3:
                        obs[base : base + 3] = p[:3]
                ori = pb.get("orientation_wxyz")
                if ori is not None:
                    o = np.asarray(ori, dtype=np.float32).reshape(-1)
                    if o.size >= 4:
                        obs[base + 3 : base + 7] = o[:4]
                dim = pb.get("dimensions")
                if dim is not None:
                    m = np.asarray(dim, dtype=np.float32).reshape(-1)
                    if m.size >= 3:
                        obs[base + 7 : base + 10] = m[:3]
            except (TypeError, ValueError):
                continue

        dens = float(self.density_list[-1]) if self.density_list else 0.0
        obs[-1] = dens
        return obs

    def obs(self):
        return self._build_obs()

    def _compute_reward(self, prev_box):
        if prev_box is None or len(self.density_list) < 2:
            return 0.0
        density_change = self.density_list[-1] - self.density_list[-2]
        d0, d1, d2 = prev_box["dimensions"]
        volume = float(d0 * d1 * d2)
        xmax = self._xmax_reached if self.x_reached else float(d0)
        w = float(self.truck["width"])
        h = float(self.truck["height"])
        denom = volume / (xmax * w * h)
        if denom == 0.0:
            return 0.0
        return float(density_change / denom)

    def reward(self):
        return self._compute_reward(getattr(self, "_last_placed_box", None))

    def _info_dict(self, api_error=None, *, terminated=None, truncated=None):
        # notes.txt: info includes boxes_remaining, termination_reason, game_id, decision_latency_ms
        info = {
            "density_list": self.density_list,
            "x_reached": self.x_reached,
            "truck": self.truck,
            "current_box": self.current_box,
            "boxes_remaining": self.boxes_remaining,
            "game_status": self.game_status,
            "game_id": self.game_id,
            "termination_reason": self.termination_reason,
            "decision_latency_ms": self.decision_latency_ms,
        }
        if api_error is not None:
            info["api_error"] = api_error
        elif self._last_api_error is not None:
            info["api_error"] = self._last_api_error
        if terminated is not None:
            info["terminated"] = terminated
        if truncated is not None:
            info["truncated"] = truncated
        if self.mode == "compete":
            lpr = self.last_place_response
            if isinstance(lpr, dict):
                info["place_response_status"] = lpr.get("status")
            else:
                info["place_response_status"] = None
            boxes = self.placed_boxes
            info["placed_boxes_count"] = (
                len(boxes) if isinstance(boxes, list) else 0
            )
            info["last_place_request"] = self._last_place_request
            info["last_place_response_summary"] = self._place_response_summary(
                lpr
            )
        return info

    @staticmethod
    def _place_response_summary(lpr):
        """Bounded subset of /place JSON for compete diagnostics."""
        if not isinstance(lpr, dict):
            return None
        out = {
            "status": lpr.get("status"),
            "game_status": lpr.get("game_status"),
            "boxes_remaining": lpr.get("boxes_remaining"),
            "termination_reason": lpr.get("termination_reason"),
        }
        cb = lpr.get("current_box")
        if isinstance(cb, dict):
            out["current_box_id"] = cb.get("id")
        pbs = lpr.get("placed_boxes")
        if isinstance(pbs, list):
            out["placed_boxes_len"] = len(pbs)
        if "detail" in lpr:
            det = lpr.get("detail")
            s = det if isinstance(det, str) else str(det)
            if len(s) > 200:
                s = s[:200] + "..."
            out["detail"] = s
        return out

    def info(self):
        return self._info_dict()

    def terminated(self):
        # Reference client: natural end when game_status is "completed" (then break out of while box).
        return self.game_status == "completed"

    def truncated(self):
        if self._last_api_error is not None:
            return True
        if self.game_status == "completed":
            return False
        # while box: stops when current_box is missing; also any explicit non-in_progress end.
        if self.current_box is None:
            return True
        if (
            self.game_status is not None
            and self.game_status != "in_progress"
        ):
            return True
        return False

    def close(self):
        pass