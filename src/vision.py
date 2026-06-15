"""Screen-reading helpers for Geometry Dash.

The Vision class is deliberately decoupled from the game control flow.
It takes a raw RGB frame and returns a small state object; callers decide
what to do with ``is_dead`` and ``just_died``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np


@dataclass
class VisionState:
    is_dead: bool
    just_died: bool
    # Reserved for future use; percentage reading is currently disabled
    # because survival time already proxies for progress.
    percent: Optional[int] = None


class Vision:
    """Read game state from 800x600 RGB frames."""

    def __init__(
        self,
        death_filter_path: str,
        death_threshold: float = 0.8,
        exact_position: bool = False,
    ) -> None:
        self.death_threshold = death_threshold
        self.exact_position = exact_position

        # Filters are stored as RGBA PNGs; OpenCV loads them as BGRA.
        # Convert to BGR and keep the alpha mask so templates match the
        # color space returned by ``cv2.cvtColor(rgb_frame, COLOR_RGB2BGR)``.
        death_full, death_mask_full = self._load_filter(death_filter_path)
        self._death_search_bbox = self._bbox(death_mask_full)
        if self._death_search_bbox is not None:
            x1, y1, x2, y2 = self._death_search_bbox
            self._death_template = death_full[y1:y2, x1:x2]
            self._death_mask = death_mask_full[y1:y2, x1:x2]
        else:
            self._death_template = death_full
            self._death_mask = death_mask_full

        self._prev_dead = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, frame: np.ndarray) -> VisionState:
        """Inspect a new frame and return the current vision state."""
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        is_dead = self._detect_death(frame_bgr)
        just_died = is_dead and not self._prev_dead
        self._prev_dead = is_dead

        return VisionState(is_dead=is_dead, just_died=just_died)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _load_filter(path: str) -> Tuple[np.ndarray, np.ndarray]:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f"Could not load filter: {path}")

        if img.shape[2] == 4:
            bgr = img[:, :, :3]
            mask = (img[:, :, 3] > 0).astype(np.uint8)
        else:
            bgr = img
            mask = np.ones(img.shape[:2], dtype=np.uint8)

        return bgr, mask

    @staticmethod
    def _bbox(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        ys, xs = np.where(mask)
        if len(xs) == 0:
            return None
        return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1

    def _detect_death(self, frame_bgr: np.ndarray) -> bool:
        if self._death_template is None or self._death_search_bbox is None:
            return False

        # The death menu is always in the same screen location, so only match
        # in that region instead of the whole frame.
        x1, y1, x2, y2 = self._death_search_bbox
        search = frame_bgr[y1:y2, x1:x2]
        if search.size == 0:
            return False

        if self.exact_position:
            score = self._masked_ncc(search, self._death_template, self._death_mask)
        else:
            result = cv2.matchTemplate(
                search,
                self._death_template,
                cv2.TM_CCOEFF_NORMED,
                mask=self._death_mask,
            )
            score = cv2.minMaxLoc(result)[1]

        return score >= self.death_threshold

    @staticmethod
    def _masked_ncc(
        frame: np.ndarray, template: np.ndarray, mask: np.ndarray
    ) -> float:
        """Normalized cross-correlation over the masked pixels.

        Equivalent to the score ``cv2.matchTemplate`` would return at the
        single aligned position, but without the sliding-window overhead.
        """
        mask_bool = mask.astype(bool)
        if not np.any(mask_bool):
            return 0.0

        f = frame[mask_bool].astype(np.float32)
        t = template[mask_bool].astype(np.float32)

        f_mean = f.mean()
        t_mean = t.mean()

        f_centered = f - f_mean
        t_centered = t - t_mean

        num = np.sum(f_centered * t_centered)
        den = np.sqrt(np.sum(f_centered ** 2) * np.sum(t_centered ** 2))

        if den < 1e-9:
            return 0.0
        return float(num / den)
