"""10 Hz constant-velocity Kalman on body-frame entities (howto 1.3.2).

Predicts E, not a. No target GT odom. No OWL.
"""

from __future__ import annotations

import math
from collections import deque

import numpy as np

from .types import Detection, Snapshot, Track

DT = 0.1
SIGMA_A = 0.5
GATE_CHI2 = 9.21
APPEAR_W = 2.0
APPEAR_EMA = 0.8
APPEAR_DIM = 512
SAME_VISUAL_BEAR = math.radians(10.0)
# OWL runs at 2 Hz. Keep a predicted track across short glare/occlusion gaps
# for about four seconds; 10 Hz prediction ticks themselves never consume this
# budget. The learned null selector still decides whether a retained entity is
# useful for the current task.
MAX_MISSES = 8
HIST_S = 1.0
# Waterline intersection is noisier than a direct range sensor.  The measured
# closed-loop position MAE is about 0.5 m, so a 0.1 m covariance made normal
# bbox jitter fail the chi-square gate and spawned duplicate tracks.
SIGMA_RANGE = 0.35
SIGMA_BEAR = math.radians(2.0)
VEL_PRIOR = 2.0
ASSIGN_BIG = 1.0e9
H = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float64)


def _fq(dt: float) -> tuple[np.ndarray, np.ndarray]:
    dt = float(dt)
    f = np.array(
        [
            [1.0, 0.0, dt, 0.0],
            [0.0, 1.0, 0.0, dt],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    sa2 = SIGMA_A * SIGMA_A
    dt2 = dt * dt
    dt3 = dt2 * dt
    dt4 = dt2 * dt2
    q1 = sa2 * dt4 / 4.0
    q2 = sa2 * dt3 / 2.0
    q3 = sa2 * dt2
    q = np.array(
        [
            [q1, 0.0, q2, 0.0],
            [0.0, q1, 0.0, q2],
            [q2, 0.0, q3, 0.0],
            [0.0, q2, 0.0, q3],
        ],
        dtype=np.float64,
    )
    return f, q


def _rot2(dpsi: float) -> np.ndarray:
    c = math.cos(dpsi)
    s = math.sin(dpsi)
    return np.array([[c, s], [-s, c]], dtype=np.float64)


def meas_R(x: float, y: float) -> np.ndarray:
    r = math.hypot(x, y)
    if r < 1e-3:
        return np.eye(2, dtype=np.float64) * (SIGMA_RANGE ** 2)
    th = math.atan2(y, x)
    ct, st = math.cos(th), math.sin(th)
    jac = np.array([[ct, -r * st], [st, r * ct]], dtype=np.float64)
    d = np.diag([SIGMA_RANGE ** 2, SIGMA_BEAR ** 2])
    return jac @ d @ jac.T


def _cosine(a: np.ndarray | None, b: np.ndarray | None) -> float:
    if a is None or b is None:
        return 1.0
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-9 or nb < 1e-9:
        return 1.0
    return float(np.dot(a, b) / (na * nb))


def _assign(cost: np.ndarray) -> list[tuple[int, int]]:
    if cost.size == 0:
        return []
    try:
        from scipy.optimize import linear_sum_assignment

        ri, ci = linear_sum_assignment(cost)
        return list(zip((int(i) for i in ri), (int(j) for j in ci)))
    except Exception:
        n, m = cost.shape
        used_r: set[int] = set()
        used_c: set[int] = set()
        pairs: list[tuple[int, int]] = []
        items = sorted((float(cost[i, j]), i, j) for i in range(n) for j in range(m))
        for c, i, j in items:
            if not math.isfinite(c) or c >= ASSIGN_BIG * 0.5:
                continue
            if i in used_r or j in used_c:
                continue
            pairs.append((i, j))
            used_r.add(i)
            used_c.add(j)
        return pairs


def _maha2(track: Track, det: Detection) -> float:
    z = np.array([det.x, det.y], dtype=np.float64)
    x = track.state()
    s = H @ track.P @ H.T + meas_R(det.x, det.y)
    innov = z - (H @ x)
    try:
        sol = np.linalg.solve(s, innov)
    except np.linalg.LinAlgError:
        return ASSIGN_BIG
    return float(innov @ sol)


def _bearing_error(track: Track, det: Detection) -> float:
    a = math.atan2(float(track.y), float(track.x))
    b = math.atan2(float(det.y), float(det.x))
    return abs(math.atan2(math.sin(a - b), math.cos(a - b)))


class KalmanTracker:
    def __init__(self, max_misses: int = MAX_MISSES):
        self.max_misses = int(max_misses)
        self.tracks: list[Track] = []
        self._t: float | None = None
        self._next_id = 1
        self._hist: deque[Snapshot] = deque()

    def entity_matrix(self) -> np.ndarray:
        if not self.tracks:
            return np.zeros((0, 5), dtype=np.float64)
        rows = [[tr.x, tr.y, tr.vx, tr.vy, tr.rho] for tr in self.tracks]
        return np.asarray(rows, dtype=np.float64)

    def raw_entity_matrix(self, ego_velocity=(0.0, 0.0)) -> np.ndarray:
        """Task-independent tracker observations: kinematics + OWL appearance.

        This is the source representation for the semantic16 pipeline.  The
        learned projector consumes columns 4:516 and produces the 16-D
        semantic feature; rho is intentionally absent.
        """
        if not self.tracks:
            return np.zeros((0, 4 + APPEAR_DIM), dtype=np.float64)
        rows = []
        for tr in self.tracks:
            appearance = np.zeros(APPEAR_DIM, dtype=np.float64)
            if tr.appearance is not None:
                src = np.asarray(tr.appearance, dtype=np.float64).reshape(-1)
                n = min(APPEAR_DIM, int(src.size))
                appearance[:n] = src[:n]
            kinematics = np.array(
                [
                    tr.x,
                    tr.y,
                    tr.vx - float(ego_velocity[0]),
                    tr.vy - float(ego_velocity[1]),
                ],
                dtype=np.float64,
            )
            rows.append(np.concatenate((kinematics, appearance)))
        return np.asarray(rows, dtype=np.float64)

    def step(
        self,
        t: float,
        yaw_rate: float,
        detections: list[Detection] | None = None,
        ego_velocity=(0.0, 0.0),
    ) -> np.ndarray:
        t = float(t)
        yaw_rate = float(yaw_rate)
        detector_ran = detections is not None
        dets = list(detections or [])
        if self._t is None:
            self._t = t
            if dets:
                self._associate_and_update(dets)
            self._push(t, yaw_rate, ego_velocity)
            return self.entity_matrix()

        t_img = None
        if dets:
            t_img = min(float(d.stamp) for d in dets)
            if t_img < self._t - 1e-6:
                self._rewind_to(t_img)
                self._roll_forward(t_img, yaw_rate, ego_velocity)

        target = t_img if t_img is not None else t
        if target > self._t + 1e-9:
            self._predict(target - self._t, yaw_rate, ego_velocity)
            self._t = target

        if dets:
            self._associate_and_update(dets)
            if t > self._t + 1e-9:
                self._roll_forward(t, yaw_rate, ego_velocity)
        elif detector_ran:
            # None means that this is only a 10 Hz prediction/control tick.
            # An empty list means the 2 Hz detector actually ran and found
            # nothing, so only that case counts as one missed observation.
            for tr in self.tracks:
                tr.misses += 1
            self._prune()

        self._t = t
        self._push(t, yaw_rate, ego_velocity)
        return self.entity_matrix()

    def _push(self, t: float, yaw_rate: float, ego_velocity=(0.0, 0.0)) -> None:
        snap = Snapshot(
            t=t,
            yaw_rate=yaw_rate,
            ego_velocity=(float(ego_velocity[0]), float(ego_velocity[1])),
            tracks=[tr.copy() for tr in self.tracks],
        )
        if self._hist and abs(self._hist[-1].t - t) <= 1e-9:
            self._hist[-1] = snap
        else:
            self._hist.append(snap)
        while self._hist and (t - self._hist[0].t) > HIST_S + 1e-9:
            self._hist.popleft()

    def _rewind_to(self, t_img: float) -> None:
        chosen: Snapshot | None = None
        for snap in self._hist:
            if snap.t <= t_img + 1e-9:
                chosen = snap
        if chosen is None:
            return
        self.tracks = [tr.copy() for tr in chosen.tracks]
        self._t = chosen.t

    def _roll_forward(self, t_to: float, yaw_fallback: float, ego_fallback=(0.0, 0.0)) -> None:
        if self._t is None:
            self._t = t_to
            return
        t = self._t
        for snap in self._hist:
            if snap.t <= t + 1e-9:
                continue
            if snap.t > t_to + 1e-9:
                break
            self._predict(snap.t - t, snap.yaw_rate, snap.ego_velocity)
            t = snap.t
            # A delayed measurement changed the state in the past.  Persist
            # the corrected replayed state at every later history point so
            # the next delayed image does not restore an obsolete trajectory.
            snap.tracks = [tr.copy() for tr in self.tracks]
        if t_to > t + 1e-9:
            self._predict(t_to - t, yaw_fallback, ego_fallback)
            t = t_to
        self._t = t

    def _predict(self, dt: float, yaw_rate: float, ego_velocity=(0.0, 0.0)) -> None:
        if dt <= 1e-12:
            return
        f, q = _fq(dt)
        dpsi = float(yaw_rate) * float(dt)
        r2 = _rot2(dpsi)
        t4 = np.zeros((4, 4), dtype=np.float64)
        t4[:2, :2] = r2
        t4[2:, 2:] = r2
        ego_step = np.array(ego_velocity, dtype=np.float64) * float(dt)
        for tr in self.tracks:
            x = f @ tr.state()
            x[:2] -= ego_step
            x[:2] = r2 @ x[:2]
            x[2:] = r2 @ x[2:]
            tr.x, tr.y, tr.vx, tr.vy = (float(v) for v in x)
            tr.P = t4 @ (f @ tr.P @ f.T + q) @ t4.T

    def _rotate_body(self, dpsi: float) -> None:
        if abs(dpsi) < 1e-18:
            return
        r2 = _rot2(dpsi)
        t4 = np.zeros((4, 4), dtype=np.float64)
        t4[:2, :2] = r2
        t4[2:, 2:] = r2
        for tr in self.tracks:
            xy = r2 @ np.array([tr.x, tr.y], dtype=np.float64)
            vv = r2 @ np.array([tr.vx, tr.vy], dtype=np.float64)
            tr.x, tr.y = float(xy[0]), float(xy[1])
            tr.vx, tr.vy = float(vv[0]), float(vv[1])
            tr.P = t4 @ tr.P @ t4.T

    def _associate_and_update(self, dets: list[Detection]) -> None:
        n, m = len(self.tracks), len(dets)
        matched_t: set[int] = set()
        matched_d: set[int] = set()
        if n and m:
            cost = np.full((n, m), ASSIGN_BIG, dtype=np.float64)
            maha = np.full((n, m), ASSIGN_BIG, dtype=np.float64)
            for i, tr in enumerate(self.tracks):
                for j, det in enumerate(dets):
                    d2 = _maha2(tr, det)
                    maha[i, j] = d2
                    cosine = _cosine(tr.appearance, det.appearance)
                    bearing = _bearing_error(tr, det)
                    same_visual_ray = cosine >= 0.94 and bearing <= SAME_VISUAL_BEAR
                    if d2 <= GATE_CHI2 or same_visual_ray:
                        appear = APPEAR_W * (1.0 - cosine)
                        # Cap the Cartesian term for same-ray matches because
                        # monocular waterline range has scale jitter; bearing
                        # and OWL appearance remain stable for that entity.
                        spatial = min(d2, GATE_CHI2) if same_visual_ray else d2
                        cost[i, j] = spatial + appear + (bearing / math.radians(2.0)) ** 2
            for i, j in _assign(cost):
                same_visual_ray = (
                    _cosine(self.tracks[i].appearance, dets[j].appearance) >= 0.94
                    and _bearing_error(self.tracks[i], dets[j]) <= SAME_VISUAL_BEAR
                )
                if maha[i, j] > GATE_CHI2 and not same_visual_ray:
                    continue
                self._kalman_update(self.tracks[i], dets[j])
                matched_t.add(i)
                matched_d.add(j)
        for i, tr in enumerate(self.tracks):
            if i in matched_t:
                tr.misses = 0
            else:
                tr.misses += 1
        self._prune()
        for j, det in enumerate(dets):
            if j not in matched_d:
                self._spawn(det)

    def _kalman_update(self, tr: Track, det: Detection) -> None:
        z = np.array([det.x, det.y], dtype=np.float64)
        x = tr.state()
        r = meas_R(det.x, det.y)
        s = H @ tr.P @ H.T + r
        k = tr.P @ H.T @ np.linalg.inv(s)
        x = x + k @ (z - H @ x)
        tr.x, tr.y, tr.vx, tr.vy = (float(v) for v in x)
        i4 = np.eye(4)
        tr.P = (i4 - k @ H) @ tr.P
        tr.rho = float(det.rho)
        tr.hits += 1
        if det.appearance is not None:
            observed = np.asarray(det.appearance, dtype=np.float64).reshape(-1)
            norm = float(np.linalg.norm(observed))
            if norm > 1e-12:
                observed = observed / norm
            if tr.appearance is None or np.asarray(tr.appearance).shape != observed.shape:
                tr.appearance = np.array(observed, copy=True)
            else:
                # Entity semantics should be stable across frames.  Keep the
                # raw OWL feature task-independent and smooth it on the track.
                blended = APPEAR_EMA * np.asarray(tr.appearance) + (1.0 - APPEAR_EMA) * observed
                blended_norm = float(np.linalg.norm(blended))
                tr.appearance = blended / blended_norm if blended_norm > 1e-12 else blended

    def _spawn(self, det: Detection) -> None:
        r = meas_R(det.x, det.y)
        p = np.zeros((4, 4), dtype=np.float64)
        p[:2, :2] = r
        p[2, 2] = VEL_PRIOR ** 2
        p[3, 3] = VEL_PRIOR ** 2
        app = None if det.appearance is None else np.asarray(det.appearance, dtype=np.float64).reshape(-1)
        if app is not None:
            norm = float(np.linalg.norm(app))
            app = app / norm if norm > 1e-12 else app
        self.tracks.append(
            Track(
                id=self._next_id,
                x=float(det.x),
                y=float(det.y),
                vx=0.0,
                vy=0.0,
                P=p,
                rho=float(det.rho),
                appearance=app,
                misses=0,
                hits=1,
            )
        )
        self._next_id += 1

    def _prune(self) -> None:
        self.tracks = [tr for tr in self.tracks if tr.misses < self.max_misses]
