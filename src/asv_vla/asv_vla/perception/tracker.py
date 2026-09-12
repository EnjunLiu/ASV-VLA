from __future__ import annotations

import math
from collections import deque

import numpy as np
from scipy.optimize import linear_sum_assignment

from .types import Detection, Snapshot, Track

# 匀加速过程噪声标准差 (m/s^2)
SIGMA_A = 0.5
# 马氏距离卡方门限（2 自由度，约 99%）
GATE_CHI2 = 9.21
# 外观代价权重
APPEAR_W = 2.0
# 外观特征 EMA 平滑系数
APPEAR_EMA = 0.8
# OWL-ViT 外观特征维度
APPEAR_DIM = 512
# 同一视觉射线的方位角容差
SAME_VISUAL_BEAR = math.radians(10.0)
# OWL 以 2 Hz 运行。短时反光/遮挡时用预测航迹维持约 4 秒；
# 10 Hz 预测节拍本身不消耗该预算。学到的空选择器仍决定保留实体是否对当前任务有用。
MAX_MISSES = 8
# 历史快照窗口（秒），用于迟到检测回放
HIST_S = 1.0
# 水线交点测距比直接距离传感器更噪。闭环位置 MAE 约 0.5 m，
# 若协方差取 0.1 m，正常框抖动会打不进卡方门限并分裂出航迹。
SIGMA_RANGE = 0.35
# 方位角测量噪声
SIGMA_BEAR = math.radians(2.0)
# 新生航迹速度先验标准差
VEL_PRIOR = 2.0
# 匈牙利分配中的不可关联代价
ASSIGN_BIG = 1.0e9
# 观测矩阵：只观测船体坐标系位置 x, y
H = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float64)


# 匀加速运动模型 F 与过程噪声 Q
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


# 船体坐标系绕 yaw 的二维旋转
def _rot2(dpsi: float) -> np.ndarray:
    c = math.cos(dpsi)
    s = math.sin(dpsi)
    return np.array([[c, s], [-s, c]], dtype=np.float64)


# 将距离/方位噪声映射到笛卡尔位置协方差
def meas_R(x: float, y: float) -> np.ndarray:
    r = math.hypot(x, y)
    if r < 1e-3:
        return np.eye(2, dtype=np.float64) * (SIGMA_RANGE ** 2)
    th = math.atan2(y, x)
    ct, st = math.cos(th), math.sin(th)
    jac = np.array([[ct, -r * st], [st, r * ct]], dtype=np.float64)
    d = np.diag([SIGMA_RANGE ** 2, SIGMA_BEAR ** 2])
    return jac @ d @ jac.T


# 外观余弦相似度；缺特征时视为相同
def _cosine(a: np.ndarray | None, b: np.ndarray | None) -> float:
    if a is None or b is None:
        return 1.0
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-9 or nb < 1e-9:
        return 1.0
    return float(np.dot(a, b) / (na * nb))


# 匈牙利分配
def _assign(cost: np.ndarray) -> list[tuple[int, int]]:
    if cost.size == 0:
        return []
    rows, columns = linear_sum_assignment(cost)
    return [(int(row), int(column)) for row, column in zip(rows, columns)]


# 检测相对航迹的马氏距离平方
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


# 航迹与检测的方位角误差
def _bearing_error(track: Track, det: Detection) -> float:
    a = math.atan2(float(track.y), float(track.x))
    b = math.atan2(float(det.y), float(det.x))
    return abs(math.atan2(math.sin(a - b), math.cos(a - b)))


# 卡尔曼跟踪器：船体坐标系位置/速度 + OWL 外观
class KalmanTracker:
    def __init__(self, max_misses: int = MAX_MISSES):
        self.max_misses = int(max_misses)
        self.tracks: list[Track] = []
        self._t: float | None = None
        self._next_id = 1
        self._hist: deque[Snapshot] = deque()

    # 任务无关的跟踪观测：运动学 + OWL 外观。semantic16 投影器使用第 4:516 列。
    def raw_entity_matrix(self, ego_velocity=(0.0, 0.0)) -> np.ndarray:
        entities = np.zeros((len(self.tracks), 4 + APPEAR_DIM), dtype=np.float64)
        for row, tr in zip(entities, self.tracks):
            row[:4] = (tr.x, tr.y, tr.vx - float(ego_velocity[0]), tr.vy - float(ego_velocity[1]))
            if tr.appearance is not None:
                src = np.asarray(tr.appearance, dtype=np.float64).reshape(-1)
                n = min(APPEAR_DIM, int(src.size))
                row[4 : 4 + n] = src[:n]
        return entities

    # 主循环：预测、关联更新、迟到图像回放
    def step(
        self,
        t: float,
        yaw_rate: float,
        detections: list[Detection] | None = None,
        ego_velocity=(0.0, 0.0),
    ) -> None:
        t = float(t)
        yaw_rate = float(yaw_rate)
        detector_ran = detections is not None
        dets = list(detections or [])
        if self._t is None:
            self._t = t
            if dets:
                self._associate_and_update(dets)
            self._push(t, yaw_rate, ego_velocity)
            return

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
            # 空列表：2 Hz 检测器跑过且无目标，记一次漏检。
            # detections is None 走不到这里，那是 10 Hz 预测/控制节拍。
            for tr in self.tracks:
                tr.misses += 1
            self._prune()

        self._t = t
        self._push(t, yaw_rate, ego_velocity)

    # 压入历史快照，超出 HIST_S 的旧帧丢弃
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

    # 回退到不晚于图像时刻的快照，用于迟到检测
    def _rewind_to(self, t_img: float) -> None:
        chosen: Snapshot | None = None
        for snap in self._hist:
            if snap.t <= t_img + 1e-9:
                chosen = snap
        if chosen is None:
            return
        self.tracks = [tr.copy() for tr in chosen.tracks]
        self._t = chosen.t

    # 从当前时刻沿历史快照前滚到目标时刻
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
            # 迟到量测已修正过去状态。把回放后的状态写回后续历史点，
            # 避免下一帧迟到图像再恢复过时轨迹。
            snap.tracks = [tr.copy() for tr in self.tracks]
        if t_to > t + 1e-9:
            self._predict(t_to - t, yaw_fallback, ego_fallback)
            t = t_to
        self._t = t

    # 船体坐标系下的卡尔曼预测：扣本船位移并随 yaw 旋转
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

    # 检测与航迹关联：马氏门限 + OWL 外观/方位射线
    def _associate_and_update(self, dets: list[Detection]) -> None:
        n, m = len(self.tracks), len(dets)
        matched_t: set[int] = set()
        matched_d: set[int] = set()
        if n and m:
            cost = np.full((n, m), ASSIGN_BIG, dtype=np.float64)
            rejected = np.zeros((n, m), dtype=np.bool_)
            for i, tr in enumerate(self.tracks):
                for j, det in enumerate(dets):
                    d2 = _maha2(tr, det)
                    cosine = _cosine(tr.appearance, det.appearance)
                    bearing = _bearing_error(tr, det)
                    same_visual_ray = cosine >= 0.94 and bearing <= SAME_VISUAL_BEAR
                    rejected[i, j] = d2 > GATE_CHI2 and not same_visual_ray
                    if d2 <= GATE_CHI2 or same_visual_ray:
                        appear = APPEAR_W * (1.0 - cosine)
                        # 同一射线匹配时截断笛卡尔项：单目水线距离有尺度抖动，
                        # 方位和 OWL 外观对该实体更稳定。
                        spatial = min(d2, GATE_CHI2) if same_visual_ray else d2
                        cost[i, j] = spatial + appear + (bearing / math.radians(2.0)) ** 2
            for i, j in _assign(cost):
                if rejected[i, j]:
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

    # 卡尔曼量测更新，并用 EMA 平滑 OWL 外观
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
        tr.hits += 1
        if det.appearance is not None:
            observed = np.asarray(det.appearance, dtype=np.float64).reshape(-1)
            norm = float(np.linalg.norm(observed))
            if norm > 1e-12:
                observed = observed / norm
            if tr.appearance is None or np.asarray(tr.appearance).shape != observed.shape:
                tr.appearance = np.array(observed, copy=True)
            else:
                # 实体语义应跨帧稳定。原始 OWL 特征保持任务无关，只在航迹上平滑。
                blended = APPEAR_EMA * np.asarray(tr.appearance) + (1.0 - APPEAR_EMA) * observed
                blended_norm = float(np.linalg.norm(blended))
                tr.appearance = blended / blended_norm if blended_norm > 1e-12 else blended

    # 从未匹配检测新生航迹
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
                appearance=app,
                misses=0,
                hits=1,
            )
        )
        self._next_id += 1

    # 删除连续漏检超过上限的航迹
    def _prune(self) -> None:
        self.tracks = [tr for tr in self.tracks if tr.misses < self.max_misses]
