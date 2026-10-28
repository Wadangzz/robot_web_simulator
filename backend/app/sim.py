"""시뮬레이션 엔진.

두 종류의 plant 를 제공한다 (controller 이름으로 분기):
- 토크 입력 (`_run_torque`): PD / PID / Computed Torque / 임피던스.
  2차계 — forward_dynamics 로 θ̈ 적분 (폐루프가 stiff → Radau).
- 속도 입력 (`_run_velocity`): joint_velocity / resolved_rate (MR §11.3).
  1차계 — θ̇ 가 곧 제어출력, θ 만 적분(기구학 시뮬). stiff 아님 → RK45.

robot_math.Dynamics 는 Mlist/Glist/Slist 를 인자로 받으므로 엔진은 로봇 무관(n-무관)이다.
대상 로봇은 RobotModel 인스턴스(robot)로 주입한다 — robot.fk/jacobian_body/dls_inv/
manipulability·robot.n·robot.dyn_args·robot.gravity 를 통해 기구학·동역학 파라미터에 접근.
run_simulation(robot=None) 이면 기본 로봇 UR5 를 레지스트리에서 가져온다(데모/하위호환).
"""
import numpy as np
from motion_core.loads import ArticulatedRobot
from motion_core.profiles import InfeasiblePath, time_path
from motion_core.robots import dynamics_fast as df  # numba JIT RNE (~14x)
from motion_core.robots.robot_math import SE3, SO3, quintic_time_scaling
from scipy.integrate import solve_ivp

# --- 공통 적분/정착/발산 상수 ---
SETTLE_MAX = 2.0        # 모션 종료 후 평형 대기(s) — 속도제어 등 고정창 경로용
SETTLE_HARD = 10.0      # adaptive 종료 절대 상한(s) — 정착 못하면 여기서 끊음(안전망)
VEL_TOL = 0.02          # 정상상태(평형) 판정 속도 임계 rad/s
ACC_TOL = 0.5           # 평형 판정 가속도 임계 rad/s²(·m/s²) — 진동 꼭대기(속도0·가속도↑) 배제
VEL_WIN = 0.2           # 그 속도 이하를 유지해야 하는 윈도우 s
THETA_MAX = 3 * np.pi   # 발산 안전망: 위치(rad), 오버슛보다 큼
OMEGA_MAX = 20.0        # 발산 안전망: 속도(rad/s), 정상 ~2 보다 큼
KD_NULL = 5.0           # 널공간 감쇠(여유 DOF) — 7-DOF 내부 드리프트 방지

VELOCITY_CONTROLLERS = ("joint_velocity", "resolved_rate")


def _default_robot():
    """기본 로봇(UR5) — robot 미지정 시 사용. 레지스트리 캐시."""
    from motion_core.robots import robot_registry as registry
    return registry.get_robot("ur5")


def warmup():
    """numba RNE 를 미리 JIT 컴파일(서버 부팅 시 1회, 기본 로봇 UR5 기준). fresh 환경
    첫 /api/run 의 ~14s 컴파일 지연을 부팅으로 옮긴다(이후 디스크 캐시로 ~0.5s)."""
    robot = _default_robot()
    n = robot.n
    Mp, Gp, Sp = df.stack_model(*robot.dyn_args)
    th = robot.ready.astype(float)
    z, g, z6 = np.zeros(n), robot.gravity, np.zeros(6)
    df.forward_dynamics(th, z, z, g, z6, Mp, Gp, Sp)
    df.mass_matrix(th, Mp, Gp, Sp)
    df.gravity_forces(th, g, Mp, Gp, Sp)
    df.coriolis_forces(th, z, Mp, Gp, Sp)


def run_simulation(waypoints, controller="computed_torque", gains=None,
                   gravity_comp=True, t_seg=1.2, hold=0.6, hz=30,
                   plant=None, ctrl=None, disturbance=None, traj_mode="joint",
                   friction=0.0, tau_max=0.0, control_rate=0.0, noise=0.0,
                   noise_seed=0, force_task=None, robot=None,
                   null_obj=None, null_gain=1.0,
                   traj_tau_max=None, traj_dq_max=0.0):
    """관절 경유점(rad)을 제어기로 순회. → {t, theta, tcp, error, torque, qdot, ...}.

    궤적생성(traj_mode)과 제어기는 직교(독립):
    - traj_mode='joint': 관절공간 quintic 보간(현행). 말단 경로는 곡선.
    - traj_mode='task' : 직교 SE(3) 직선보간 → IK 샘플 → θ_d. 말단 직선이지만
      특이점/작업영역밖에서 IK 실패·관절속도 폭발 시 그 지점까지 자르고 traj_error 보고.
    - traj_mode='joint_timeopt': 같은 관절 직선 경로를 motion-core
      time_path 로 타이밍: 구간 시간 T 를 t_seg 가 아니라 토크 한계
      (traj_tau_max, N·m 관절별)·속도 한계(traj_dq_max, rad/s)에서 역산.
      한계 미지정이면 traj_error 로 표면화.
    제어기는 모두 θ_d(t) 를 받으므로 어느 궤적이든 동일하게 추종한다.

    controller 가 속도제어/어드미턴스면 기구학 plant, 그 외는 동역학 plant 로 분기.
    plant/ctrl/disturbance 는 동역학 plant·어드미턴스에서만 의미가 있다.
    """
    gains = gains or {}
    robot = robot if robot is not None else _default_robot()
    WP = _prep_waypoints(waypoints)
    ref, T, traj_error = _build_ref(WP, t_seg, hold, traj_mode, robot,
                                    traj_tau_max, traj_dq_max)
    if T <= 0:                                  # 궤적 생성 자체 실패(첫 점부터 불가)
        return _empty_result(WP, traj_error, robot)

    if controller == "admittance":
        result = _run_admittance(WP, ref, T, gains, hz, disturbance, robot)
    elif controller == "force":
        result = _run_force(WP, gains, hz, plant, force_task, robot)
    elif controller == "hybrid":
        result = _run_hybrid(WP, gains, hz, plant, force_task, robot)
    elif controller in VELOCITY_CONTROLLERS:
        result = _run_velocity(WP, ref, T, controller, gains, hz, robot,
                               null_obj, null_gain)
    else:
        result = _run_torque(WP, ref, T, controller, gains, gravity_comp, hz,
                             plant, ctrl, disturbance, friction, tau_max,
                             control_rate, noise, noise_seed, robot)
    result["traj_error"] = traj_error
    return result


# ----------------------------------------------------------------------
#  공유 헬퍼 (토크·속도 plant 공용)
# ----------------------------------------------------------------------
def _prep_waypoints(waypoints):
    WP = [np.asarray(w, dtype=float) for w in waypoints]
    if len(WP) < 2:
        WP = WP + WP
    return WP


def _make_ref(WP, t_seg, hold, robot):
    """경유점 → 시간함수 ref(t)=(θ_d, θ̇_d, θ̈_d) (구간별 quintic + hold). (ref, T) 반환."""
    N = robot.n
    n_seg = len(WP) - 1
    seg = t_seg + hold
    T = n_seg * seg

    def ref(t):
        k = min(int(t // seg), n_seg - 1)
        loc = t - k * seg
        q0, q1 = WP[k], WP[k + 1]
        dq = q1 - q0
        if loc < t_seg:
            s, sd, sdd = quintic_time_scaling(loc, t_seg)
            return q0 + s * dq, sd * dq, sdd * dq
        return q1.copy(), np.zeros(N), np.zeros(N)

    return ref, T


def _se3_straight(X0, X1, s):
    """직교 직선 보간: 위치 선형 + 회전 측지선(SLERP 등가). s∈[0,1]."""
    Td = np.eye(4)
    Td[:3, 3] = (1 - s) * X0[:3, 3] + s * X1[:3, 3]
    Rrel = SO3.log(X0[:3, :3].T @ X1[:3, :3])   # 상대회전(회전벡터)
    Td[:3, :3] = X0[:3, :3] @ SO3.exp(s * Rrel)
    return Td


def _make_ref_timeopt(WP, hold, robot, tau_max, dq_max):
    """경유점 → 토크 한계 역산 타이밍 ref(t). (ref, T, traj_error) 반환.

    경로는 quintic 모드와 같은 관절공간 직선, 타이밍만 바뀐다: 구간
    시간을 사람이 주는 t_seg 대신 motion-core `time_path` 가 토크·속도
    예산에서 역산한다(보수적 경계라 시간최적 아님, path_timing.py 참조).
    토크 한계는 URDF effort 가 아니라 호출자가 명시한다(관절별/스칼라).
    """
    N = robot.n
    if tau_max is None or not np.all(np.asarray(tau_max, dtype=float) > 0):
        return None, 0.0, "토크 한계(traj_tau_max) 미지정: timeopt 불가"
    if dq_max is None or dq_max <= 0:
        return None, 0.0, "속도 한계(traj_dq_max) 미지정: timeopt 불가"
    tau = np.broadcast_to(np.asarray(tau_max, dtype=float), (N,))
    # rated(RMS) 한계는 time_path 가 쓰지 않는다 — peak 와 같은 값으로 채운다.
    load = ArticulatedRobot(robot, max_torque=tau, rated_torque=tau)
    try:
        timings = [time_path(load, WP[k], WP[k + 1], dq_max=dq_max)
                   for k in range(len(WP) - 1)]
    except InfeasiblePath as e:
        return None, 0.0, f"토크 한계로 실행 불가: {e}"
    starts = np.r_[0.0, np.cumsum([tm.duration + hold for tm in timings])]
    T = float(starts[-1])

    def ref(t):
        k = min(max(int(np.searchsorted(starts, t, side="right")) - 1, 0),
                len(timings) - 1)
        q, dq, ddq = timings[k].sample(np.r_[t - starts[k]])
        return q[0], dq[0], ddq[0]

    return ref, T, None


def _build_ref(WP, t_seg, hold, traj_mode, robot, traj_tau_max=None,
               traj_dq_max=0.0):
    """궤적 생성 → (ref(t)→(θ_d,θ̇_d,θ̈_d), T, traj_error).

    joint: 관절 quintic(해석적, 항상 실행가능).
    joint_timeopt: 같은 경로, 토크 한계 역산 타이밍(_make_ref_timeopt).
    task : 직교 SE(3) 직선보간 → IK 샘플(precompute) → θ_d 격자 보간.
           IK 비수렴(작업영역밖) 또는 관절속도 폭발(특이점) 시 그 지점까지 자르고
           traj_error 를 채운다(='실행 불가'를 정직하게 표면화).
    """
    N = robot.n
    if traj_mode == "joint_timeopt":
        ref, T, traj_error = _make_ref_timeopt(WP, hold, robot,
                                               traj_tau_max, traj_dq_max)
        if ref is None:
            def ref(_t):
                return WP[0].copy(), np.zeros(N), np.zeros(N)
        return ref, T, traj_error
    if traj_mode != "task":
        ref, T = _make_ref(WP, t_seg, hold, robot)
        return ref, T, None

    n_seg = len(WP) - 1
    seg = t_seg + hold
    T = n_seg * seg
    poses = [robot.fk(w) for w in WP]             # 웨이포인트 SE(3)
    ts = np.linspace(0, T, int(T * 200) + 1)    # 조밀 격자(200 Hz)
    TH = np.zeros((len(ts), N))
    seed = WP[0].copy()
    traj_error = None
    n_ok = len(ts)
    for i, t in enumerate(ts):
        k = min(int(t // seg), n_seg - 1)
        loc = t - k * seg
        s = quintic_time_scaling(loc, t_seg)[0] if loc < t_seg else 1.0
        Xd = _se3_straight(poses[k], poses[k + 1], s)
        th, ok = robot.ik(Xd, seed)
        if not ok:
            traj_error = f"직교 직선이 작업영역/특이점에 막힘 (t≈{t:.2f}s) — 실행 불가"
            n_ok = i
            break
        TH[i], seed = th, th
    ts, TH = ts[:n_ok], TH[:n_ok]
    if len(ts) < 2:
        return (lambda _t: (WP[0].copy(), np.zeros(N), np.zeros(N))), 0.0, traj_error

    dt = ts[1] - ts[0]
    DTH = np.gradient(TH, dt, axis=0)
    # 특이점 부근 관절속도 폭발 감지 → 그 지점까지 절단
    over = np.where(np.max(np.abs(DTH), axis=1) > OMEGA_MAX)[0]
    if traj_error is None and len(over):
        cut = max(over[0], 2)
        traj_error = f"특이점 부근 관절속도 폭발 (t≈{ts[cut - 1]:.2f}s) — 실행 불가"
        ts, TH, DTH = ts[:cut], TH[:cut], DTH[:cut]
    DDTH = np.gradient(DTH, dt, axis=0)
    T_eff = float(ts[-1])

    def ref(t):
        tc = min(max(t, 0.0), T_eff)
        thd = np.array([np.interp(tc, ts, TH[:, j]) for j in range(N)])
        dthd = np.array([np.interp(tc, ts, DTH[:, j]) for j in range(N)])
        ddthd = np.array([np.interp(tc, ts, DDTH[:, j]) for j in range(N)])
        return thd, dthd, ddthd

    return ref, T_eff, traj_error


class _SolShim:
    """이산 ZOH 적분 결과를 solve_ivp 의 sol(.t/.y/.status)과 호환되게 담는 객체.
    + tau_log(샘플별 실제 인가 토크, 노이즈 chatter 포함)."""
    __slots__ = ("t", "y", "status", "tau_log", "diverged", "settle_t")


def _empty_result(WP, traj_error, robot):
    """궤적 생성 실패 시 빈 결과(재생할 것 없음 + 사유)."""
    return {
        "t": [], "theta": [], "tcp": [], "error": [], "torque": [], "qdot": [],
        "waypoints_tcp": [robot.fk(w)[:3, 3].tolist() for w in WP],
        "settle_time": None, "steady_state_error": None,
        "diverged": True, "traj_error": traj_error,
    }


def _blowup_event(N):
    """연속 적분(Radau)용: 위치/속도 폭주 시 즉시 종료(발산 적분 길어짐 방지)."""
    def blowup(t, x):
        return min(THETA_MAX - float(np.max(np.abs(x[:N]))),
                   OMEGA_MAX - float(np.max(np.abs(x[N:2 * N]))))
    blowup.terminal = True
    blowup.direction = -1
    return blowup


def _detect_settle(sol, T, err, vmax, hz, require_excite=False):
    """정상상태(평형) 도달 판정. → (diverged, settle_time, steady_state_error).

    멈춤 = 모션(T) 이후 속도가 VEL_WIN 동안 VEL_TOL 이하 유지. 잔류오차가 있어도
    평형이면 멈춤(그 시점 오차 = 정상상태 오차). 끝까지 안 멈추고 오차 증가추세면 발산.

    require_excite: 외란은 모션 종료(T) 후 인가되므로, 인가 전 정지 상태를 평형으로
    오판하지 않도록 '한 번 흔들린 뒤(excited)' 평형만 인정한다(어드미턴스와 동일 규약).
    """
    diverged = sol.status != 0          # 안전망 이벤트 발동 or 적분 실패
    if diverged:
        return True, None, None

    vw = max(1, int(VEL_WIN * hz))
    rest = None
    excited = not require_excite
    for i in range(len(sol.t)):
        if sol.t[i] < T:
            continue
        if vmax[i] >= VEL_TOL:
            excited = True
        if excited and np.max(vmax[max(0, i - vw):i + 1]) < VEL_TOL:
            rest = i
            break
    if rest is not None:
        return False, float(sol.t[rest]), float(err[rest])

    # 평형 미도달 + 오차 증가추세 + 절대오차도 유의미하게 클 때만 발산 판정.
    # (이산 ZOH 는 미세 task오차에서 trend 만으로 오판하기 쉬움 → 절대 임계 추가)
    win = max(1, int(0.5 * hz))
    if (len(err) > win and err[-1] > err[-1 - win] * 1.05
            and err[-1] > 0.05):
        return True, None, None
    return False, None, None


def _keep_idx(sol, settle_time, diverged, T):
    """반환 구간: 평형 시 그 시점+마진, 그 외(발산·미도달)는 계산한 전체.

    미도달(settle=None)일 때 임의 시점(예: T+1s)에서 자르면 이동 중인 모션이 화면에서
    뚝 끊겨 보인다 → 적분한 전체를 보여줘 끊김을 없앤다(느린/언더댐프 응답 대비).
    """
    if settle_time is not None:
        keep = sol.t <= settle_time + 0.3
    else:
        keep = np.ones(len(sol.t), dtype=bool)
    return np.where(keep)[0]


# ----------------------------------------------------------------------
#  동역학 plant — 토크 입력 (PD / PID / Computed Torque / 임피던스)
# ----------------------------------------------------------------------
def _run_torque(WP, ref, T, controller, gains, gravity_comp, hz,
                plant, ctrl, disturbance, friction=0.0, tau_max=0.0,
                control_rate=0.0, noise=0.0, noise_seed=0, robot=None):
    N = robot.n
    Kp = float(gains.get("kp", 100.0))
    Kd = float(gains.get("kd", 20.0))
    Ki = float(gains.get("ki", 0.0))
    plant = plant or robot.dyn_args
    ctrl = ctrl or robot.dyn_args
    # 모델을 numba 친화적 스택 배열로 1회 변환 (rhs 안에서 매번 변환하지 않도록)
    Mp, Gp, Sp = df.stack_model(*plant)
    Mc, Gc, Sc = df.stack_model(*ctrl)

    def g_ctrl(th):
        return df.gravity_forces(th, robot.gravity, Mc, Gc, Sc)

    use_I = controller in ("pid", "computed_torque")

    def torque(th, dth, th_d, dth_d, ddth_d, I):
        e, ed = th_d - th, dth_d - dth
        if controller == "computed_torque":
            aq = ddth_d + Kd * ed + Kp * e + Ki * I
            M = df.mass_matrix(th, Mc, Gc, Sc)
            c = df.coriolis_forces(th, dth, Mc, Gc, Sc)
            return M @ aq + c + g_ctrl(th)
        if controller == "pid":
            tau = Kp * e + Ki * I + Kd * ed
            return tau + g_ctrl(th) if gravity_comp else tau
        if controller == "impedance":
            # 직교(전체 pose) 임피던스: body twist 오차에 스프링-댐퍼.
            # V_err = log(T⁻¹·T_d) (body twist), τ = Jbᵀ(Kp·V_err − Kd·Jb·θ̇) + g
            Verr = SE3.log(np.linalg.inv(robot.fk(th)) @ robot.fk(th_d))[0]
            Jb = robot.jacobian_body(th)
            F = Kp * Verr - Kd * (Jb @ dth)
            return Jb.T @ F + g_ctrl(th)
        tau = Kp * e + Kd * ed                  # pd
        return tau + g_ctrl(th) if gravity_comp else tau

    # 외란: base 프레임 외력(N)을 모션 종료(T) 후 끝단에 인가.
    # 컨트롤러는 모르는 힘(plant 의 Ftip) → PID/CT 는 버티고 임피던스는 밀린다.
    dist = np.asarray(disturbance, dtype=float) if disturbance else None
    has_dist = dist is not None and np.any(dist)

    def Ftip(t, th):
        if not has_dist or t < T:
            return np.zeros(6)
        # 외란 = 로봇에 가해지는 외력. forward_dynamics 의 Ftip 규약은
        # 'EE 가 환경에 가하는 렌치'(MR 표준, θ̈=M⁻¹(τ−c−g−Jᵀ·Ftip)) 이므로
        # 로봇에 +dist 를 가하려면 Ftip=−dist 로 넣어야 한다(힘제어 벽 규약과 동일).
        f_b = -robot.fk(th)[:3, :3].T @ dist    # base→EE 프레임, 부호 반전
        return np.concatenate([np.zeros(3), f_b])  # [모멘트; 힘] (EE 프레임)

    def applied(th, dth, th_d, dth_d, ddth_d, I):
        """액추에이터가 실제 내는 토크 = 명령 토크에 포화 적용(plot 도 이 값)."""
        tau = torque(th, dth, th_d, dth_d, ddth_d, I)
        return np.clip(tau, -tau_max, tau_max) if tau_max > 0 else tau

    # 제어 모드: control_rate=0 → 연속(이상화, 빠른 Radau) / >0 → 이산 ZOH(현실).
    # 노이즈는 이산 샘플링이 필요하므로 control_rate=0 이면 기본 제어율로 이산화한다.
    # T_cap 은 고정 대기가 아니라 안전 상한 — 실제로는 정착 순간 adaptive 종료한다.
    T_cap = T + SETTLE_HARD
    rate = control_rate
    if noise > 0 and rate <= 0:
        rate = 1000.0

    def _continuous():
        """연속 제어(제어율 ∞ 이상화) — 암시적 Radau. 빠르고 항상 안정적.
        고정 시간이 아니라 '진짜 평형(속도·가속도 모두 작음)'에서 adaptive 종료한다."""
        def rhs(t, x):
            th, dth = x[:N], x[N:2 * N]
            I = x[2 * N:] if use_I else np.zeros(N)
            th_d, dth_d, ddth_d = ref(t)
            tau = applied(th, dth, th_d, dth_d, ddth_d, I)
            if friction > 0:
                tau = tau - friction * np.tanh(dth / 0.02)
            ddth = df.forward_dynamics(th, dth, tau, robot.gravity, Ftip(t, th), Mp, Gp, Sp)
            parts = [dth, ddth] + ([th_d - th] if use_I else [])
            return np.concatenate(parts)

        # adaptive 종료 이벤트: 모션(T) 이후 속도·가속도가 모두 임계 이하 → 평형 → 정지.
        # 가속도도 보므로 언더댐프 진동의 꼭대기(속도0·가속도 최대)에선 안 멈춘다.
        def rest(t, x):
            if t < T + VEL_WIN:                     # 모션/외란 전 + 정착창 확보 전엔 비활성
                return 1.0
            th, dth = x[:N], x[N:2 * N]
            I = x[2 * N:] if use_I else np.zeros(N)
            th_d, dth_d, ddth_d = ref(t)
            tau = applied(th, dth, th_d, dth_d, ddth_d, I)
            if friction > 0:
                tau = tau - friction * np.tanh(dth / 0.02)
            ddth = df.forward_dynamics(th, dth, tau, robot.gravity, Ftip(t, th), Mp, Gp, Sp)
            return max(np.max(np.abs(dth)) / VEL_TOL,
                       np.max(np.abs(ddth)) / ACC_TOL) - 1.0
        rest.terminal = True
        rest.direction = -1

        x0 = np.concatenate([WP[0], np.zeros(N)] + ([np.zeros(N)] if use_I else []))
        t_eval = np.linspace(0, T_cap, int(T_cap * hz) + 1)
        s = solve_ivp(rhs, (0, T_cap), x0, t_eval=t_eval, method="Radau",
                      rtol=1e-4, atol=1e-7, events=[_blowup_event(N), rest])
        s.diverged = len(s.t_events[0]) > 0         # blowup 발동 = 발산
        s.settle_t = (float(s.t_events[1][0])       # rest 발동 = 정착 시각
                      if len(s.t_events[1]) else None)
        if s.settle_t is not None and len(s.t):     # 정착(거의 정지) 프레임을 끝에 추가
            s.t = np.append(s.t, s.settle_t)        # (t_eval 은 이벤트 직전까지만 반환)
            s.y = np.column_stack([s.y, s.y_events[1][0]])
        s.tau_log = None                            # 패킹 때 applied()로 재계산
        return s

    def _zoh(rate):
        """이산 ZOH 제어(실제 디지털 제어) — rate(Hz) 샘플·토크 ZOH 유지, plant 는 RK4.
        제어율이 낮으면 불안정해지는 실제 현상이 그대로 나타난다. 노이즈 주입 가능."""
        h = 1.0 / rate
        n_ticks = int(T_cap * rate) + 1
        out_every = max(1, round(rate / hz))
        if noise > 0:
            rng = np.random.default_rng(noise_seed)  # seed 고정 = 재현성(같은 realization)
            n_pos = noise * rng.standard_normal((n_ticks, N))
            n_vel = 5.0 * noise * rng.standard_normal((n_ticks, N))
        th, dth, I = WP[0].copy(), np.zeros(N), np.zeros(N)
        excited_z = not has_dist            # 외란 시 흔들린 뒤에만 정착 인정
        ts, THl, DTHl, IIl, taul = [], [], [], [], []
        diverged = False
        settle_t = None
        for k in range(n_ticks):
            t = k * h
            th_d, dth_d, ddth_d = ref(t)
            thm = th + n_pos[k] if noise > 0 else th    # 센서 측정값(노이즈 포함)
            dthm = dth + n_vel[k] if noise > 0 else dth
            tau = applied(thm, dthm, th_d, dth_d, ddth_d, I)   # ZOH: h 동안 유지

            def f(x):                               # plant: tau 고정, 마찰 연속
                q, qd = x[:N], x[N:]
                tn = tau - friction * np.tanh(qd / 0.02) if friction > 0 else tau
                qdd = df.forward_dynamics(q, qd, tn, robot.gravity, Ftip(t, q), Mp, Gp, Sp)
                return np.concatenate([qd, qdd])

            x = np.concatenate([th, dth])
            k1 = f(x); k2 = f(x + 0.5 * h * k1)
            k3 = f(x + 0.5 * h * k2); k4 = f(x + h * k3)
            x = x + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            th, dth = x[:N], x[N:]
            if (not np.all(np.isfinite(x))
                    or np.max(np.abs(th)) > THETA_MAX
                    or np.max(np.abs(dth)) > OMEGA_MAX):
                diverged = True
                break
            if use_I:
                I = I + (th_d - thm) * h             # 이산 적분(측정 오차 기준)
            # adaptive 종료: 모션 후 흔들린 뒤 속도·가속도 모두 작으면 평형 → 정지
            rest = False
            if t >= T + VEL_WIN:
                if np.max(np.abs(dth)) >= VEL_TOL:
                    excited_z = True
                if excited_z and np.max(np.abs(dth)) < VEL_TOL:
                    qdd = f(np.concatenate([th, dth]))[N:]
                    rest = np.max(np.abs(qdd)) < ACC_TOL
            if k % out_every == 0 or rest:
                ts.append(t); THl.append(th.copy()); DTHl.append(dth.copy())
                IIl.append(I.copy()); taul.append(tau.copy())
            if rest:
                settle_t = t
                break

        s = _SolShim()
        s.t = np.array(ts)
        rows = [np.array(THl).T, np.array(DTHl).T] + ([np.array(IIl).T] if use_I else [])
        s.y = np.vstack(rows) if ts else np.zeros((2 * N + (N if use_I else 0), 0))
        s.status = 1 if diverged else 0
        s.diverged = diverged
        s.settle_t = settle_t
        s.tau_log = np.array(taul) if ts else np.zeros((0, N))
        return s

    sol = _zoh(rate) if rate > 0 else _continuous()

    TH, DTH = sol.y[:N].T, sol.y[N:2 * N].T
    II = sol.y[2 * N:].T if use_I else np.zeros((len(sol.t), N))
    # 추종오차: 임피던스는 task-space(TCP 위치 m), 그 외는 joint-space(rad).
    # (임피던스는 같은 pose 를 다른 관절 config 로 도달할 수 있어 joint 오차가 무의미)
    if controller == "impedance":
        err = np.array([np.linalg.norm(robot.fk(TH[i])[:3, 3] - robot.fk(ref(t)[0])[:3, 3])
                        for i, t in enumerate(sol.t)])
    else:
        err = np.array([np.linalg.norm(TH[i] - ref(t)[0]) for i, t in enumerate(sol.t)])
    # 정착/발산은 adaptive 종료 이벤트(연속) / _zoh 플래그(이산)에서 직접 — 윈도우
    # 스캔보다 정확하고, 외란 인가 전 정지를 오판하지 않는다(rest 이벤트가 T+VEL_WIN 게이트).
    diverged = sol.diverged
    settle_time = sol.settle_t
    # 안전망(blowup)에 안 걸린 '느린 발산'(오차 증가추세) 보강
    if not diverged and settle_time is None and len(err) > int(0.5 * hz):
        win = int(0.5 * hz)
        if err[-1] > err[-1 - win] * 1.05 and err[-1] > 0.05:
            diverged = True
    steady_state_error = (float(err[int(np.argmin(np.abs(sol.t - settle_time)))])
                          if settle_time is not None and len(err) else None)
    idx = _keep_idx(sol, settle_time, diverged, T)

    tcp, error, torque_log = [], [], []
    for i in idx:
        tcp.append(robot.fk(TH[i])[:3, 3].tolist())
        error.append(float(err[i]))
        if sol.tau_log is not None:                 # 이산: 기록된 ZOH 토크(노이즈 포함)
            torque_log.append([float(v) for v in sol.tau_log[i]])
        else:                                       # 연속: applied()로 재계산
            th_d, dth_d, ddth_d = ref(sol.t[i])
            torque_log.append([float(v)
                               for v in applied(TH[i], DTH[i], th_d, dth_d, ddth_d, II[i])])

    return {
        "t": sol.t[idx].tolist(),
        "theta": TH[idx].tolist(),
        "tcp": tcp,
        "error": error,
        "torque": torque_log,
        "qdot": [],
        "waypoints_tcp": [robot.fk(w)[:3, 3].tolist() for w in WP],
        "settle_time": settle_time,
        "steady_state_error": steady_state_error,
        "diverged": bool(diverged),
    }


# ----------------------------------------------------------------------
#  기구학 plant — 속도 입력 (resolved-rate, MR §11.3)
# ----------------------------------------------------------------------
def null_gradient(robot, th, mode, k0):
    """부가목표 H 의 관절공간 그래디언트 ∇H (EE 무관, 널공간 투영 전 원시 방향).
    리던던시(여유 DOF) 해소에 쓴다 — 널공간 투영 후 EE 는 안 건드린다.
      posture        : ∇H = −(θ−θ_ready)  → 편한 자세로 수렴(joint centering)
      manipulability : ∇H = ∇w  (수치 중앙차분) → 특이점에서 멀어짐(w=√det(JJᵀ)↑)
    """
    if mode == "posture":
        return -k0 * (th - robot.ready)
    if mode == "manipulability":
        h = 1e-4
        grad = np.zeros(robot.n)
        for i in range(robot.n):
            tp = th.copy(); tp[i] += h
            tm = th.copy(); tm[i] -= h
            grad[i] = (robot.manipulability(tp)["w"]
                       - robot.manipulability(tm)["w"]) / (2.0 * h)
        return k0 * grad
    return np.zeros(robot.n)


def _run_velocity(WP, ref, T, controller, gains, hz, robot,
                  null_obj=None, null_gain=1.0):
    """속도제어: θ̇=제어출력, θ만 적분(1차계). 동역학·중력 불필요 → RK45.

    joint_velocity (§11.3.2): θ̇ = θ̇_d + Kp(θ_d−θ) + Ki∫(θ_d−θ)
    resolved_rate   (§11.3.3): Vb = Ad_{X⁻¹X_d}·V_d + Kp·Xe + Ki∫Xe,  θ̇ = J†_b·Vb
                               Xe = log(X⁻¹X_d) (body twist 오차)
    게인은 1차계라 스케일이 작다(≈ 1/시정수). kd 없음.

    널공간(여유 DOF) 해소: θ̇ = J†Vb + (I−J†J)·θ̇₀ (resolved_rate 한정).
    투영 (I−J†J) 이 θ̇₀ 의 EE 영향분을 제거 → 주작업 무손상. 6-DOF 는 J 정방이라
    J†J=I → 투영≈0 → 완전 불변. θ̇₀ 는 null_gradient(부가목표).
    """
    N = robot.n
    Kp = float(gains.get("kp", 2.0))
    Ki = float(gains.get("ki", 0.0))
    use_I = Ki != 0.0
    is_rr = controller == "resolved_rate"
    idim = 6 if is_rr else N             # 적분항 차원 (task=6, joint=n)
    null_mode = null_obj if is_rr else None    # 널공간은 resolved_rate(여유 task)만

    def law(th, th_d, dth_d, I):
        """제어출력 θ̇ 와 적분항 integrand 반환."""
        if not is_rr:                                   # joint_velocity
            e = th_d - th
            return dth_d + Kp * e + Ki * I, e
        X, Xd = robot.fk(th), robot.fk(th_d)
        Tdiff = np.linalg.inv(X) @ Xd
        Xe = SE3.log(Tdiff)[0]                          # body twist 오차
        Vd = robot.jacobian_body(th_d) @ dth_d           # 피드포워드 트위스트(frame {d})
        Vb = SE3.Adjoint(Tdiff) @ Vd + Kp * Xe + Ki * I
        Jb = robot.jacobian_body(th)
        Jpinv = robot.dls_inv(Jb)
        qd = Jpinv @ Vb
        if null_mode:                                   # 널공간 부가목표(여유 DOF)
            qd0 = null_gradient(robot, th, null_mode, null_gain)
            qd = qd + (np.eye(N) - Jpinv @ Jb) @ qd0
        return qd, Xe

    def rhs(t, x):
        th = x[:N]
        I = x[N:N + idim] if use_I else np.zeros(idim)
        th_d, dth_d, _ = ref(t)
        qd, integrand = law(th, th_d, dth_d, I)
        return np.concatenate([qd] + ([integrand] if use_I else []))

    x0 = np.concatenate([WP[0]] + ([np.zeros(idim)] if use_I else []))
    T_cap = T + SETTLE_MAX
    t_eval = np.linspace(0, T_cap, int(T_cap * hz) + 1)

    # 기구학 1차계는 stiff 아님 → RK45. 위치만 폭주 가드(속도는 제어식이 유계).
    def blowup(t, x):
        return THETA_MAX - float(np.max(np.abs(x[:N])))
    blowup.terminal = True
    blowup.direction = -1
    sol = solve_ivp(rhs, (0, T_cap), x0, t_eval=t_eval, method="RK45",
                    rtol=1e-5, atol=1e-8, events=blowup)

    TH = sol.y[:N].T
    II = sol.y[N:N + idim].T if use_I else np.zeros((len(sol.t), idim))

    # 전 구간 θ̇(=제어출력)·오차 재계산 → 정착판정용 vmax 확보
    QD, err = [], []
    for i, t in enumerate(sol.t):
        th_d, dth_d, _ = ref(t)
        qd, _ = law(TH[i], th_d, dth_d, II[i])
        QD.append(qd)
        if is_rr:                                       # resolved-rate=task 오차(m)
            err.append(float(np.linalg.norm(robot.fk(TH[i])[:3, 3] - robot.fk(th_d)[:3, 3])))
        else:                                           # joint 오차(rad)
            err.append(float(np.linalg.norm(TH[i] - th_d)))
    QD = np.array(QD) if len(QD) else np.zeros((0, N))
    err = np.array(err)
    vmax = np.max(np.abs(QD), axis=1) if len(QD) else np.zeros(0)

    diverged, settle_time, steady_state_error = _detect_settle(sol, T, err, vmax, hz)
    idx = _keep_idx(sol, settle_time, diverged, T)

    return {
        "t": sol.t[idx].tolist(),
        "theta": TH[idx].tolist(),
        "tcp": [robot.fk(TH[i])[:3, 3].tolist() for i in idx],
        "error": [err[i] for i in idx],
        "torque": [],
        "qdot": QD[idx].tolist(),
        "waypoints_tcp": [robot.fk(w)[:3, 3].tolist() for w in WP],
        "settle_time": settle_time,
        "steady_state_error": steady_state_error,
        "diverged": bool(diverged),
    }


# ----------------------------------------------------------------------
#  어드미턴스 (MR §11.7.2) — 외력을 '센싱'해 움직임 생성 (임피던스의 쌍)
# ----------------------------------------------------------------------
def _run_admittance(WP, ref, T, gains, hz, disturbance, robot):
    """어드미턴스: 가상 질량-스프링-댐퍼를 시뮬레이션해 외력→움직임을 만든다.

    M·Δẍ + B·Δẋ + K·Δp = f_ext  (base 프레임 3D 병진, MR 식 11.66).
    그 가상 변위 Δp 를 nominal setpoint 에 더한 목표 pose 를 resolved-rate 로 추종
    (모션 plant — 임피던스가 토크 plant 인 것과 대칭: 어드미턴스 로봇=모션제어).
    외란을 인가하면 로봇이 그 힘에 능동적으로 양보하며 움직인다(정상상태 Δp=f/K).
    힘센서 없는 시뮬에선 외란(이미 아는 값)을 f_ext 로 재사용. M>0 라 2차 응답(오버슛).
    """
    N = robot.n
    M = max(float(gains.get("m", 2.0)), 0.1)    # 가상 질량 kg (0 방지)
    B = float(gains.get("b", 40.0))             # 가상 감쇠 N·s/m
    K = float(gains.get("k", 600.0))            # 가상 강성 N/m
    KP_TRACK = 20.0                            # 내부 모션 추종(resolved-rate) 게인
    # ↑ 가상 동역학(ωn=√(K/M))보다 빨라야 M-B-K 응답(오버슛 등)이 가려지지 않음

    dist = np.asarray(disturbance, dtype=float) if disturbance else None
    has_dist = dist is not None and np.any(dist)

    def f_ext(t):                               # 외란 = 센싱된 외력 (모션 종료 후 인가)
        return dist if (has_dist and t >= T) else np.zeros(3)

    def rhs(t, x):
        th, dp, dv = x[:N], x[N:N + 3], x[N + 3:N + 6]
        T_set = robot.fk(ref(t)[0])                      # nominal setpoint pose
        da = (f_ext(t) - B * dv - K * dp) / M          # 가상 어드미턴스 동역학
        T_d = T_set.copy()
        T_d[:3, 3] = T_set[:3, 3] + dp                 # 목표 = setpoint + 가상 변위
        Xe = SE3.log(np.linalg.inv(robot.fk(th)) @ T_d)[0]
        qd = robot.dls_inv(robot.jacobian_body(th)) @ (KP_TRACK * Xe)
        return np.concatenate([qd, dv, da])

    x0 = np.concatenate([WP[0], np.zeros(6)])
    T_cap = T + SETTLE_HARD                      # 안전 상한 — 실제론 정착서 adaptive 종료
    t_eval = np.linspace(0, T_cap, int(T_cap * hz) + 1)

    def blowup(t, x):
        return THETA_MAX - float(np.max(np.abs(x[:N])))
    blowup.terminal = True
    blowup.direction = -1

    # adaptive 종료: 모션(T) 후 '로봇이 실제로 멈춤(qd 작음) + 가상목표 가속도 작음'.
    # 로봇 추종(qd)이 가상 동역학(dv)보다 늦으므로 qd 기준이라야 진짜 정지에서 끊는다.
    # 가속도(da)도 보므로 언더댐프 진동 꼭대기(qd≈0·da↑)에선 조기 종료하지 않는다.
    def rest(t, x):
        if t < T + VEL_WIN:
            return 1.0
        th, dp, dv = x[:N], x[N:N + 3], x[N + 3:N + 6]
        da = (f_ext(t) - B * dv - K * dp) / M
        T_set = robot.fk(ref(t)[0])
        T_d = T_set.copy()
        T_d[:3, 3] = T_set[:3, 3] + dp
        Xe = SE3.log(np.linalg.inv(robot.fk(th)) @ T_d)[0]
        qd = robot.dls_inv(robot.jacobian_body(th)) @ (KP_TRACK * Xe)
        return max(np.max(np.abs(qd)) / VEL_TOL,
                   np.max(np.abs(da)) / ACC_TOL) - 1.0
    rest.terminal = True
    rest.direction = -1
    sol = solve_ivp(rhs, (0, T_cap), x0, t_eval=t_eval, method="RK45",
                    rtol=1e-5, atol=1e-8, events=[blowup, rest])
    if len(sol.t_events[1]) and len(sol.t):         # 정착(거의 정지) 프레임을 끝에 추가
        sol.t = np.append(sol.t, sol.t_events[1][0])
        sol.y = np.column_stack([sol.y, sol.y_events[1][0]])

    TH = sol.y[:N].T
    QD, err = [], []
    for i, t in enumerate(sol.t):
        T_set = robot.fk(ref(t)[0])
        dp = sol.y[N:N + 3, i]
        T_d = T_set.copy()
        T_d[:3, 3] = T_set[:3, 3] + dp
        Xe = SE3.log(np.linalg.inv(robot.fk(TH[i])) @ T_d)[0]
        QD.append(robot.dls_inv(robot.jacobian_body(TH[i])) @ (KP_TRACK * Xe))
        # 추종오차 = 실제 EE 가 nominal setpoint 에서 벗어난 변위 (= 컴플라이언스)
        err.append(float(np.linalg.norm(robot.fk(TH[i])[:3, 3] - T_set[:3, 3])))
    QD = np.array(QD) if len(QD) else np.zeros((0, N))
    err = np.array(err)

    # 정착/발산은 종료 이벤트에서 직접: blowup=발산, rest=정착(로봇 정지+가상 평형).
    # 둘 다 terminal 이라 status 로는 구분 못 하므로 t_events 로 가른다.
    diverged = len(sol.t_events[0]) > 0
    settle_time = (float(sol.t_events[1][0]) if len(sol.t_events[1]) else None)
    steady_state_error = (float(err[int(np.argmin(np.abs(sol.t - settle_time)))])
                          if settle_time is not None and len(err) else None)

    idx = _keep_idx(sol, settle_time, diverged, T)

    return {
        "t": sol.t[idx].tolist(),
        "theta": TH[idx].tolist(),
        "tcp": [robot.fk(TH[i])[:3, 3].tolist() for i in idx],
        "error": [err[i] for i in idx],
        "torque": [],
        "qdot": QD[idx].tolist(),
        "waypoints_tcp": [robot.fk(w)[:3, 3].tolist() for w in WP],
        "settle_time": settle_time,
        "steady_state_error": steady_state_error,
        "diverged": bool(diverged),
    }


# ----------------------------------------------------------------------
#  컴플라이언트 평면 벽 (힘·하이브리드 제어 공용 환경 모델)
# ----------------------------------------------------------------------
def _wall_contact(n, p_w, k_env, b_env, robot):
    """penalty 평면 벽 접촉 함수 factory. 반환 contact(th,dth) →
    (f_meas 법선반력 N, δ 침투깊이 m, F_env 반력 3-vec base, +n 방향)."""
    def contact(th, dth):
        Tee = robot.fk(th)
        R, p = Tee[:3, :3], Tee[:3, 3]
        delta = max(0.0, -float(n @ (p - p_w)))             # 평면 침투깊이(+면 비접촉)
        if delta <= 0.0:
            return 0.0, 0.0, np.zeros(3)
        pdot = R @ (robot.jacobian_body(th) @ dth)[3:6]     # base EE 선속도
        ddelta = -float(n @ pdot)                           # 침투 속도
        f_meas = max(0.0, k_env * delta + b_env * ddelta)   # 법선반력(벽은 밀기만)
        return f_meas, delta, n * f_meas                    # 반력은 +n(로봇 밀어냄)
    return contact


# ----------------------------------------------------------------------
#  곡면 환경 모델 (하이브리드 컨투어 추종 — 평면/원기둥/구)
# ----------------------------------------------------------------------
def _make_surface(ft):
    """force_task 표면 사양 → surface(p) → (n 외향단위법선, δ 침투깊이, p_proj 표면투영).

    shape: 'plane'(기본) | 'cylinder' | 'sphere' | 'mesh'. 곡면은 법선 n(p)가 위치종속
    (MR §11.6 의 구속 A(θ)) — 제어식(투영 P)은 동일하고 매 스텝 접촉점 법선만 다르게
    들어간다. 침투 δ = max(0, R−거리): EE 가 표면 안으로 들어간 깊이. δ̇=−n·ṗ(호출측).
    mesh 는 임의 모델(trimesh): 최근접점·면법선으로 동일 인터페이스 제공(컨트롤러가
    ft['_pq']=ProximityQuery, ft['_mesh']=배치된 trimesh 를 실어 보낸다).
    """
    shape = ft.get("shape", "plane")
    if shape == "mesh":
        import trimesh  # 보간용(점→바리센트릭)
        mesh = ft["_mesh"]                            # 배치(transform 적용)된 trimesh
        pq = ft["_pq"]                               # trimesh ProximityQuery(BVH 캐시)
        faces, verts = mesh.faces, mesh.vertices
        vnorm, fnorm = mesh.vertex_normals, mesh.face_normals

        def surf(p):
            # 면 법선은 면을 넘을 때 불연속(힘 chatter) → 정점법선을 바리센트릭 보간해
            # 매끄러운 법선장을 만든다(해석적 곡면처럼 연속).
            p = np.asarray(p, float)
            cp, _, tri = pq.on_surface(p.reshape(1, 3))   # 최근접 표면점·삼각형
            cp, ti = cp[0], int(tri[0])
            tri_v = verts[faces[ti]]                 # 3×3 삼각형 정점
            bary = trimesh.triangles.points_to_barycentric(
                tri_v.reshape(1, 3, 3), cp.reshape(1, 3))[0]
            n = (bary[:, None] * vnorm[faces[ti]]).sum(0)
            nn = float(np.linalg.norm(n))
            n = n / nn if nn > 1e-9 else fnorm[ti]
            signed = float((p - cp) @ n)             # >0 표면 밖, <0 침투
            return n, max(0.0, -signed), cp

        return surf
    if shape == "sphere":
        c = np.asarray(ft.get("center", [0.0, 0.0, 0.0]), float)
        R = float(ft.get("radius", 0.1))

        def surf(p):
            d = p - c
            rho = float(np.linalg.norm(d))
            n = d / rho if rho > 1e-9 else np.array([0.0, 0.0, 1.0])
            return n, max(0.0, R - rho), c + R * n

        return surf
    if shape == "cylinder":
        c = np.asarray(ft.get("center", [0.0, 0.0, 0.0]), float)
        a = np.asarray(ft.get("axis", [1.0, 0.0, 0.0]), float)
        a = a / np.linalg.norm(a)
        R = float(ft.get("radius", 0.1))

        def surf(p):
            d = p - c
            ax = float(d @ a)                       # 축방향 성분
            d_perp = d - ax * a
            rho = float(np.linalg.norm(d_perp))
            n = d_perp / rho if rho > 1e-9 else np.array([1.0, 0.0, 0.0])
            return n, max(0.0, R - rho), c + ax * a + R * n

        return surf
    # plane (기본, 현행과 동일)
    n0 = np.asarray(ft.get("normal", [0.0, 0.0, 1.0]), float)
    n0 = n0 / np.linalg.norm(n0)
    p_w = np.asarray(ft.get("point", [0.0, 0.0, 0.0]), float)

    def surf(p):
        signed = float(n0 @ (p - p_w))
        return n0, max(0.0, -signed), p - signed * n0

    return surf


def _surface_contact(surf, k_env, b_env, robot):
    """penalty 곡면/평면 접촉 factory. 반환 contact(th,dth) →
    (f_meas 법선반력 N, δ 침투깊이 m, F_env 반력 3-vec base +n). 표면은 surf 가 정의."""
    def contact(th, dth):
        Tee = robot.fk(th)
        R, p = Tee[:3, :3], Tee[:3, 3]
        n, delta, _ = surf(p)
        if delta <= 0.0:
            return 0.0, 0.0, np.zeros(3)
        pdot = R @ (robot.jacobian_body(th) @ dth)[3:6]     # base EE 선속도
        ddelta = -float(n @ pdot)                           # 침투 속도(δ̇=−n·ṗ)
        f_meas = max(0.0, k_env * delta + b_env * ddelta)
        return f_meas, delta, n * f_meas

    return contact


def _align_rot(a, b):
    """단위벡터 a 를 b 로 보내는 최소회전 행렬 (Rodrigues). a≈b 면 단위행렬.
    Level 2 컨투어: 공구 z축(시작 누름방향)을 접촉점 법선에 정렬하는 데 쓴다."""
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    v = np.cross(a, b)
    s = float(np.linalg.norm(v))
    if s < 1e-9:                                  # 평행(정상) — 회전 없음
        return np.eye(3)
    c = float(np.clip(a @ b, -1.0, 1.0))
    return SO3.exp(v / s * np.arccos(c))


# ----------------------------------------------------------------------
#  토크 plant — 힘 제어 (MR §11.5, 식 11.51/11.54)
# ----------------------------------------------------------------------
def _run_force(WP, gains, hz, plant, force_task, robot):
    """task-space 힘 제어: 가상 벽을 목표 힘으로 누른다 (토크 plant).

    τ = g̃(θ) + Jbᵀ(Fd + Kfp·Fe + Kfi∫Fe − Kdamp·V)   (MR 식 11.54)
    여기서 Fe = Fd − F_meas (힘 오차), V = Jb·θ̇ (body twist).

    환경은 컴플라이언트(penalty) 평면 벽: 침투 시 법선 반력 f = K_env·δ + B_env·δ̇.
    (교재 §11.6 은 강체 구속 A(θ)V=0 을 가정하나 수치적으로 불안정 → §11.7 이 밝히듯
    실제 환경은 컴플라이언스가 불가피하므로 강성 큰 스프링-댐퍼로 근사한다.)
    이 반력이 곧 '힘센서 측정값' F_meas 이자 plant 의 Ftip 이 된다(폐루프 힘 피드백).

    force_task: {normal(외향 단위법선,base), point(벽면 한 점,base), fd(목표 누름힘 N),
                 k_env, b_env, t_end}.  gains: {kfp, kfi, kdamp}.
    """
    ft = force_task or {}
    n = np.asarray(ft.get("normal", [0.0, 0.0, 1.0]), float)
    n = n / np.linalg.norm(n)                    # 벽 외향 단위법선(접근 방향의 반대)
    p_w = np.asarray(ft.get("point", [0.0, 0.0, 0.0]), float)
    fd = float(ft.get("fd", 20.0))               # 목표 누름힘(−n 방향, N)
    k_env = float(ft.get("k_env", 4000.0))       # 벽 강성 N/m
    b_env = float(ft.get("b_env", 40.0))         # 벽 감쇠 N·s/m
    t_end = float(ft.get("t_end", 4.0))
    Kfp = float(gains.get("kfp", 0.5))
    Kfi = float(gains.get("kfi", 4.0))
    Kdamp = float(gains.get("kdamp", 25.0))
    N = robot.n
    plant = plant or robot.dyn_args
    Mp, Gp, Sp = df.stack_model(*plant)
    contact = _wall_contact(n, p_w, k_env, b_env, robot)

    def rhs(t, x):
        th, dth, Ie = x[:N], x[N:2 * N], x[2 * N]
        R = robot.fk(th)[:3, :3]
        f_meas, delta, F_env = contact(th, dth)
        fe = fd - f_meas
        Jb = robot.jacobian_body(th)
        # 명령 렌치(누름 = −n 방향), body 프레임으로 변환해 Jbᵀ 적용
        F_cmd = -n * (fd + Kfp * fe + Kfi * Ie)
        wrench_b = np.concatenate([np.zeros(3), R.T @ F_cmd])
        g = df.gravity_forces(th, robot.gravity, Mp, Gp, Sp)
        tau = Jb.T @ (wrench_b - Kdamp * (Jb @ dth)) + g
        # Ftip 규약 = 'EE 가 환경에 가하는 렌치'(MR 표준). 환경이 EE 를 +n 으로
        # 미는 반력 F_env 를 표현하려면 부호를 뒤집어 −F_env 를 넣는다.
        Ftip_b = np.concatenate([np.zeros(3), R.T @ (-F_env)])
        ddth = df.forward_dynamics(th, dth, tau, robot.gravity, Ftip_b, Mp, Gp, Sp)
        dIe = fe if delta > 0.0 else 0.0            # 적분은 접촉 중에만(자유공간 windup 방지)
        return np.concatenate([dth, ddth, [dIe]])

    x0 = np.concatenate([WP[0], np.zeros(N), [0.0]])
    t_eval = np.linspace(0, t_end, int(t_end * hz) + 1)
    # 강체에 가까운 벽 + 폐루프 → stiff. 토크제어와 동일하게 Radau.
    sol = solve_ivp(rhs, (0, t_end), x0, t_eval=t_eval, method="Radau",
                    rtol=1e-4, atol=1e-7, events=_blowup_event(N))

    TH, DTH = sol.y[:N].T, sol.y[N:2 * N].T
    fmeas, err = [], []
    for i in range(len(sol.t)):
        fm, _, _ = contact(TH[i], DTH[i])
        fmeas.append(fm)
        err.append(abs(fd - fm))                   # 추종오차 = 힘 오차(N)
    fmeas, err = np.array(fmeas), np.array(err)
    vmax = (np.max(np.abs(DTH), axis=1) if len(sol.t) else np.zeros(0))

    # 정착: '한 번 움직였다가(excited)' 속도가 잦아든 시점 = 접촉 후 평형
    diverged = sol.status != 0
    settle_time = steady_state_error = None
    if not diverged and len(sol.t):
        vw = max(1, int(VEL_WIN * hz))
        excited = False
        for i in range(len(sol.t)):
            if vmax[i] >= VEL_TOL:
                excited = True
            if excited and np.max(vmax[max(0, i - vw):i + 1]) < VEL_TOL:
                settle_time = float(sol.t[i])
                steady_state_error = float(err[i])
                break

    return {
        "t": sol.t.tolist(),
        "theta": TH.tolist(),
        "tcp": [robot.fk(TH[i])[:3, 3].tolist() for i in range(len(sol.t))],
        "error": err.tolist(),                     # 힘 오차(N)
        "torque": [],
        "qdot": [],
        "force": fmeas.tolist(),                   # 측정 법선반력(N)
        "force_des": fd,
        "waypoints_tcp": [robot.fk(w)[:3, 3].tolist() for w in WP],
        "settle_time": settle_time,
        "steady_state_error": steady_state_error,
        "diverged": bool(diverged),
    }


# ----------------------------------------------------------------------
#  토크 plant — 하이브리드 모션/힘 제어 (MR §11.6, 식 11.60/11.61)
# ----------------------------------------------------------------------
def _run_hybrid(WP, gains, hz, plant, force_task, robot):
    """하이브리드 모션/힘: 법선=힘 제어, 접선=위치 제어를 동시에 (칠판 예).

    투영행렬로 렌치공간을 두 직교 부분공간으로 분리한다 (모두 body 프레임 {b}):
        P = I − Aᵀ(AΛ⁻¹Aᵀ)⁻¹AΛ⁻¹            (식 11.60),  Λ⁻¹ = Jb·M⁻¹·Jbᵀ
        τ = Jbᵀ[ P·(Λ·a_motion) + (I−P)·F_force ] + c + g     (식 11.61)
    A = [0 0 0 | nb] (1×6): 법선 방향 운동 구속(nb·v_b=0) → P 가 그 방향을 모션에서
    제거하고 (I−P) 가 힘 제어로 넘긴다. 접선·자세는 모션 PD 가 추종.

    a_motion = Kp·Xe + Kd·Ve  (task PD, Xe=log(X⁻¹X_d) body twist 오차)
    F_force  = −nb·(fd + Kfp·fe + Kfi∫fe) − Kfd·(nb·v_b)·nb  (PI 힘 + 법선 감쇠)
    (모션 Kd 는 접선만 감쇠하므로, 컴플라이언트 벽의 법선 진동은 Kfd 로 따로 잡는다.)

    곡면 컨투어(shape: plane|cylinder|sphere): 구속 A(θ)가 위치종속이라 곡면은 접촉점
    법선 nb=Rᵀ·n(p)만 매 스텝 달라질 뿐 식은 동일. 티칭 waypoint(distinct≥2)면 표면에
    투영한 경로를 §9 quintic 으로 접선 추종, 1점이면 직선 한 획(move_len, 하위호환).
    공구 자세는 누름축(z)을 −n(p) 에 정렬(Level 2, 평면이면 자동 고정).
    force_task: shape·center·radius·axis(곡면) | normal·point(평면), fd·k_env·b_env·
    move_start·move_time.  gains: {kp,kd,kiv,kfp,kfi,kfd}.
    """
    ft = force_task or {}
    fd = float(ft.get("fd", 20.0))
    k_env = float(ft.get("k_env", 4000.0))
    b_env = float(ft.get("b_env", 40.0))
    t_hat = np.asarray(ft.get("tangent", [1.0, 0.0, 0.0]), float)
    t_hat = t_hat / np.linalg.norm(t_hat)        # 접선(단일획 fallback, base)
    move_len = float(ft.get("move_len", 0.10))   # 단일획 접선 이동거리 m
    move_start = float(ft.get("move_start", 0.8))
    move_time = float(ft.get("move_time", 1.5))  # 경로 구간당 시간
    Kp = float(gains.get("kp", 100.0))           # 모션 PID (task accel 게인)
    Kd = float(gains.get("kd", 20.0))
    Kiv = float(gains.get("kiv", 60.0))          # 모션 적분(식 11.61 의 Ki∫Xe)
    Kfp = float(gains.get("kfp", 0.5))           # 힘 PI
    Kfi = float(gains.get("kfi", 4.0))
    Kfd = float(gains.get("kfd", 10.0))          # 법선 속도 감쇠
    N = robot.n
    plant = plant or robot.dyn_args
    Mp, Gp, Sp = df.stack_model(*plant)
    surf = _make_surface(ft)                      # plane | cylinder | sphere
    contact = _surface_contact(surf, k_env, b_env, robot)

    R0 = robot.fk(WP[0])[:3, :3]                  # 시작 자세
    p0 = robot.fk(WP[0])[:3, 3]
    tool_z0 = -surf(p0)[0]                         # 시작 누름방향(= −n(p0), 공구 z 기준)
    # 모션 경로: 티칭 waypoint(>1)면 표면에 투영한 경로를 접선 추종(컨투어), 1점이면
    # 접선 직선 한 획(move_len, 하위호환). 목표를 표면에 투영해 둬야 법선 위치오차가
    # 접선으로 새지 않는다 (교재 가정 A·Vd=0).
    # _prep_waypoints 가 단일 포즈를 [w,w] 로 복제하므로 연속 중복을 제거한다.
    pts = [surf(robot.fk(w)[:3, 3])[2] for w in WP]
    path = [pts[0]]
    for q in pts[1:]:
        if np.linalg.norm(q - path[-1]) > 1e-6:
            path.append(q)
    if len(path) < 2:                            # 티칭 경로 없음 → 접선 직선 한 획
        path = [surf(p0)[2], surf(p0 + t_hat * move_len)[2]]
    n_seg = len(path) - 1
    move_total = n_seg * move_time
    t_end = max(float(ft.get("t_end", 4.0)), move_start + move_total + 1.0)

    def motion_des(t):
        """경로 목표 → (X_d 4x4, V_d body twist). 표면 재투영 + 접선 속도(자세 R0 고정)."""
        tau = t - move_start
        if tau <= 0.0:
            k, s, sd = 0, 0.0, 0.0
        elif tau >= move_total:
            k, s, sd = n_seg - 1, 1.0, 0.0
        else:
            k = int(tau // move_time)
            s, sd, _ = quintic_time_scaling(tau - k * move_time, move_time)
        p_raw = (1.0 - s) * path[k] + s * path[k + 1]
        v_raw = sd * (path[k + 1] - path[k])
        n_loc, _, p_d = surf(p_raw)               # 표면 재투영 + 국소 법선
        v_tan = v_raw - float(n_loc @ v_raw) * n_loc
        # Level 2: 공구 z축(누름방향)을 −n(p) 에 정렬 (평면이면 R_d=R0 자동, 곡면은 추종)
        R_d = _align_rot(tool_z0, -n_loc) @ R0
        X_d = np.eye(4)
        X_d[:3, :3] = R_d
        X_d[:3, 3] = p_d
        Vd = np.concatenate([np.zeros(3), R_d.T @ v_tan])
        return X_d, Vd

    def rhs(t, x):
        th, dth, Ie = x[:N], x[N:2 * N], x[2 * N]
        Iv = x[2 * N + 1:2 * N + 7]               # 모션 적분(body twist 오차 누적)
        X = robot.fk(th)
        R = X[:3, :3]
        Jb = robot.jacobian_body(th)
        Vb = Jb @ dth
        M = df.mass_matrix(th, Mp, Gp, Sp)
        Minv = np.linalg.inv(M)
        Linv = Jb @ Minv @ Jb.T                       # Λ⁻¹ = Jb M⁻¹ Jbᵀ
        nl, delta, _ = surf(X[:3, 3])                 # 접촉점 법선·침투(쿼리 1회, 투영+힘 공용)
        nb = R.T @ nl                                 # body 프레임 법선(단위)

        # 투영행렬 P (식 11.60): A=[0 0 0 | nb] 법선 운동 구속
        A = np.concatenate([np.zeros(3), nb])[None, :]   # 1×6
        s = float((A @ Linv @ A.T)[0, 0])            # 스칼라(k=1)
        P = np.eye(6) - (A.T @ A @ Linv) / s         # I − Aᵀ(AΛ⁻¹Aᵀ)⁻¹AΛ⁻¹

        # 모션 부분공간: task PD → 렌치 Λ·a_motion
        X_d, Vd = motion_des(t)
        Tdiff = np.linalg.inv(X) @ X_d
        Xe = SE3.log(Tdiff)[0]                        # body twist 오차
        Ve = SE3.Adjoint(Tdiff) @ Vd - Vb
        a_motion = Kp * Xe + Kiv * Iv + Kd * Ve
        Lam = np.linalg.inv(Linv)
        W_motion = Lam @ a_motion

        # 힘 부분공간: PI 힘 + 법선 감쇠 (누름 = −nb). surf 재사용(쿼리 절약).
        if delta > 0.0:
            ddelta = -float(nl @ (R @ Vb[3:6]))       # 침투 속도(δ̇=−n·ṗ)
            f_meas = max(0.0, k_env * delta + b_env * ddelta)
        else:
            f_meas = 0.0
        F_env = nl * f_meas
        fe = fd - f_meas
        vn = float(nb @ Vb[3:6])                      # 법선 속도(body)
        W_force = -nb * (fd + Kfp * fe + Kfi * Ie) - Kfd * vn * nb
        W_force = np.concatenate([np.zeros(3), W_force])

        c = df.coriolis_forces(th, dth, Mp, Gp, Sp)
        g = df.gravity_forces(th, robot.gravity, Mp, Gp, Sp)
        tau = Jb.T @ (P @ W_motion + (np.eye(6) - P) @ W_force) + c + g
        # 널공간 감쇠(여유 DOF): 동역학일관 투영 N=I−Jbᵀ(Λ Jb M⁻¹) 으로 task
        # 무영향, 여유관절 속도만 감쇠. 7-DOF 드리프트 방지(6-DOF 는 N≈0).
        JbarT = Lam @ Jb @ Minv                       # 동역학일관 J̄ᵀ = Λ Jb M⁻¹
        tau = tau + (np.eye(N) - Jb.T @ JbarT) @ (-KD_NULL * dth)
        Ftip_b = np.concatenate([np.zeros(3), R.T @ (-F_env)])   # MR 규약(−반력)
        ddth = df.forward_dynamics(th, dth, tau, robot.gravity, Ftip_b, Mp, Gp, Sp)
        dIe = fe if delta > 0.0 else 0.0
        return np.concatenate([dth, ddth, [dIe], Xe])     # İv = Xe(모션 적분)

    x0 = np.concatenate([WP[0], np.zeros(N), [0.0], np.zeros(6)])
    t_eval = np.linspace(0, t_end, int(t_end * hz) + 1)
    # 메시는 면 보간 법선의 미세 꺾임에 적응솔버가 스텝을 잘게 쪼개므로 허용오차 완화.
    rtol, atol = (2e-3, 1e-5) if ft.get("shape") == "mesh" else (1e-4, 1e-7)
    sol = solve_ivp(rhs, (0, t_end), x0, t_eval=t_eval, method="Radau",
                    rtol=rtol, atol=atol, events=_blowup_event(N))

    TH, DTH = sol.y[:N].T, sol.y[N:2 * N].T
    fmeas, ferr, tan_pos, tan_des, tcp_des = [], [], [], [], []
    for i in range(len(sol.t)):
        fm, _, _ = contact(TH[i], DTH[i])
        fmeas.append(fm)
        ferr.append(abs(fd - fm))
        p = robot.fk(TH[i])[:3, 3]
        tan_pos.append(float(t_hat @ (p - p0)))      # 실제 접선 변위
        X_d, _ = motion_des(sol.t[i])
        tan_des.append(float(t_hat @ (X_d[:3, 3] - p0)))   # 목표 접선 변위
        tcp_des.append(X_d[:3, 3].tolist())          # 목표 경로 위치(컨투어 추종 검증)
    fmeas = np.array(fmeas)
    diverged = sol.status != 0

    return {
        "t": sol.t.tolist(),
        "theta": TH.tolist(),
        "tcp": [robot.fk(TH[i])[:3, 3].tolist() for i in range(len(sol.t))],
        "error": ferr,                             # 힘 오차(N)
        "torque": [],
        "qdot": [],
        "force": fmeas.tolist(),                   # 측정 법선반력(N)
        "force_des": fd,
        "tan_pos": tan_pos,                        # 실제 접선 변위(m)
        "tan_des": tan_des,                        # 목표 접선 변위(m)
        "tcp_des": tcp_des,                        # 목표 경로 위치(컨투어, base)
        "waypoints_tcp": [robot.fk(w)[:3, 3].tolist() for w in WP],
        "settle_time": None,
        "steady_state_error": None,
        "diverged": bool(diverged),
    }
