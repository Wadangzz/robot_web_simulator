"""실시간 인터랙티브 제어 세션 (grab & push).

기존 sim.py 는 '한 번에 precompute → θ(t) 재생' 인 반면, 여기서는 매 틱마다 외부에서
주는 잡기힘(grab force, EE 에 가하는 base 프레임 외력)에 로봇이 실시간 반응한다.
plant 는 동일한 numba RNE(forward_dynamics) — fd ~0.3ms 라 100Hz 루프가 가능하다.

토크 plant 제어기(PD/PID/CT/임피던스)를 setpoint 유지 모드로 돌린다:
  - 위치제어(PD/PID/CT): 외력에 버틴다(뻣뻣).  임피던스: 양보한다(물렁).
  → 같은 외력에 '싸우는 제어기 vs 양보하는 제어기' 대비가 핵심 데모.

발산 처리 두 모드:
  - 'honest' : 추가 안전층 없음. 폭주하면 감지·정지(실로봇 보호정지) + diverged.
  - 'safe'   : 진짜 안전층(토크 포화·속도/관절 리밋)으로 봉쇄. 폭발 대신 한계서 멈춤.
               개입하면 flags 로 알린다(조용히 '정상'으로 보이지 않게).
  ※ 'safe' 는 '마법으로 성공'이 아니라 '안전하게 실패' — 나쁜 게인은 여전히 작업 실패.
"""
import numpy as np
from motion_core.robots import dynamics_fast as df
from motion_core.robots.robot_math import SE3

from app.sim import OMEGA_MAX, THETA_MAX

# 물리 적분 dt 상한(s). 감쇠항(Kd·θ̇)을 암시적으로 풀어(implicit) explicit Euler 의
# stiff 발산을 없앴다 → 1kHz 에서도 기본게인 안정 + 실시간(~0.4x). 스트리밍 dt 를 서브스텝.
# (그래도 게인을 크게 올리면 강성항(explicit)이 발산 = 이산 불안정 시연 — honest 가 잡음)
PHYS_DT = 0.001            # 1kHz 물리 (암시적 감쇠로 안정)

# safe 모드 기본 안전 한계 (UR5 기준 보수값)
TAU_MAX_SAFE = 150.0        # 관절 토크 포화 N·m
VEL_MAX_SAFE = 5.0          # 관절 속도 클램프 rad/s


class InteractiveSession:
    """한 번에 한 로봇의 실시간 grab & push 세션. step() 을 루프에서 호출한다."""

    def __init__(self, robot, controller="impedance", gains=None,
                 mode="honest", setpoint=None):
        self.robot = robot
        self.controller = controller
        self.is_adm = controller == "admittance"     # 모션 plant(별도 경로)
        g = gains or {}
        self.Kp = float(g.get("kp", 600.0))
        self.Kd = float(g.get("kd", 60.0))
        self.Ki = float(g.get("ki", 0.0))
        # 어드미턴스(가상 M-B-K) — 힘 측정→움직임 출력. 임피던스의 쌍.
        self.M_v = max(float(g.get("m", 2.0)), 0.1)
        self.B_v = float(g.get("b", 40.0))
        self.K_v = float(g.get("k", 600.0))
        self.KP_TRACK = 20.0                         # 내부 resolved-rate 추종 게인
        self.dp = np.zeros(3)                        # 가상 변위(base 3D)
        self.dv = np.zeros(3)                        # 가상 속도
        self.mode = mode
        sp = robot.ready if setpoint is None else np.asarray(setpoint, float)
        self.th_d = sp.copy()                    # 유지 setpoint
        self.T_d = robot.fk(sp)                  # setpoint pose (임피던스용, 상수 → 선계산)
        self.th = sp.copy()
        self.dth = np.zeros(robot.n)
        self.I = np.zeros(robot.n)               # 적분(PID/CT)
        self.dead = False                        # 발산 후 정지(보호정지)
        self.pm = df.stack_model(*robot.dyn_args)   # plant 모델(스택)
        self.cm = self.pm                        # 컨트롤러 모델(=plant, 불확실성 없음)
        if mode == "safe":
            self.tau_max = TAU_MAX_SAFE
            self.vel_max = VEL_MAX_SAFE
            if robot.joint_limits is not None:
                lim = np.asarray(robot.joint_limits, float)
                self.lo, self.hi = lim[:, 0].copy(), lim[:, 1].copy()
            else:                                # 상수빌드 UR5 등 한계 미상 → 넉넉히
                self.lo = np.full(robot.n, -2 * np.pi)
                self.hi = np.full(robot.n, 2 * np.pi)
        else:
            self.tau_max = self.vel_max = self.lo = self.hi = None

    def _g(self, th):
        return df.gravity_forces(th, self.robot.gravity, *self.cm)

    def _control_split(self, th, dth, M):
        """제어토크를 비감쇠부 τ_other 와 관절공간 감쇠행렬 D_eff 로 분리한다.
        실제 명령토크 = τ_other − D_eff·θ̇ (기존 _run_torque 법칙과 동일).
        감쇠항이 explicit Euler 에서 stiff → 암시적으로 풀려고 떼어낸다.
            pd/pid : D_eff = Kd·I          impedance : D_eff = Kd·Jbᵀ·Jb
            ct     : D_eff = Kd·M (CT 감쇠 = M·Kd·θ̇)
        """
        Kp, Kd, Ki = self.Kp, self.Kd, self.Ki
        e = self.th_d - th
        g = self._g(th)
        n = self.robot.n
        c = self.controller
        if c == "impedance":
            Verr = SE3.log(np.linalg.inv(self.robot.fk(th)) @ self.T_d)[0]
            Jb = self.robot.jacobian_body(th)
            return Jb.T @ (Kp * Verr) + g, Kd * (Jb.T @ Jb)
        if c == "computed_torque":
            cor = df.coriolis_forces(th, dth, *self.cm)
            return M @ (Kp * e + Ki * self.I) + cor + g, Kd * M
        if c == "pid":
            return Kp * e + Ki * self.I + g, Kd * np.eye(n)
        return Kp * e + g, Kd * np.eye(n)        # pd

    def _result(self, flags):
        ee = self.robot.fk(self.th)
        return {
            "theta": self.th.tolist(),
            "ee_pos": ee[:3, 3].tolist(),
            "flags": flags,
            "diverged": self.dead,
        }

    def step(self, grab_force=None, dt=0.01, substeps=None):
        """1틱 전진(스트리밍 dt). 물리를 ≥1kHz(PHYS_DT)로 서브스텝, 감쇠는 암시적이라 안정.
        grab_force = EE 에 가하는 base 프레임 외력(3-vec, N).
        → {theta, ee_pos, flags, diverged}."""
        if self.dead:                            # 보호정지 후엔 동결
            return self._result(["protective_stop"])
        if substeps is None:
            substeps = max(1, int(np.ceil(dt / PHYS_DT)))   # 물리 ≥ 1kHz
        h = dt / substeps
        flags = set()
        for _ in range(substeps):
            self._substep(grab_force, h, flags)
            if self.dead:
                break
        return self._result(sorted(flags))

    def _substep(self, grab_force, dt, flags):
        """물리 한 스텝 — 토크 plant(임피던스/PD/PID/CT) vs 모션 plant(어드미턴스) 분기."""
        if self.is_adm:
            self._substep_admittance(grab_force, dt, flags)
        else:
            self._substep_torque(grab_force, dt, flags)

    def _substep_admittance(self, grab_force, dt, flags):
        """어드미턴스: 잡기힘 f_ext 로 가상 M-B-K 적분(힘→움직임) → 변위 dp 를
        resolved-rate 로 추종. 임피던스(움직임→힘)의 쌍. 기구학 plant라 명시적로 충분."""
        th = self.th
        f_ext = (np.zeros(3) if grab_force is None
                 else np.asarray(grab_force, float))       # 외력 그대로(부호변환 없음)
        da = (f_ext - self.B_v * self.dv - self.K_v * self.dp) / self.M_v
        self.dv = self.dv + da * dt
        self.dp = self.dp + self.dv * dt
        T_d = self.T_d.copy()
        T_d[:3, 3] = self.T_d[:3, 3] + self.dp           # 목표 = setpoint + 가상 변위
        Xe = SE3.log(np.linalg.inv(self.robot.fk(th)) @ T_d)[0]
        qd = self.robot.dls_inv(self.robot.jacobian_body(th)) @ (self.KP_TRACK * Xe)
        self._finish(th, qd, dt, flags)                  # 리밋 클램프·발산판정 공용

    def _substep_torque(self, grab_force, dt, flags):
        """토크 plant 한 스텝. 감쇠 암시적: (M+h·D_eff)·θ̇⁺ = M·θ̇ + h·(τ_other − bias).
        토크 포화 시엔 클립 토크로 명시적 적분(포화가 봉쇄 역할). flags 누적(set)."""
        th, dth = self.th, self.dth
        n = self.robot.n
        # 잡기힘 = 로봇에 가하는 외력 → Ftip=−Rᵀ·grab (외란 부호 규약과 동일)
        grab = np.zeros(3) if grab_force is None else np.asarray(grab_force, float)
        if np.any(grab):
            f_ee = -self.robot.fk(th)[:3, :3].T @ grab
            Ftip = np.concatenate([np.zeros(3), f_ee])
        else:
            Ftip = np.zeros(6)

        M = df.mass_matrix(th, *self.pm)
        bias = df.inverse_dynamics(th, dth, np.zeros(n),       # c + g + Jᵀ·Ftip
                                   self.robot.gravity, Ftip, *self.pm)
        tau_other, D_eff = self._control_split(th, dth, M)

        if self.tau_max is not None:             # safe: 토크 포화 → 클립 후 명시적
            tau_full = tau_other - D_eff @ dth
            clip = np.clip(tau_full, -self.tau_max, self.tau_max)
            if not np.array_equal(clip, tau_full):
                flags.add("saturation")
                ddth = np.linalg.solve(M, clip - bias)        # M·θ̈ = τ − bias
                self._finish(th, dth + ddth * dt, dt, flags)
                return
        # 비포화: 감쇠항만 암시적으로 풀어 stiff 발산 제거
        dth_new = np.linalg.solve(M + dt * D_eff, M @ dth + dt * (tau_other - bias))
        self._finish(th, dth_new, dt, flags)

    def _finish(self, th_old, dth, dt, flags):
        """θ⁺=θ+h·θ̇⁺(semi-implicit), PID 적분, safe 리밋 클램프, 발산판정·상태저장."""
        th = th_old + dt * dth
        if self.controller in ("pid", "computed_torque"):
            self.I = self.I + (self.th_d - th) * dt
        if self.mode == "safe":                  # 속도·관절 리밋 클램프
            vclip = np.clip(dth, -self.vel_max, self.vel_max)
            if not np.array_equal(vclip, dth):
                flags.add("vel_limit")
            dth = vclip
            tclip = np.clip(th, self.lo, self.hi)
            if not np.array_equal(tclip, th):
                flags.add("joint_limit")
                dth = np.where(tclip != th, 0.0, dth)    # 한계서 정지
            th = tclip
        self.th, self.dth = th, dth
        # 발산 안전망(양 모드): 상태 폭주 → 보호정지
        if (not np.all(np.isfinite(th))
                or np.max(np.abs(th)) > THETA_MAX
                or np.max(np.abs(dth)) > OMEGA_MAX):
            self.dead = True
            flags.add("protective_stop")
