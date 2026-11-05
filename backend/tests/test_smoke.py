"""엔진이 패키지(`motion-core`)로 빠진 뒤에도 앱이 도는지 박제한다.

기존 parity 테스트는 수식 소유권과 함께 motion-core 로 이식됐다. 여기 남는 것은
**소비자 쪽 계약**이다 — 레지스트리·라우터·시뮬 루프가 외부 엔진으로 동작하는지.
"""
import numpy as np
from fastapi.testclient import TestClient
from motion_core.robots import robot_registry as reg
from motion_core.verify.closed_loop import saturated_samples

from app.main import app
from app.sim import run_simulation

client = TestClient(app)


def test_robot_list_endpoint():
    res = client.get('/api/robots')
    assert res.status_code == 200
    assert 'ur5' in [r['id'] for r in res.json()]


def test_fk_endpoint_matches_engine():
    """라우터 경로와 엔진 직접 호출이 같은 말단 위치를 준다."""
    robot = reg.get_robot('ur5')
    theta = robot.ready
    res = client.post('/api/fk', json={'robot': 'ur5',
                                       'theta': theta.tolist()})
    assert res.status_code == 200
    assert np.allclose(res.json()['pose'][:3], robot.fk(theta)[:3, 3])


def test_simulation_closes_the_loop():
    """ready → +0.2rad 1구간. 발산 없이 목표에 수렴하고 토크가 유한하다."""
    robot = reg.get_robot('ur5')
    target = robot.ready + 0.2
    out = run_simulation([robot.ready.tolist(), target.tolist()],
                         controller='computed_torque', hz=20,
                         t_seg=0.5, hold=0.2)
    assert not out['diverged']
    assert np.all(np.isfinite(out['torque']))
    assert np.allclose(out['theta'][-1], target, atol=1e-2)


def test_timeopt_trajectory_closes_the_loop():
    """joint_timeopt: T 를 토크 한계에서 역산한 궤적도 폐루프로 수렴한다."""
    robot = reg.get_robot('ur5')
    target = robot.ready + 0.2
    out = run_simulation([robot.ready.tolist(), target.tolist()],
                         controller='computed_torque', hz=20, hold=0.2,
                         traj_mode='joint_timeopt',
                         traj_tau_max=[150, 150, 150, 28, 28, 28],
                         traj_dq_max=np.pi)
    assert out['traj_error'] is None
    assert not out['diverged']
    assert np.allclose(out['theta'][-1], target, atol=1e-2)


def test_timeopt_without_limits_reports_error():
    """한계 미지정이면 숫자를 지어내지 않고 traj_error 로 표면화한다."""
    robot = reg.get_robot('ur5')
    out = run_simulation([robot.ready.tolist(), (robot.ready + 0.2).tolist()],
                         traj_mode='joint_timeopt')
    assert out['traj_error'] and 'traj_tau_max' in out['traj_error']
    assert out['t'] == []


def test_commanded_torque_makes_saturation_countable():
    """포화 판정(V3)은 명령 토크가 있어야 센다.

    클립된 계열로는 '한계에 닿은 샘플'과 '잘린 샘플'이 같은 수가 되어, 실제로
    포화한 런이 포화 0건으로 보인다. 그 구멍을 막은 것이 `torque_cmd` 다.
    """
    robot = reg.get_robot('ur5')
    target = robot.ready + 0.6
    tau_max = 20.0
    out = run_simulation([robot.ready.tolist(), target.tolist()],
                         controller='computed_torque', hz=20,
                         t_seg=0.4, hold=0.2, tau_max=tau_max)
    cmd = np.array(out['torque_cmd'])
    applied = np.array(out['torque'])
    assert cmd.shape == applied.shape
    # 액추에이터가 낸 토크는 한계 안이고, 명령은 한계를 넘었다.
    assert np.all(np.abs(applied) <= tau_max + 1e-9)
    assert saturated_samples(cmd, tau_max).sum() > 0
    # 같은 질문을 클립된 계열에 하면 0건으로 보인다 — 이 테스트의 요점.
    assert saturated_samples(applied, tau_max).sum() == 0


def test_no_commanded_torque_series_without_a_limit():
    """한계가 없으면 명령 = 실제라 payload 를 두 번 싣지 않는다."""
    robot = reg.get_robot('ur5')
    out = run_simulation([robot.ready.tolist(), (robot.ready + 0.2).tolist()],
                         controller='computed_torque', hz=20,
                         t_seg=0.5, hold=0.2)
    assert out['torque_cmd'] == []
