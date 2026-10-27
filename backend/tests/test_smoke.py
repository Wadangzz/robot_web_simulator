"""엔진이 패키지(`motion-core`)로 빠진 뒤에도 앱이 도는지 박제한다.

기존 parity 테스트는 수식 소유권과 함께 motion-core 로 이식됐다. 여기 남는 것은
**소비자 쪽 계약**이다 — 레지스트리·라우터·시뮬 루프가 외부 엔진으로 동작하는지.
"""
import numpy as np
from fastapi.testclient import TestClient
from motion_core.robots import robot_registry as reg

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
