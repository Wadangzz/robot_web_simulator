"""기구학 비즈니스 로직 (IK / FK). 로봇은 레지스트리에서 robot_id 로 해석."""
import numpy as np
from fastapi import HTTPException
from motion_core.robots import robot_registry as registry
from motion_core.robots.robot_math import SE3

from app.schemas import FKRequest, FKResponse, IKRequest, IKResponse


def _robot(robot_id):
    try:
        return registry.get_robot(robot_id)
    except KeyError:
        raise HTTPException(404, f"등록되지 않은 로봇: {robot_id}") from None


def solve_ik(req: IKRequest) -> IKResponse:
    robot = _robot(req.robot_id)
    T = SE3.from_pose(req.pose)
    seed = np.array(req.seed) if req.seed else np.zeros(robot.n)
    theta, ok = robot.ik(T, seed)
    return IKResponse(theta=theta.tolist(), converged=bool(ok),
                      tcp=robot.fk(theta)[:3, 3].tolist())


def solve_fk(req: FKRequest) -> FKResponse:
    robot = _robot(req.robot_id)
    T = robot.fk(np.array(req.theta))
    pos = T[:3, 3].tolist()
    quat, _ = SE3.orientation(T)            # [x,y,z,w]
    m = robot.manipulability(req.theta)     # 특이점 근접도
    return FKResponse(pose=pos + quat, tcp=pos,
                      w=m["w"], sigma_min=m["sigma_min"])
