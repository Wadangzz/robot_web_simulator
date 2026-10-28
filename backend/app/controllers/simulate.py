"""시뮬레이션 실행 비즈니스 로직 — 로봇 해석 + 프로그램 해석(IK) + 엔진 호출."""
import numpy as np
import trimesh
from fastapi import HTTPException
from motion_core.robots import robot_registry as registry
from motion_core.robots.robot_math import SE3
from sqlmodel import Session, select

from app import mesh_store, realism, sim
from app.models import Pose
from app.schemas import ForceTask, RunRequest


def run(req: RunRequest, session: Session):
    try:
        robot = registry.get_robot(req.robot_id)
    except KeyError:
        raise HTTPException(404, f"등록되지 않은 로봇: {req.robot_id}") from None

    if req.waypoints is not None:
        waypoints = req.waypoints
    elif req.program_id is not None:
        poses = session.exec(
            select(Pose).where(Pose.program_id == req.program_id).order_by(Pose.id)).all()
        if not poses:
            raise HTTPException(404, f"프로그램 {req.program_id} 에 저장된 포즈가 없습니다")
        # 현재 자세에서 출발(실제 로봇처럼 "지금 위치에서 프로그램 실행").
        # start 미지정 시 ready 로 폴백. 티치 관절각(theta)이 있으면 그대로, 없으면 IK.
        waypoints = [list(req.start) if req.start else robot.ready.tolist()]
        for p in poses:
            if p.theta:
                waypoints.append(list(p.theta))
            else:
                T = SE3.from_pose([p.x, p.y, p.z, p.qx, p.qy, p.qz, p.qw])
                theta, _ = robot.ik(T, np.array(waypoints[-1]))
                waypoints.append(theta.tolist())
    else:
        raise HTTPException(400, "program_id 또는 waypoints 중 하나는 필요합니다")

    # 힘/하이브리드 제어: 표면을 시작자세 EE 도구축(보는 방향)에 배치한다.
    # push = R0·tool_axis (EE 가 누르는 방향). 평면=법선 −push·point gap 앞,
    # 곡면=중심을 push 로 (gap+radius) 앞에 둬 시작 EE 가 표면 밖 gap 에 선다.
    force_task = None
    wall = None
    if req.controller in ("force", "hybrid"):
        ft = (req.force_task or ForceTask()).model_dump()
        T0 = robot.fk(np.array(waypoints[0]))
        R0, p_start = T0[:3, :3], T0[:3, 3]
        push = R0 @ np.array(ft["tool_axis"], float)
        push = push / np.linalg.norm(push)
        ft["tangent"] = (R0 @ np.array(ft["tan_axis"], float)).tolist()
        shape = ft.get("shape", "plane")
        if shape == "mesh":
            # 업로드 메시를 배치 transform(scale→rotate→translate) 적용해 표면모델로.
            try:
                base = mesh_store.get(ft["mesh_id"]).copy()
            except KeyError:
                raise HTTPException(404, "메시를 먼저 업로드하세요") from None
            base.apply_scale(float(ft.get("mesh_scale", 1.0)))
            base.apply_transform(SE3.from_pose(list(ft["mesh_pos"])
                                               + list(ft["mesh_quat"])))
            ft["_mesh"] = base
            ft["_pq"] = trimesh.proximity.ProximityQuery(base)
            wall = {"shape": "mesh"}              # 프론트는 자기 로컬 파일로 렌더
        elif shape in ("cylinder", "sphere"):
            radius = float(ft["radius"])
            center = p_start + push * (ft["gap"] + radius)       # 중심 = 표면 뒤
            ft["center"] = center.tolist()
            axis = ft["tangent"]                                 # 원기둥 축 = 접선(EE x)
            if shape == "cylinder":
                ft["axis"] = axis
            wall = {"shape": shape, "center": center.tolist(),
                    "axis": axis, "radius": radius}
        else:                                                    # plane (현행)
            ft["normal"] = (-push).tolist()                      # 벽 외향(로봇 쪽)
            ft["point"] = (p_start + push * ft["gap"]).tolist()
            wall = {"shape": "plane", "point": ft["point"],
                    "normal": ft["normal"]}
        force_task = ft

    # 사용자가 3D 표면에 그린 경로(force_path, base xyz) → 표면 위 자세정렬 IK 로
    # 하이브리드 컨투어 waypoints 생성(start 에서 출발). 점을 표면에 투영하고 공구
    # 누름축을 그 점 법선에 맞춰 IK(sim 의 표면모델·정렬 재사용).
    if req.controller == "hybrid" and req.force_path and force_task is not None:
        surf = sim._make_surface(force_task)
        T0 = robot.fk(np.array(waypoints[0]))
        R0, p0 = T0[:3, :3], T0[:3, 3]
        tool_z0 = -surf(p0)[0]                        # 시작 누름방향
        wp, seed = [np.array(waypoints[0], float)], np.array(waypoints[0], float)
        for p in req.force_path:
            n, _, p_proj = surf(np.array(p, float))
            R_t = sim._align_rot(tool_z0, -n) @ R0
            T = np.eye(4)
            T[:3, :3], T[:3, 3] = R_t, p_proj
            th, _ = robot.ik(T, seed)
            wp.append(th)
            seed = th
        waypoints = [w.tolist() for w in wp]

    plant, ctrl = realism.build_models(robot.dyn_args, req.payload, req.model_scale)
    result = sim.run_simulation(
        waypoints, controller=req.controller, gains=req.gains,
        gravity_comp=req.gravity_comp, t_seg=req.t_seg, hold=req.hold,
        plant=plant, ctrl=ctrl, disturbance=req.disturbance,
        traj_mode=req.traj_mode, friction=req.friction, tau_max=req.tau_max,
        traj_tau_max=req.traj_tau_max,
        traj_dq_max=req.traj_dq_max,
        control_rate=req.control_rate, noise=req.noise,
        noise_seed=req.noise_seed, force_task=force_task, robot=robot)
    if force_task is not None:                    # 3D 표면 렌더용 기하
        result["wall"] = wall

    # 재생 중 직교좌표·manipulability 를 라이브로 보이게: θ(t) 에서 EE pose·σ_min
    # 시계열을 한 번에 계산해 응답에 실어 보낸다 (프레임당 REST 왕복 없이 인덱싱).
    ee_pose, sigma_min = [], []
    for th in result["theta"]:
        T = robot.fk(np.array(th))
        quat, _ = SE3.orientation(T)                 # [x,y,z,w]
        ee_pose.append(T[:3, 3].tolist() + quat)
        sigma_min.append(robot.manipulability(th)["sigma_min"])
    result["ee_pose"] = ee_pose
    result["sigma_min"] = sigma_min
    return result
