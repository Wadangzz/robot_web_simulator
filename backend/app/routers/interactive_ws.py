"""실시간 grab & push WebSocket 엔드포인트 (Phase 2).

InteractiveSession 을 들고 ~60Hz 루프로 step → θ 스트리밍한다. precompute→replay REST
와 별개의 실시간 경로 — 프레임당 백엔드지만 REST 폴링이 아니라 WS 스트리밍이라 원칙 OK.
물리는 step 내부에서 1kHz 로 서브스텝(암시적 감쇠)되므로 틱은 렌더 주사율급이면 된다.

프로토콜:
  client→server: {type:'config', robot_id, controller, gains, mode, setpoint?}  세션 (재)설정
                 {type:'grab',   force:[fx,fy,fz]}                              현재 잡기힘(base, N)
                 {type:'reset'}                                                 setpoint 로 리셋
  server→client(매 틱): {theta:[...], ee_pos:[...], flags:[...], diverged:bool}
"""
import asyncio

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from motion_core.robots import robot_registry as registry

from app.interactive import InteractiveSession

router = APIRouter()

TICK_HZ = 60.0          # 스트리밍 틱(렌더 주사율급). 물리는 step 내부서 1kHz 서브스텝.


def _make_session(cfg):
    robot = registry.get_robot(cfg.get("robot_id", "ur5"))
    return InteractiveSession(robot, cfg.get("controller", "impedance"),
                              cfg.get("gains"), cfg.get("mode", "honest"),
                              cfg.get("setpoint"))


@router.websocket("/ws/interactive")
async def interactive_ws(ws: WebSocket):
    await ws.accept()
    st = {"session": None, "cfg": None, "grab": None, "stop": False}

    async def receiver():
        """클라 메시지 비동기 수신 → 공유 상태 갱신(루프와 분리)."""
        try:
            while True:
                msg = await ws.receive_json()
                t = msg.get("type")
                if t in ("config", "reset"):
                    cfg = msg if t == "config" else st["cfg"]
                    if cfg is None:
                        continue
                    try:
                        st["session"] = _make_session(cfg)
                        st["cfg"], st["grab"] = cfg, None
                    except Exception as e:                # 미등록 robot 등
                        await ws.send_json({"error": f"config 실패: {e}"})
                elif t == "grab":
                    st["grab"] = msg.get("force")
        except WebSocketDisconnect:
            st["stop"] = True

    recv = asyncio.create_task(receiver())
    loop = asyncio.get_running_loop()
    dt = 1.0 / TICK_HZ
    try:
        while not st["stop"]:
            t0 = loop.time()
            s = st["session"]
            if s is not None:
                await ws.send_json(s.step(st["grab"], dt))   # 물리 1kHz 서브스텝
            # 실시간 유지: 계산시간 빼고 남은 만큼만 sleep
            await asyncio.sleep(max(0.0, dt - (loop.time() - t0)))
    except WebSocketDisconnect:
        pass
    finally:
        recv.cancel()
