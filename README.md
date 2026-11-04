# Robot Web Simulator

스크류 이론(Modern Robotics) 기반의 **로봇 기구학·동역학·제어 웹 시뮬레이터**.
직접 구현한 `robot_math` 라이브러리(FastAPI 백엔드)로 무거운 시뮬레이션을 한 번에
계산하고, React + react-three-fiber 프런트가 실제 URDF 메시를 렌더하며 결과 θ(t)를
브라우저에서 재생한다.

**멀티로봇 지원** — UR5(6축) · Franka Panda · KUKA iiwa14(각 7축) · OpenManipulator-X(4축).
같은 엔진이 축 수와 무관하게 동작한다.

> 핵심 원칙: **무거운 계산(시뮬)은 백엔드에서 한 번, 브라우저는 그 결과만 재생.**
> 관절 슬라이더 조그는 브라우저 FK(urdf-loader)로 처리해 프레임당 통신이 0이다.

---

## ✨ 기능

**기구학**
- 정기구학(FK) — PoE 지수곱(`T = e^{[S₁]θ₁}···M`), URDF에서 스크류축 직접 추출
- 역기구학(IK) — 수치 IK(Newton-Raphson) + **DLS**(특이점 감쇠), manipulability(`w=√det(JJᵀ)`, σ_min) 라이브 표시
- 직교좌표 ↔ 관절각 양방향 라이브 변환(PoseEditor)

**동역학**
- RNE 역/순동역학 — `inverse_dynamics` · `mass_matrix` · `gravity_forces` · `coriolis_forces` · `forward_dynamics`
- **Numba JIT 가속**(`forward_dynamics` ~3.5ms→~254µs ≈14×, 연속 시뮬 ~7× 단축), numpy 구현과 비트일치(pytest parity)

**제어기 (한 시뮬 엔진, 로봇 무관)**
- 토크: **PD+중력보상 · PID · Computed Torque · 임피던스**
- 모션: **작업공간 속도제어(resolved-rate) · 어드미턴스**
- 접촉: **힘 제어 · 하이브리드 모션/힘**(법선=힘, 접선=위치 투영 분리)
- 자동튜닝(극배치), ζ·ωₙ 배지, 적분 안정여유(Routh)

**궤적 · 환경**
- 관절/작업공간(직교 직선) 궤적, 5차 다항식 타임스케일링
- 현실 모델 — 관절 마찰·토크 포화, 이산 ZOH 제어율, 센서 노이즈
- **하이브리드 컨투어** — 곡면(평면/원기둥/구) 또는 업로드 메시(STL/OBJ)에 마우스로 경로를 그려 일정 힘으로 추종

**그 외**
- 단일 포트 서빙 + 선택적 로그인 인증(`SIM_KEY`)

---

## 🧱 기술 스택

| | |
|---|---|
| 백엔드 | Python 3.12+ · FastAPI · SQLModel(SQLite) · NumPy/SciPy · Numba · trimesh · yourdfpy · [`uv`](https://docs.astral.sh/uv/) |
| 프런트 | React 19 · react-three-fiber v9 · drei v10 · urdf-loader · Vite · TailwindCSS · shadcn-style UI · pnpm |

---

## 🚀 빠른 시작

두 서버를 같이 띄운다(프런트는 자체 데이터가 없고 `/api`·`/meshes`를 백엔드로 프록시).

```bash
# 1) 백엔드  (http://localhost:8500/docs = Swagger)
cd backend
uv sync            # 최초 1회
uv run main.py

# 2) 프런트  (Vite 개발 서버)
cd frontend
pnpm install       # 최초 1회
pnpm dev
```

### 단일 포트 배포 (프록시 불필요)

```bash
cd frontend && pnpm build      # → frontend/dist
cd backend  && uv run main.py  # dist 를 같은 출처로 서빙 (프런트+API 한 포트)
```

- 접근 제어: `backend/.env`의 `SIM_KEY` 설정 시 시뮬 실행·쓰기에 로그인 필요(읽기는 공개). 빈 값이면 인증 OFF.
- LAN 공유: `HOST`/`PORT` 환경변수(기본 `0.0.0.0:8500`).

### 라우트

| 경로 | 내용 |
|---|---|
| `/` | `/simulation` 으로 리다이렉트 |
| `/simulation` | 시뮬레이터 (인증 보호 대상) |
| `/docs` | FastAPI Swagger |

---

## 🗂 구조

```text
backend/
├─ app/
│  ├─ core/        paths  (수학·모델 엔진은 motion-core 패키지로 분리)
│  ├─ routers/     robot · robots · kinematics · programs · simulate · mesh
│  ├─ controllers/ kinematics · programs · simulate
│  ├─ sim.py       시뮬레이션 엔진(solve_ivp, 제어 루프)
│  ├─ realism.py   plant≠controller 모델 불확실성 · mesh_store.py(업로드 메시)
│  └─ models.py · schemas.py · db.py · meshes_setup.py · main.py
├─ tests/          RNE numpy↔numba parity (pytest)
└─ pyproject.toml · uv.lock

frontend/app/
├─ components/  RobotView · ControlPanel · PoseEditor · ProgramList · Plots · WallPlane …
├─ api.ts · store.ts · root.tsx · main.web.tsx
```

주요 엔드포인트: `GET /api/robots` · `POST /api/ik` · `POST /api/fk` · `GET/POST/DELETE /api/programs/...` · `POST /api/run`(시뮬) · `POST /api/mesh`(메시 업로드).

백엔드 설계·아키텍처는 [`backend/WEB_DESIGN.md`](backend/WEB_DESIGN.md) 참고.

---

## 📝 메모

- UR5의 **영점(전관절 0) 자세는 특이점**이라 시작 시 σ_min=0 경고는 정상. 운용 시작은 비특이 `READY` 자세.
- `Pose` 등 DB 스키마 변경 시 `backend/robot.db` 삭제 후 재시작(`create_all`은 ALTER 안 함).
- 런타임 산출물(`robot.db` · `app/meshes/` · `outputs/` · `node_modules` · `dist`)은 gitignore.

---

## 🔗 License

MIT License.
