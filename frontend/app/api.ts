// FastAPI 백엔드 클라이언트 (axios). vite 프록시로 /api → :8500
import axios from 'axios';

const http = axios.create({ baseURL: '/api' });

// 시뮬 인증 키(로그인 폼이 받음). sessionStorage 에 저장 → 새로고침은 유지하되
// 탭/브라우저 닫으면 사라짐(다음에 다시 로그인). 모든 /api 요청에 헤더로 첨부.
// 백엔드 SIM_KEY 미설정이면 인증 OFF(키 없이도 통과) — dev 는 로그인 화면 안 뜸.
const SIM_KEY_STORAGE = 'sim_key';
export const getSimKey = () => sessionStorage.getItem(SIM_KEY_STORAGE) ?? '';
export const setSimKey = (k: string) =>
  sessionStorage.setItem(SIM_KEY_STORAGE, k);
http.interceptors.request.use((config) => {
  const k = getSimKey();
  if (k) config.headers['X-Sim-Key'] = k;
  return config;
});
// 현재 저장된 키로 시뮬 권한이 있나(또는 인증 비활성). 200=ok → 시뮬, 401=로그인 필요.
export const checkAuth = () =>
  http
    .get('/auth')
    .then(() => true)
    .catch(() => false);
// 로그인 폼 제출: 키 검증 후 맞으면 저장. 반환 true=성공/false=틀림.
export const login = async (key: string) => {
  try {
    await http.post('/login', { key });
    setSimKey(key);
    return true;
  } catch {
    return false;
  }
};

// 멀티로봇: 목록(로드 없이 메타) + 상세(렌더용 urdf_url·packages 포함)
export interface RobotSummary {
  id: string;
  name: string;
  dof: number;
}

export interface RobotDetail {
  id: string;
  name: string;
  n: number;
  joint_names: string[];
  joint_limits: number[][] | null;
  ready: number[]; // 비특이 운용 출발 자세
  home_tcp: number[];
  ready_tcp: number[];
  urdf_url: string;
  packages: Record<string, string>;
}

export interface ControllerSpec {
  name: string;
  label: string;
  params: {
    key: string;
    default: number;
    min: number;
    max: number;
    step?: number; // 슬라이더 스텝(기본 1). 힘 게인처럼 소수면 0.1 등
    label?: string;
  }[];
}

// 제어기 스펙(정적 UI 메타데이터). 백엔드는 controller 문자열+gains 만 받으면 되므로
// 굳이 API 로 받지 않고 프론트가 보유. 게인 키(kp/ki/kd)는 백엔드 torque 와의 계약.
export const CONTROLLERS: ControllerSpec[] = [
  {
    name: 'pd',
    label: 'PD + 중력보상',
    params: [
      { key: 'kp', default: 100, min: 0, max: 400 },
      { key: 'kd', default: 20, min: 0, max: 100 },
    ],
  },
  {
    name: 'pid',
    label: 'PID',
    params: [
      { key: 'kp', default: 120, min: 0, max: 400 },
      { key: 'ki', default: 60, min: 0, max: 300 },
      { key: 'kd', default: 35, min: 0, max: 100 },
    ],
  },
  {
    name: 'computed_torque',
    label: 'Computed Torque',
    params: [
      { key: 'kp', default: 100, min: 0, max: 400 },
      { key: 'ki', default: 0, min: 0, max: 200 },
      { key: 'kd', default: 20, min: 0, max: 100 },
    ],
  },
  {
    name: 'impedance',
    label: '임피던스',
    params: [
      { key: 'kp', default: 600, min: 0, max: 3000, label: 'K (강성, N/m)' },
      { key: 'kd', default: 60, min: 0, max: 400, label: 'D (감쇠, N·s/m)' },
    ],
  },
  // 속도제어(MR §11.3): 1차계 — θ̇ 가 곧 출력. 게인 kp/ki 만(kd 없음), 스케일 작음.
  {
    name: 'joint_velocity',
    label: '관절 속도제어',
    params: [
      { key: 'kp', default: 3, min: 0, max: 10 },
      { key: 'ki', default: 0, min: 0, max: 5 },
    ],
  },
  {
    name: 'resolved_rate',
    label: 'Resolved-Rate (직교속도)',
    params: [
      { key: 'kp', default: 3, min: 0, max: 10 },
      { key: 'ki', default: 0, min: 0, max: 5 },
    ],
  },
  // 어드미턴스(MR §11.7.2): 외력→움직임. 가상 M-B-K, 외란을 f_ext 로 사용.
  {
    name: 'admittance',
    label: '어드미턴스',
    params: [
      { key: 'm', default: 2, min: 1, max: 10, label: 'M (가상질량, kg)' },
      { key: 'b', default: 40, min: 0, max: 200, label: 'B (감쇠, N·s/m)' },
      { key: 'k', default: 600, min: 0, max: 3000, label: 'K (강성, N/m)' },
    ],
  },
  // 힘 제어(MR §11.5): 가상 벽을 목표 힘으로 누름. PI 힘 피드백(kfp/kfi).
  {
    name: 'force',
    label: '힘 제어 (§11.5)',
    params: [
      {
        key: 'kfp',
        default: 0.5,
        min: 0,
        max: 2,
        step: 0.1,
        label: 'Kfp (힘 P)',
      },
      {
        key: 'kfi',
        default: 4,
        min: 0,
        max: 20,
        step: 0.5,
        label: 'Kfi (힘 I)',
      },
    ],
  },
  // 하이브리드 모션/힘(MR §11.6): 법선=힘·접선=위치 동시(투영 P). "칠판 선 긋기".
  {
    name: 'hybrid',
    label: '하이브리드 모션/힘 (§11.6)',
    params: [
      { key: 'kp', default: 100, min: 0, max: 400, label: 'Kp (모션)' },
      {
        key: 'kfp',
        default: 0.5,
        min: 0,
        max: 2,
        step: 0.1,
        label: 'Kfp (힘 P)',
      },
      {
        key: 'kfi',
        default: 4,
        min: 0,
        max: 20,
        step: 0.5,
        label: 'Kfi (힘 I)',
      },
    ],
  },
];

// 극배치(2차계) 자동튜닝이 의미있는 제어기. 속도제어(1차계)·임피던스(직교강성)는 제외.
export const POLE_PLACE = ['pd', 'pid', 'computed_torque'];

// 힘/하이브리드 제어의 환경(표면)·목표. 표면 기하(center/axis/point/normal)는 백엔드가 계산.
export interface ForceTask {
  fd: number; // 목표 누름힘(N)
  gap: number; // 시작 시 표면까지 거리(m)
  move_len: number; // 접선 이동거리(m, 하이브리드 단일획)
  shape: string; // plane | cylinder | sphere | mesh (컨투어)
  radius: number; // 곡면 반경(m, cylinder/sphere)
  // 업로드 메시 표면(shape='mesh'): id + 배치 transform(프론트가 EE 자세로 자동 계산)
  mesh_id?: string;
  mesh_pos?: number[]; // 배치 위치(base)
  mesh_quat?: number[]; // 배치 회전(쿼터니언 xyzw)
  mesh_scale?: number; // 배치 스케일(균일)
}

// 메시 업로드 응답 (mesh_store 가 부여한 id + 면수·치수)
export interface MeshUploadResponse {
  mesh_id: string;
  n_faces: number;
  extents: number[]; // 로컬 AABB 치수 [ex,ey,ez](m)
}

export interface RunRequest {
  robot_id?: string; // 대상 로봇 (레지스트리 id). 미지정 시 백엔드 ur5
  waypoints?: number[][]; // 관절 경유점(rad)
  program_id?: string;
  start?: number[]; // 프로그램 실행 출발 관절각(현재 자세)
  controller?: string; // pd | pid | computed_torque
  gains?: Record<string, number>;
  traj_mode?: string; // joint(관절 quintic) | task(직교 직선) | joint_timeopt(토크 역산)
  traj_tau_max?: number[]; // joint_timeopt: 관절 피크 토크(N·m), 필수
  traj_dq_max?: number; // joint_timeopt: 관절 속도 한계(rad/s), 필수
  gravity_comp?: boolean;
  t_seg?: number;
  hold?: number;
  payload?: number; // 끝단 미지 질량(kg), plant 에만
  model_scale?: number; // 컨트롤러 질량 배율(1.0=정확)
  disturbance?: number[]; // 끝단 외력(base 프레임, N), 모션 후 인가
  friction?: number; // 관절 쿨롱 마찰(N·m, plant) — 토크 제어만
  tau_max?: number; // 액추에이터 토크 한계(N·m, 0=무제한) — 토크 제어만
  control_rate?: number; // 0=연속(이상) / >0=이산 ZOH 제어율(Hz) — 토크 제어
  noise?: number; // 센서 측정 노이즈 std(rad), 이산 ZOH — 토크 제어
  noise_seed?: number; // 노이즈 realization seed(고정=재현성, 바꾸면 다른 패턴)
  force_task?: ForceTask; // 힘/하이브리드 제어 환경(표면)·목표
  force_path?: number[][]; // 표면에 그린 경로(base xyz) — 하이브리드 컨투어
}

export interface RunResponse {
  t: number[];
  theta: number[][]; // [frame][6] 관절각(rad)
  tcp: number[][]; // [frame][3]
  error: number[];
  torque: number[][]; // [frame][6] 토크제어 출력(N·m). 속도제어 시 빈 배열
  torque_cmd?: number[][]; // 포화 전 명령 토크(N·m). 한계 없으면 빈 배열
  qdot: number[][]; // [frame][6] 속도제어 출력 commanded 관절속도(rad/s)
  ee_pose: number[][]; // [frame][7] EE pose [x,y,z,qx,qy,qz,qw] (재생 중 직교좌표)
  sigma_min: number[]; // [frame] 최소특이값 (재생 중 manipulability)
  waypoints_tcp: number[][];
  settle_time: number | null; // 평형 도달 시각(s), 미도달 null
  steady_state_error: number | null; // 정상상태 오차(rad)
  diverged: boolean; // 발산 여부
  traj_error: string | null; // task궤적 실패 사유(특이점/작업영역밖), 없으면 null
  force?: number[]; // 측정 법선반력(N) — 힘/하이브리드
  force_des?: number | null; // 목표 힘(N)
  tan_pos?: number[]; // 접선 실제 변위(m) — 하이브리드
  tan_des?: number[]; // 접선 목표 변위(m) — 하이브리드
  // 표면 기하(3D 렌더용). plane=point/normal, cylinder=center/axis/radius, sphere=center/radius
  wall?: {
    shape: string;
    point?: number[];
    normal?: number[];
    center?: number[];
    axis?: number[];
    radius?: number;
  } | null;
}

// URDF 에서 추출한 관절 메타 (슬라이더 범위용, rad)
export interface JointMeta {
  name: string;
  lower: number;
  upper: number;
}

export interface IKResponse {
  theta: number[];
  converged: boolean;
  tcp: number[];
}

export interface FKResponse {
  pose: number[]; // [x,y,z, qx,qy,qz,qw]
  tcp: number[];
  w: number; // 조작성 √det(JJᵀ)
  sigma_min: number; // 최소 특이값 (→0 특이점)
}

export interface Pose {
  id: number;
  program_id: string;
  x: number;
  y: number;
  z: number;
  qx: number;
  qy: number;
  qz: number;
  qw: number;
  theta: number[];
}

// 멀티로봇 목록/상세
export const getRobots = () =>
  http.get<RobotSummary[]>('/robots').then((r) => r.data);
export const getRobotDetail = (id: string) =>
  http.get<RobotDetail>(`/robots/${id}`).then((r) => r.data);

export const runSimulation = (req: RunRequest) =>
  http.post<RunResponse>('/run', req).then((r) => r.data);

// 메시(STL/OBJ) 업로드 → 백엔드 인메모리 저장 후 mesh_id 반환 (하이브리드 컨투어 표면)
export const uploadMesh = (file: File) => {
  const form = new FormData();
  form.append('file', file);
  return http.post<MeshUploadResponse>('/mesh', form).then((r) => r.data);
};

// robotId 미지정 시 robot_id 생략 → 백엔드 기본 ur5 (하위호환)
export const fk = (theta: number[], robotId?: string) =>
  http
    .post<FKResponse>('/fk', { theta, robot_id: robotId })
    .then((r) => r.data);
export const ik = (pose: number[], seed?: number[], robotId?: string) =>
  http
    .post<IKResponse>('/ik', { pose, seed, robot_id: robotId })
    .then((r) => r.data);

export const getProgram = (id: string) =>
  http.get<Pose[]>(`/programs/${id}`).then((r) => r.data);
export const savePose = (id: string, pose: number[], theta: number[]) => {
  const [x, y, z, qx, qy, qz, qw] = pose;
  return http
    .post<Pose>(`/programs/${id}/poses`, { x, y, z, qx, qy, qz, qw, theta })
    .then((r) => r.data);
};
export const resetProgram = (id: string) =>
  http.delete(`/programs/${id}`).then((r) => r.data);
export const deletePose = (id: string, poseId: number) =>
  http.delete(`/programs/${id}/poses/${poseId}`).then((r) => r.data);
