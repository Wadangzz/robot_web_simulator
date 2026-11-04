/**
 * 시뮬레이터 전역 상태 (zustand + persist).
 *
 * 라우트를 오가면 App 이 언마운트돼 useState 가 날아간다. 여기 둔 상태는 React 트리
 * 밖에 살아 라우트 전환에도 유지되고, persist(localStorage) 로 새로고침에도 보존된다.
 *
 * setter 는 useState 와 동일한 시그니처(값 또는 갱신함수)라, 컴포넌트의 호출부
 * (`setX(v)` / `setX(prev => ...)`)를 그대로 둔 채 선언만 store 셀렉터로 바꾸면 된다.
 */
import { create } from 'zustand';
import { persist } from 'zustand/middleware';

type Updater<T> = T | ((prev: T) => T);
const apply = <T>(prev: T, u: Updater<T>): T =>
  typeof u === 'function' ? (u as (p: T) => T)(prev) : u;

export interface ForceTask {
  fd: number;
  gap: number;
  move_len: number;
  shape: string; // plane | cylinder | sphere | mesh
  radius: number;
  mesh_scale: number;
}

interface SimState {
  robotId: string;
  controller: string;
  trajMode: string;
  trajTauMax: number[]; // joint_timeopt 관절별 토크 한계(N·m), []=미입력
  trajDqMax: number; // joint_timeopt 속도 한계(rad/s), 0=미입력
  gains: Record<string, number>;
  autoTune: boolean;
  targetTs: number;
  payload: number;
  modelScale: number;
  friction: number;
  tauMax: number;
  controlRate: number;
  noise: number;
  forceTask: ForceTask;
  push: number[];
  joints: number[]; // 현재 포즈 — 와리가리/새로고침 후 복원
  drawnPath: number[][];
  camPos: number[]; // OrbitControls 카메라 위치 (줌·회전)
  camTarget: number[]; // OrbitControls 타깃 (팬)

  setRobotId: (v: Updater<string>) => void;
  setController: (v: Updater<string>) => void;
  setTrajMode: (v: Updater<string>) => void;
  setTrajTauMax: (v: Updater<number[]>) => void;
  setTrajDqMax: (v: Updater<number>) => void;
  setGains: (v: Updater<Record<string, number>>) => void;
  setAutoTune: (v: Updater<boolean>) => void;
  setTargetTs: (v: Updater<number>) => void;
  setPayload: (v: Updater<number>) => void;
  setModelScale: (v: Updater<number>) => void;
  setFriction: (v: Updater<number>) => void;
  setTauMax: (v: Updater<number>) => void;
  setControlRate: (v: Updater<number>) => void;
  setNoise: (v: Updater<number>) => void;
  setForceTask: (v: Updater<ForceTask>) => void;
  setPush: (v: Updater<number[]>) => void;
  setJoints: (v: Updater<number[]>) => void;
  setDrawnPath: (v: Updater<number[][]>) => void;
  setCamPos: (v: Updater<number[]>) => void;
  setCamTarget: (v: Updater<number[]>) => void;
}

export const useSim = create<SimState>()(
  persist(
    (set) => ({
      robotId: 'ur5',
      controller: 'computed_torque',
      trajMode: 'joint',
      trajTauMax: [],
      trajDqMax: 0,
      gains: {},
      autoTune: false,
      targetTs: 0.6,
      payload: 0,
      modelScale: 1,
      friction: 0,
      tauMax: 0,
      controlRate: 0,
      noise: 0,
      forceTask: {
        fd: 20,
        gap: 0.05,
        move_len: 0.1,
        shape: 'plane',
        radius: 0.08,
        mesh_scale: 1,
      },
      push: [0, 0, 0],
      joints: [],
      drawnPath: [],
      camPos: [1.3, -1.3, 0.9],
      camTarget: [0, 0, 0.4],

      setRobotId: (v) => set((s) => ({ robotId: apply(s.robotId, v) })),
      setController: (v) =>
        set((s) => ({ controller: apply(s.controller, v) })),
      setTrajMode: (v) => set((s) => ({ trajMode: apply(s.trajMode, v) })),
      setTrajTauMax: (v) =>
        set((s) => ({ trajTauMax: apply(s.trajTauMax, v) })),
      setTrajDqMax: (v) => set((s) => ({ trajDqMax: apply(s.trajDqMax, v) })),
      setGains: (v) => set((s) => ({ gains: apply(s.gains, v) })),
      setAutoTune: (v) => set((s) => ({ autoTune: apply(s.autoTune, v) })),
      setTargetTs: (v) => set((s) => ({ targetTs: apply(s.targetTs, v) })),
      setPayload: (v) => set((s) => ({ payload: apply(s.payload, v) })),
      setModelScale: (v) =>
        set((s) => ({ modelScale: apply(s.modelScale, v) })),
      setFriction: (v) => set((s) => ({ friction: apply(s.friction, v) })),
      setTauMax: (v) => set((s) => ({ tauMax: apply(s.tauMax, v) })),
      setControlRate: (v) =>
        set((s) => ({ controlRate: apply(s.controlRate, v) })),
      setNoise: (v) => set((s) => ({ noise: apply(s.noise, v) })),
      setForceTask: (v) => set((s) => ({ forceTask: apply(s.forceTask, v) })),
      setPush: (v) => set((s) => ({ push: apply(s.push, v) })),
      setJoints: (v) => set((s) => ({ joints: apply(s.joints, v) })),
      setDrawnPath: (v) => set((s) => ({ drawnPath: apply(s.drawnPath, v) })),
      setCamPos: (v) => set((s) => ({ camPos: apply(s.camPos, v) })),
      setCamTarget: (v) => set((s) => ({ camTarget: apply(s.camTarget, v) })),
    }),
    {
      name: 'ur5-sim',
      // 설정·포즈만 저장(runtime 객체는 컴포넌트 로컬에 둠). 저장 끄려면 이 블록 제거.
      partialize: (s) => ({
        robotId: s.robotId,
        controller: s.controller,
        trajMode: s.trajMode,
        trajTauMax: s.trajTauMax,
        trajDqMax: s.trajDqMax,
        gains: s.gains,
        autoTune: s.autoTune,
        targetTs: s.targetTs,
        payload: s.payload,
        modelScale: s.modelScale,
        friction: s.friction,
        tauMax: s.tauMax,
        controlRate: s.controlRate,
        noise: s.noise,
        forceTask: s.forceTask,
        push: s.push,
        joints: s.joints,
        drawnPath: s.drawnPath,
        camPos: s.camPos,
        camTarget: s.camTarget,
      }),
    },
  ),
);
