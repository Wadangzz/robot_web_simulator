import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Slider } from '@/components/ui/slider';
import { Label } from '@/components/ui/label';
import { Button } from '@/components/ui/button';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { Switch } from '@/components/ui/switch';
import { Input } from '@/components/ui/input';
import { POLE_PLACE, type ControllerSpec, type JointMeta } from '@/api';

interface Props {
  meta: JointMeta[];
  onReset: () => void;
  onReady: () => void;
  controllers: ControllerSpec[];
  controller: string;
  onControllerChange: (value: string) => void;
  trajMode: string;
  onTrajModeChange: (value: string) => void;
  trajTauMax: number[]; // joint_timeopt 관절별 토크 한계(N·m), 0=미입력
  onTrajTauMaxChange: (index: number, value: number) => void;
  trajDqMax: number; // joint_timeopt 속도 한계(rad/s), 0=미입력
  onTrajDqMaxChange: (value: number) => void;
  gains: Record<string, number>;
  onGainChange: (key: string, value: number) => void;
  autoTune: boolean;
  onAutoTuneChange: (on: boolean) => void;
  targetTs: number;
  onTargetTsChange: (value: number) => void;
  payload: number;
  onPayloadChange: (value: number) => void;
  modelScale: number;
  onModelScaleChange: (value: number) => void;
  friction: number;
  onFrictionChange: (value: number) => void;
  tauMax: number;
  onTauMaxChange: (value: number) => void;

  controlRate: number;
  onControlRateChange: (value: number) => void;
  noise: number;
  onNoiseChange: (value: number) => void;
  noiseSeed: number;
  forceTask: {
    fd: number;
    gap: number;
    move_len: number;
    shape: string;
    radius: number;
    mesh_scale: number;
  };
  onForceTaskChange: (
    key: 'fd' | 'gap' | 'move_len' | 'radius' | 'shape' | 'mesh_scale',
    value: number | string,
  ) => void;
  onMeshUpload: (file: File) => void;
  meshInfo: { n_faces: number; extents: number[] } | null; // 업로드 완료 정보
  push: number[];
  onPushAxisChange: (index: number, value: number) => void;
  onRun: () => void;
  running: boolean;
}

// 2차 PD 오차동역학 ë + Kd·ė + Kp·e = 0  →  ωn=√Kp, ζ=Kd/(2√Kp)
// (CT 는 비선형 상쇄 후 정확, PD/PID 는 M(θ) 변동으로 근사)
// Ki>0 이면 3차 s³+Kd·s²+Kp·s+Ki → Routh 안정조건 Ki<Kp·Kd, margin=Ki/(Kp·Kd)
function damping(kp?: number, kd?: number, ki?: number) {
  if (!kp || kp <= 0 || kd == null) return null;
  const wn = Math.sqrt(kp);
  const zeta = kd / (2 * wn);
  const label = zeta < 0.97 ? '과소감쇠' : zeta > 1.03 ? '과감쇠' : '임계감쇠';
  const color =
    zeta < 0.97
      ? 'text-amber-600'
      : zeta > 1.03
        ? 'text-blue-600'
        : 'text-emerald-600';
  const margin = ki && ki > 0 ? ki / (kp * kd) : null;
  return { wn, zeta, label, color, margin };
}

// 속도제어 PI 오차동역학: ë + Kp·ė + Ki·e = 0 → ωn=√Ki, ζ=Kp/(2√Ki).
// 특성식 s²+Kp·s+Ki 는 Kp,Ki>0 이면 두 계수 양수 → 무조건 안정(토크 PID 와 달리
// 적분 발산조건 없음). Ki=0 이면 1차계(진동 없음). Ki>0 일 때만 ζ/ωn 의미.
function velDamping(kp?: number, ki?: number) {
  if (!ki || ki <= 0 || !kp || kp <= 0) return null;
  const wn = Math.sqrt(ki);
  const zeta = kp / (2 * wn);
  const label = zeta < 0.97 ? '과소감쇠' : zeta > 1.03 ? '과감쇠' : '임계감쇠';
  const color =
    zeta < 0.97
      ? 'text-amber-600'
      : zeta > 1.03
        ? 'text-blue-600'
        : 'text-emerald-600';
  return { wn, zeta, label, color };
}

/** 관절 슬라이더 + 제어기/게인 튜닝 + Run 패널. */
export default function ControlPanel({
  meta,
  onReset,
  onReady,
  controllers,
  controller,
  onControllerChange,
  trajMode,
  onTrajModeChange,
  trajTauMax,
  onTrajTauMaxChange,
  trajDqMax,
  onTrajDqMaxChange,
  gains,
  onGainChange,
  autoTune,
  onAutoTuneChange,
  targetTs,
  onTargetTsChange,
  payload,
  onPayloadChange,
  modelScale,
  onModelScaleChange,
  friction,
  onFrictionChange,
  tauMax,
  onTauMaxChange,
  controlRate,
  onControlRateChange,
  noise,
  onNoiseChange,
  noiseSeed,
  forceTask,
  onForceTaskChange,
  onMeshUpload,
  meshInfo,
  push,
  onPushAxisChange,
  onRun,
  running,
}: Props) {
  if (meta.length === 0) return null;
  const spec = controllers.find((c) => c.name === controller);
  const d = damping(gains.kp, gains.kd, gains.ki);
  const isVel =
    controller === 'joint_velocity' || controller === 'resolved_rate';
  const vd = isVel ? velDamping(gains.kp, gains.ki) : null;

  return (
    <Card className='bg-card/95 supports-[backdrop-filter]:bg-card/80 w-72 backdrop-blur'>
      <CardHeader className='pb-3'>
        <CardTitle className='text-sm'>제어 · 시뮬레이션</CardTitle>
      </CardHeader>
      <CardContent className='space-y-4'>
        <div className='space-y-3'>
          <div className='space-y-1.5'>
            <Label className='text-muted-foreground text-xs'>제어기</Label>
            <Select
              value={controller}
              onValueChange={onControllerChange}
              disabled={running}
            >
              <SelectTrigger className='w-full' size='sm'>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {controllers.map((c) => (
                  <SelectItem key={c.name} value={c.name}>
                    {c.label}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          {/* 궤적 방식 — 제어기와 독립. 직교(직선)는 특이점서 실행불가가 정상 */}
          <div className='space-y-1.5'>
            <Label className='text-muted-foreground text-xs'>궤적 방식</Label>
            <Select
              value={trajMode}
              onValueChange={onTrajModeChange}
              disabled={running}
            >
              <SelectTrigger className='w-full' size='sm'>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value='joint'>관절공간 (곡선)</SelectItem>
                <SelectItem value='task'>직교 직선 (task)</SelectItem>
                <SelectItem value='joint_timeopt'>
                  관절 토크역산 (timeopt)
                </SelectItem>
              </SelectContent>
            </Select>
          </div>

          {/* timeopt 한계 — 백엔드 필수 입력(URDF effort 안 씀). 미입력(0)이면
              실행 시 traj_error 로 "실행 불가"가 그대로 표시된다. */}
          {trajMode === 'joint_timeopt' && (
            <div className='bg-muted/50 space-y-3 rounded-md p-2.5'>
              <p className='text-muted-foreground text-xs font-medium'>
                토크·속도 한계 (필수)
              </p>
              <div className='space-y-1.5'>
                <Label className='text-muted-foreground text-xs'>
                  관절 토크 한계 (N·m)
                </Label>
                <div className='grid grid-cols-3 gap-1.5'>
                  {meta.map((m, i) => (
                    <div key={m.name} className='space-y-0.5'>
                      <span className='text-muted-foreground text-[11px]'>
                        J{i + 1}
                      </span>
                      <Input
                        type='number'
                        min={0}
                        step={1}
                        value={trajTauMax[i] ?? 0}
                        disabled={running}
                        onChange={(e) =>
                          onTrajTauMaxChange(i, parseFloat(e.target.value) || 0)
                        }
                        className='h-8'
                      />
                    </div>
                  ))}
                </div>
              </div>
              <div className='space-y-1.5'>
                <Label className='text-muted-foreground text-xs'>
                  속도 한계 (rad/s)
                </Label>
                <Input
                  type='number'
                  min={0}
                  step={0.1}
                  value={trajDqMax}
                  disabled={running}
                  onChange={(e) =>
                    onTrajDqMaxChange(parseFloat(e.target.value) || 0)
                  }
                  className='h-8'
                />
              </div>
              <p className='text-muted-foreground text-[11px]'>
                0 = 미지정. 미지정으로 실행하면 궤적 생성 불가 에러가 표시된다.
              </p>
            </div>
          )}

          {/* 자동 튜닝(극배치) — 관절 2차계용. 속도제어(1차계)·임피던스엔 N/A 라 숨김 */}
          {POLE_PLACE.includes(controller) && (
            <>
              <div className='flex items-center justify-between'>
                <Label className='text-muted-foreground text-xs'>
                  자동 튜닝 (극배치)
                </Label>
                <Switch
                  checked={autoTune}
                  onCheckedChange={onAutoTuneChange}
                  disabled={running}
                />
              </div>
              {autoTune && (
                <div className='space-y-1.5'>
                  <div className='flex items-center justify-between'>
                    <Label className='text-muted-foreground text-xs'>
                      목표 정착시간
                    </Label>
                    <span className='text-xs tabular-nums'>
                      {targetTs.toFixed(2)}s
                    </span>
                  </div>
                  <Slider
                    min={0.2}
                    max={1.5}
                    step={0.05}
                    value={[targetTs]}
                    disabled={running}
                    onValueChange={([v]) => onTargetTsChange(v)}
                  />
                </div>
              )}
            </>
          )}

          {/* 제어기별 게인 슬라이더 (자동 모드면 계산값 표시·잠금) */}
          {spec?.params.map((p) => (
            <div key={p.key} className='space-y-1.5'>
              <div className='flex items-center justify-between'>
                <Label className='text-muted-foreground text-xs'>
                  {p.label ?? p.key.toUpperCase()}
                </Label>
                <span className='text-xs tabular-nums'>
                  {(gains[p.key] ?? p.default).toFixed(
                    p.step && p.step < 1 ? 1 : 0,
                  )}
                </span>
              </div>
              <Slider
                min={p.min}
                max={p.max}
                step={p.step ?? 1}
                value={[gains[p.key] ?? p.default]}
                disabled={
                  running || (autoTune && POLE_PLACE.includes(controller))
                }
                onValueChange={([v]) => onGainChange(p.key, v)}
              />
            </div>
          ))}

          {/* 힘/하이브리드 제어: 환경(벽)·목표 사양 (게인과 별개) */}
          {(controller === 'force' || controller === 'hybrid') && (
            <div className='bg-muted/50 space-y-3 rounded-md p-2.5'>
              <p className='text-muted-foreground text-xs font-medium'>
                표면 · 힘 목표
              </p>
              <div className='space-y-1.5'>
                <Label className='text-muted-foreground text-xs'>표면</Label>
                <Select
                  value={forceTask.shape}
                  onValueChange={(v) => onForceTaskChange('shape', v)}
                  disabled={running}
                >
                  <SelectTrigger className='w-full' size='sm'>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value='plane'>평면 (벽)</SelectItem>
                    <SelectItem value='cylinder'>원기둥</SelectItem>
                    <SelectItem value='sphere'>구</SelectItem>
                    <SelectItem value='mesh'>메시 (STL/OBJ)</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              {(forceTask.shape === 'cylinder' ||
                forceTask.shape === 'sphere') && (
                <div className='space-y-1.5'>
                  <div className='flex items-center justify-between'>
                    <Label className='text-muted-foreground text-xs'>
                      곡면 반경
                    </Label>
                    <span className='text-xs tabular-nums'>
                      {(forceTask.radius * 100).toFixed(0)} cm
                    </span>
                  </div>
                  <Slider
                    min={0.03}
                    max={0.25}
                    step={0.01}
                    value={[forceTask.radius]}
                    disabled={running}
                    onValueChange={([v]) => onForceTaskChange('radius', v)}
                  />
                </div>
              )}
              {forceTask.shape === 'mesh' && (
                <div className='space-y-1.5'>
                  <Label className='text-muted-foreground text-xs'>
                    메시 파일
                  </Label>
                  <input
                    type='file'
                    accept='.stl,.obj'
                    disabled={running}
                    onChange={(e) => {
                      const f = e.target.files?.[0];
                      if (f) onMeshUpload(f);
                    }}
                    className='text-muted-foreground file:bg-muted file:text-foreground hover:file:bg-muted/70 w-full text-xs file:mr-2 file:rounded file:border-0 file:px-2 file:py-1 file:text-xs'
                  />
                  {meshInfo && (
                    <p className='text-muted-foreground text-xs tabular-nums'>
                      {meshInfo.n_faces} 면 ·{' '}
                      {meshInfo.extents
                        .map((e) => (e * 100).toFixed(0))
                        .join('×')}{' '}
                      cm
                    </p>
                  )}
                  <div className='flex items-center justify-between pt-1'>
                    <Label className='text-muted-foreground text-xs'>
                      스케일
                    </Label>
                    <span className='text-xs tabular-nums'>
                      {forceTask.mesh_scale.toFixed(2)}×
                    </span>
                  </div>
                  <Slider
                    min={0.1}
                    max={3}
                    step={0.05}
                    value={[forceTask.mesh_scale]}
                    disabled={running || !meshInfo}
                    onValueChange={([v]) => onForceTaskChange('mesh_scale', v)}
                  />
                </div>
              )}
              <div className='space-y-1.5'>
                <div className='flex items-center justify-between'>
                  <Label className='text-muted-foreground text-xs'>
                    목표 힘 Fd
                  </Label>
                  <span className='text-xs tabular-nums'>
                    {forceTask.fd.toFixed(0)} N
                  </span>
                </div>
                <Slider
                  min={0}
                  max={50}
                  step={1}
                  value={[forceTask.fd]}
                  disabled={running}
                  onValueChange={([v]) => onForceTaskChange('fd', v)}
                />
              </div>
              <div className='space-y-1.5'>
                <div className='flex items-center justify-between'>
                  <Label className='text-muted-foreground text-xs'>
                    접근 간격
                  </Label>
                  <span className='text-xs tabular-nums'>
                    {(forceTask.gap * 1000).toFixed(0)} mm
                  </span>
                </div>
                <Slider
                  min={0.02}
                  max={0.1}
                  step={0.005}
                  value={[forceTask.gap]}
                  disabled={running}
                  onValueChange={([v]) => onForceTaskChange('gap', v)}
                />
              </div>
              {/* 하이브리드 경로는 3D 표면에 점을 찍어 지정(force_path). 경로 미지정 시
                  백엔드 기본값(move_len)으로 직선 한 획 — 별도 슬라이더 없음. */}
            </div>
          )}

          {/* 감쇠비 / 적분 안정여유 배지 (관절 2차계 기준 — 임피던스 제외) */}
          {controller !== 'impedance' && d && (
            <div className='bg-muted space-y-1 rounded-md px-2.5 py-1.5 text-xs'>
              <div className='flex items-center justify-between'>
                <span className='text-muted-foreground tabular-nums'>
                  ζ={d.zeta.toFixed(2)} · ωn={d.wn.toFixed(1)}
                </span>
                <span className={`font-medium ${d.color}`}>{d.label}</span>
              </div>
              {d.margin != null && (
                <div className='border-border/50 flex items-center justify-between border-t pt-1'>
                  <span className='text-muted-foreground tabular-nums'>
                    적분여유 Ki/(Kp·Kd)={d.margin.toFixed(2)}
                  </span>
                  <span
                    className={`font-medium ${d.margin < 1 ? 'text-emerald-600' : 'text-red-600'}`}
                  >
                    {d.margin < 1 ? '안정' : '발산위험'}
                  </span>
                </div>
              )}
            </div>
          )}

          {/* 속도제어 PI 감쇠비 배지 — ë+Kp·ė+Ki·e=0 (2차계, 무조건 안정) */}
          {isVel && (
            <div className='bg-muted space-y-1 rounded-md px-2.5 py-1.5 text-xs'>
              <div className='flex items-center justify-between'>
                <span className='text-muted-foreground tabular-nums'>
                  {vd
                    ? `ζ=${vd.zeta.toFixed(2)} · ωn=${vd.wn.toFixed(1)}`
                    : '1차계 (적분 없음)'}
                </span>
                <span
                  className={`font-medium ${vd ? vd.color : 'text-emerald-600'}`}
                >
                  {vd ? vd.label : '안정'}
                </span>
              </div>
              {vd && (
                <div className='border-border/50 flex items-center justify-between border-t pt-1'>
                  <span className='text-muted-foreground'>
                    2차계 (s²+Kp·s+Ki)
                  </span>
                  <span className='font-medium text-emerald-600'>
                    무조건 안정
                  </span>
                </div>
              )}
            </div>
          )}

          {/* 모델 불확실성: plant(진짜) ≠ controller(아는 모델) */}
          <div className='space-y-3 border-t pt-3'>
            <p className='text-muted-foreground text-xs font-medium'>
              모델 불확실성
            </p>
            <div className='space-y-1.5'>
              <div className='flex items-center justify-between'>
                <Label className='text-muted-foreground text-xs'>
                  페이로드 (미지 질량)
                </Label>
                <span className='text-xs tabular-nums'>
                  {payload.toFixed(1)} kg
                </span>
              </div>
              <Slider
                min={0}
                max={5}
                step={0.5}
                value={[payload]}
                disabled={running}
                onValueChange={([v]) => onPayloadChange(v)}
              />
            </div>
            <div className='space-y-1.5'>
              <div className='flex items-center justify-between'>
                <Label className='text-muted-foreground text-xs'>
                  모델 질량배율
                </Label>
                <span className='text-xs tabular-nums'>
                  ×{modelScale.toFixed(2)}
                </span>
              </div>
              <Slider
                min={0.5}
                max={1.5}
                step={0.05}
                value={[modelScale]}
                disabled={running}
                onValueChange={([v]) => onModelScaleChange(v)}
              />
            </div>

            {/* 현실 모델 (토크 plant 한정): 마찰→PD 정상상태오차, 포화→성능한계 */}
            <div className='space-y-1.5'>
              <div className='flex items-center justify-between'>
                <Label className='text-muted-foreground text-xs'>
                  마찰 (쿨롱)
                </Label>
                <span className='text-xs tabular-nums'>{friction} N·m</span>
              </div>
              <Slider
                min={0}
                max={30}
                step={1}
                value={[friction]}
                disabled={running}
                onValueChange={([v]) => onFrictionChange(v)}
              />
            </div>
            <div className='space-y-1.5'>
              <div className='flex items-center justify-between'>
                <Label className='text-muted-foreground text-xs'>
                  토크 한계
                </Label>
                <span className='text-xs tabular-nums'>
                  {tauMax === 0 ? '무제한' : `${tauMax} N·m`}
                </span>
              </div>
              <Slider
                min={0}
                max={150}
                step={5}
                value={[tauMax]}
                disabled={running}
                onValueChange={([v]) => onTauMaxChange(v)}
              />
            </div>

            {/* 제어 모드: 연속(이상) vs 이산 ZOH(실제 디지털, 낮추면 불안정) */}
            <div className='space-y-1.5'>
              <Label className='text-muted-foreground text-xs'>제어율</Label>
              <Select
                value={String(controlRate)}
                onValueChange={(v) => onControlRateChange(Number(v))}
                disabled={running}
              >
                <SelectTrigger className='w-full' size='sm'>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value='0'>연속 (이상)</SelectItem>
                  <SelectItem value='2000'>이산 2000 Hz</SelectItem>
                  <SelectItem value='1000'>이산 1000 Hz</SelectItem>
                  <SelectItem value='500'>이산 500 Hz</SelectItem>
                  <SelectItem value='200'>이산 200 Hz</SelectItem>
                </SelectContent>
              </Select>
            </div>
            <div className='space-y-1.5'>
              <div className='flex items-center justify-between'>
                <Label className='text-muted-foreground text-xs'>
                  센서 노이즈
                </Label>
                <span className='text-xs tabular-nums'>
                  {controlRate === 0
                    ? '이산 전용'
                    : noise === 0
                      ? '없음'
                      : `${noise.toFixed(3)} rad`}
                </span>
              </div>
              <Slider
                min={0}
                max={0.02}
                step={0.001}
                value={[noise]}
                disabled={running || controlRate === 0}
                onValueChange={([v]) => onNoiseChange(v)}
              />
              {/* 노이즈 realization 은 매 Run 마다 자동으로 달라짐(실제 센서처럼) */}
              {controlRate > 0 && noise > 0 && (
                <p className='text-muted-foreground text-[11px]'>
                  실행마다 다른 노이즈 패턴 (seed #{noiseSeed})
                </p>
              )}
            </div>
          </div>

          {/* 외란: 모션 후 끝단에 가하는 외력(base 프레임 3축). 컴플라이언스 시연용 */}
          <div className='space-y-2 border-t pt-3'>
            <p className='text-muted-foreground text-xs font-medium'>
              외란 (외력, N)
            </p>
            {[
              { i: 0, label: 'Fx', color: 'text-red-600' },
              { i: 1, label: 'Fy', color: 'text-emerald-600' },
              { i: 2, label: 'Fz', color: 'text-blue-600' },
            ].map((a) => (
              <div key={a.i} className='flex items-center gap-2'>
                <span className={`w-6 text-xs font-bold ${a.color}`}>
                  {a.label}
                </span>
                <Slider
                  min={-80}
                  max={80}
                  step={5}
                  value={[push[a.i] ?? 0]}
                  disabled={running}
                  onValueChange={([v]) => onPushAxisChange(a.i, v)}
                />
                <span className='w-8 text-right text-xs tabular-nums'>
                  {push[a.i] ?? 0}
                </span>
              </div>
            ))}
          </div>

          <Button
            className='w-full'
            size='sm'
            onClick={onRun}
            disabled={running}
          >
            {running
              ? '재생 중…'
              : controller === 'hybrid'
                ? 'Run ▶  벽 누르며 선 긋기'
                : controller === 'force'
                  ? 'Run ▶  현재자세에서 벽 누름'
                  : controller === 'impedance' || controller === 'admittance'
                    ? 'Run ▶  현재자세 정지 → 외란'
                    : 'Run ▶  ready → 현재자세'}
          </Button>
          <div className='flex gap-2'>
            <Button
              variant='outline'
              size='sm'
              className='flex-1'
              onClick={onReady}
              disabled={running}
            >
              Ready 자세
            </Button>
            <Button
              variant='outline'
              size='sm'
              className='flex-1'
              onClick={onReset}
              disabled={running}
            >
              홈 (0, 특이점)
            </Button>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
