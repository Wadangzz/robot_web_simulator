import { useEffect, useState } from 'react';
import { checkAuth, login } from '@/api';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';

// 시뮬 라우트 가드: 인증 통과해야 children(시뮬) 렌더, 아니면 로그인 폼.
// 백엔드 SIM_KEY 미설정이면 checkAuth 가 즉시 통과 → 로그인 폼 안 뜸(= dev 흐름 그대로).
export default function RequireSim({
  children,
}: {
  children: React.ReactNode;
}) {
  const [ok, setOk] = useState<boolean | null>(null);
  const [key, setKey] = useState('');
  const [error, setError] = useState(false);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    checkAuth().then(setOk);
  }, []);

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    setBusy(true);
    setError(false);
    const success = await login(key);
    setBusy(false);
    if (success) setOk(true);
    else setError(true);
  };

  if (ok === null) return null; // 권한 확인 중
  if (ok) return <>{children}</>;

  return (
    <div className='bg-background flex h-screen w-screen items-center justify-center'>
      <Card className='w-80'>
        <CardHeader>
          <CardTitle>시뮬레이션 로그인</CardTitle>
        </CardHeader>
        <CardContent>
          <form onSubmit={submit} className='space-y-3'>
            <div className='space-y-1.5'>
              <Label htmlFor='simkey' className='text-muted-foreground text-xs'>
                접근 키
              </Label>
              <Input
                id='simkey'
                type='password'
                value={key}
                onChange={(e) => setKey(e.target.value)}
                autoFocus
              />
            </div>
            {error && <p className='text-xs text-red-600'>키가 틀렸습니다.</p>}
            <Button type='submit' className='w-full' disabled={busy || !key}>
              {busy ? '확인 중…' : '입장'}
            </Button>
          </form>
        </CardContent>
      </Card>
    </div>
  );
}
