# E006 配乐：全部代码合成，节奏对齐 02_layout/shotlist.csv（160 s）
import numpy as np, wave
SR=44100; T=160; t=np.arange(int(SR*T))/SR
def env(a,b,fi=1.0,fo=1.0):
    e=np.clip((t-a)/fi,0,1)*np.clip((b-t)/fo,0,1); return e
def pad(freqs,a,b,amp=0.08,fi=2,fo=2,det=0.003):
    s=np.zeros_like(t)
    for f in freqs:
        for d in (-det,0,det): s+=np.sin(2*np.pi*f*(1+d)*t+np.random.rand()*6)
    return s/len(freqs)/3*amp*env(a,b,fi,fo)
def bell(at,f,amp=0.25,dec=2.5):
    x=np.clip(t-at,0,None); m=(t>=at)
    return amp*m*np.exp(-x/dec)*(np.sin(2*np.pi*f*x)+0.4*np.sin(2*np.pi*f*2.76*x)*np.exp(-x/0.6))
D=73.42
mix=np.zeros_like(t)
# S01-S03 0–22 s：街上飞行，音高随波长上扬的扫频 + 脉冲
sweep_f=220*2**(t/22*1.0); ph=2*np.cumsum(np.pi*sweep_f)/SR
mix+=0.05*np.sin(ph)*env(0,22,1.5,0.05)
puls=(np.sin(2*np.pi*4*t)>0.6)*0.04*np.sin(2*np.pi*D*2*t)*env(3,22,2,0.05)
mix+=puls
mix+=pad([D,D*1.5,D*2],0,22,0.10,1,0.05)
# 500 nm 急停 (~9 s)：一拍全断
mix*=np.where((t>9.0)&(t<9.6),0.05,1)
# S04 22–27：踩空，仅低频冲击
mix+=bell(22.0,36.7,0.22,3.0)
# S05 27–52：尺度段，每级一个钟声，音高逐级下沉
for i,(at,f) in enumerate([(28,880),(33,587),(38,392),(43,262),(48,175)]): mix+=bell(at,f,0.18,4)
mix+=pad([D/2,D*0.75,D],27,60,0.10,4,3)
# S06 52–60：回望，高泛音
mix+=pad([D*8,D*12],52,60,0.06,3,3)
# S07-S08 60–97：水，闷低频+慢颤音
w=np.sin(2*np.pi*0.25*t)*0.5+0.5
mix+=pad([D/2,D*0.6],60,97,0.12,3,2)*(0.6+0.4*w)
# S09 97–120：太阳，转大调
mix+=pad([D,D*1.25,D*1.5,D*2],97,120,0.26,4,2)
# S10 120–150：眼睛睁开，渐强
mix+=pad([D,D*1.25,D*1.5,D*3,D*3.75],120,150,0.30,12,3)
mix+=bell(135,D*8,0.12,6)
# S11 150–160：收尾单音衰减
mix+=bell(150,D*4,0.08,5)+pad([D,D*1.5],150,160,0.06,1,6)
# 简易混响 + 立体声
def rev(x):
    y=x.copy()
    for d,g in [(0.031,0.5),(0.047,0.4),(0.071,0.3),(0.113,0.25),(0.173,0.2)]:
        n=int(d*SR); y[n:]+=g*x[:-n]
    return y
L=rev(mix); R=rev(np.roll(mix,int(0.012*SR)))
st=np.stack([L,R],1); st/=np.abs(st).max()/0.89
with wave.open('music_v1.wav','wb') as f:
    f.setnchannels(2); f.setsampwidth(2); f.setframerate(SR); f.writeframes((st*32767).astype(np.int16).tobytes())
print('ok', st.shape)
