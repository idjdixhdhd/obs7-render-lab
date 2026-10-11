"""E006 S01–S03（0–22 s）：街亮起 → 降到低空沿街飞 → 500 nm 急停 → 撞上 750 nm 门框。
用法：python anim_s01_s03.py <chunk> <nchunks> <outdir> [scale]"""
import sys, os, numpy as np
import street
FPS = 24; DUR = 22.0
def ease(x): x = np.clip(x, 0, 1); return x * x * (3 - 2 * x)
def lerp(a, b, x): return a + (b - a) * x
def cam(t):
    # 0–3 s：俯瞰，街从 380 端逐米亮起
    if t < 3:
        return dict(cam_z=-45, cam_h=38, pitch=-0.25, cam_x=-2.5, yaw=0.07, reveal=lerp(0, 370, ease(t / 2.6)),
                    title='', sub='你这辈子见过的所有颜色', labels=False, gate_label=False, uv_label=False)
    # 3–6 s：俯冲下降到街口低空
    if t < 6:
        x = ease((t - 3) / 3)
        return dict(cam_z=lerp(-45, -6, x), cam_h=lerp(38, 5, x), pitch=lerp(-0.25, -0.05, x), cam_x=lerp(-2.5, -1.5, x),
                    yaw=lerp(0.07, 0.02, x), title='1 纳米 = 1 米', sub='', labels=True, gate_label=False, uv_label=False)
    # 6–9 s：加速飞到 500 nm（z=120），9–10 s 急停，10–22 s 再飞到门前（z=355）
    if t < 9:
        z = lerp(-6, 118, ease((t - 6) / 3)); sub = ''
    elif t < 10.2:
        z = 118 + 2 * (t - 9) / 1.2; sub = '500 纳米 · 这里是绿'
    else:
        z = lerp(120, 352, ((t - 10.2) / 11.8) ** 1.15); sub = ''
    sway = 0.9 * np.sin(t * 0.7)
    title = '人眼到此为止' if t > 19 else '1 纳米 = 1 米'
    return dict(cam_z=z, cam_h=5 + 0.6 * np.sin(t * 0.9), pitch=-0.05, cam_x=-1.5 + sway, yaw=0.02 - 0.015 * np.sin(t * 0.7),
                title=title, sub=sub, labels=t < 19, gate_label=t > 19, uv_label=False)
if __name__ == '__main__':
    ci, nc, out = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
    sc = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0
    street.W, street.H = int(1920 * sc), int(1080 * sc)
    os.makedirs(out, exist_ok=True)
    N = int(FPS * DUR)
    for i in range(ci, N, nc):
        im = street.render(None, fov=1.0, save=False, **cam(i / FPS))
        if sc != 1: im = im.resize((1920, 1080), 3)
        im.save(f'{out}/f{i:05d}.jpg', quality=92)
        print(i, flush=True)
