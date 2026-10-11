"""E006 新开场钩子 0–14.5 s（号主否决慢亮开场后重做）：第 1 帧就在街里高速冲刺，人声立刻开口，大号波长计数器。
用法：python anim_hook.py <chunk> <nchunks> <outdir>"""
import sys, os, numpy as np
from PIL import Image, ImageDraw, ImageFont
import street
FPS = 24; DUR = 14.5
SUBS = [(0.15, 4.25, '你这辈子见过的所有颜色，加起来，只有这么长'),
        (7.4, 10.4, '到这道门，人眼就到头了'),
        (10.6, 14.2, '可门外的光，比这条街长两百多亿倍')]
def ease_out(x): x = np.clip(x, 0, 1); return 1 - (1 - x) ** 3
def zpos(t):
    if t < 4.3: return -4 + 244 * (t / 4.3) ** 1.3          # 起步即高速
    if t < 7.3: return 240 + 82 * ease_out((t - 4.3) / 3.0)  # 减速逼近门
    if t < 10.5: return 322 + 4 * (t - 7.3) / 3.2              # 门前停住，缓推
    return 326 + 110 * ease_out((t - 10.5) / 3.0)               # 穿门入黑
def cam(t):
    sway = 0.8 * np.sin(t * 1.3)
    return dict(cam_z=zpos(t), cam_h=4.5 + 0.5 * np.sin(t * 1.7), pitch=-0.04, cam_x=-1.0 + sway,
                yaw=0.015 * np.sin(t * 1.1), fov=1.15 if t < 4.3 else 1.05,
                title='', sub='', labels=False, gate_label=False, uv_label=False, save=False)
def overlay(im, t):
    W, H = im.size; o = Image.new('RGBA', im.size, (0, 0, 0, 0)); d = ImageDraw.Draw(o)
    B = ImageFont.truetype(street.CJK_B, 64); R = ImageFont.truetype(street.CJK_B, 46)
    lam = 380 + zpos(t) + 6 + 18 * np.clip((zpos(t) - 200) / 122, 0, 1)
    if t < 10.5:
        lam_c = 750 if t >= 7.3 else int(np.clip(lam, 380, 750))
        c = street.mono_rgb(np.array([float(lam_c)]))[0]; c = c / max(c.max(), 1e-6)
        col = tuple(int(255 * (0.35 + 0.65 * v)) for v in c) + (255,)
        d.text((W // 2, 120), f'{lam_c} nm', font=ImageFont.truetype(street.CJK_B, 120), fill=col, anchor='mm',
               stroke_width=6, stroke_fill=(0, 0, 0, 200))
        if t >= 7.3: d.text((W // 2, 215), '人眼的尽头', font=R, fill=(255, 200, 185, 255), anchor='mm', stroke_width=4, stroke_fill=(0, 0, 0, 200))
    else:
        a = int(255 * np.clip((t - 10.6) / 0.4, 0, 1))
        d.text((W // 2, 140), '→ 红外 · 微波 · 无线电', font=R, fill=(220, 120, 100, a), anchor='mm')
    for a0, a1, s in SUBS:
        if a0 <= t <= a1:
            d.text((W // 2, H - 110), s, font=B, fill=(255, 255, 255, 255), anchor='mm', stroke_width=5, stroke_fill=(0, 0, 0, 230))
    return Image.alpha_composite(im.convert('RGBA'), o).convert('RGB')
def frame(t):
    # 高速段做 3 子帧运动模糊
    n = 3 if t < 7.3 else 1
    acc = None
    for k in range(n):
        tt = t + (k - (n - 1) / 2) / FPS / 2
        a = np.asarray(street.render(None, **cam(tt))).astype(np.float32)
        acc = a if acc is None else acc + a
    return overlay(Image.fromarray((acc / n).astype(np.uint8)), t)
if __name__ == '__main__':
    ci, nc, out = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
    os.makedirs(out, exist_ok=True)
    for i in range(ci, int(FPS * DUR), nc):
        frame(i / FPS).save(f'{out}/f{i:05d}.jpg', quality=92); print(i, flush=True)
