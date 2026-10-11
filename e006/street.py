"""E006 主视觉：把 380–750 nm 拉成一条街（1 nm = 1 m，街长 370 m）。
每一米的路灯发出该波长的单色光，颜色由 CIE 1931 2° CMF 计算（F9），亮度按 Y^0.45 压缩（单色光真实亮度两端极暗，屏幕不可直显；写进 sources）。
街两头之外是看不见的光：只画灰线。"""
import numpy as np, colour, warnings, sys
from PIL import Image, ImageDraw, ImageFont, ImageFilter
warnings.filterwarnings('ignore')
W, H = 1920, 1080
D = __file__.rsplit('/', 1)[0]
import os
F = os.environ.get('OBS7_FONTS', '/workspace/projects/73c81bcb-5a44-4717-b98c-068d6ecc01a2/douyin/fonts')
CJK_B, CJK_R = f'{F}/NotoSansCJKsc-Bold.otf', f'{F}/NotoSansCJKsc-Regular.otf'
MONO = '/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf'
cmf = colour.MSDS_CMFS['CIE 1931 2 Degree Standard Observer']
L = np.arange(360, 831); XYZ = np.array([cmf[l] for l in L])
M = np.array([[3.2406, -1.5372, -0.4986], [-0.9689, 1.8758, 0.0415], [0.0557, -0.2040, 1.0570]])

def mono_rgb(lam):
    """lam array (nm) -> linear sRGB of monochromatic light, gamut-mapped by adding white, brightness Y^0.45."""
    x = np.interp(lam, L, XYZ[:, 0]); y = np.interp(lam, L, XYZ[:, 1]); z = np.interp(lam, L, XYZ[:, 2])
    xyz = np.stack([x, y, z], -1)
    Y = np.maximum(y, 1e-6)
    rgb = (xyz / Y[..., None]) @ M.T
    mn = rgb.min(-1, keepdims=True); rgb = np.where(mn < 0, rgb - mn, rgb)
    rgb = rgb / rgb.max(-1, keepdims=True)
    return rgb * (Y ** 0.45)[..., None]

def enc(x):
    x = np.clip(x, 0, 1); return np.where(x <= 0.0031308, 12.92 * x, 1.055 * x ** (1 / 2.4) - 0.055)

def render(path, cam_z=-45.0, cam_h=38.0, pitch=-0.25, fov=1.0, cam_x=-2.5, yaw=0.07,
           title='1 纳米 = 1 米', sub='你见过的所有颜色，是一条 370 米的街', reveal=370.0, labels=True, gate_label=True, uv_label=True, save=True):
    hw = 9.0       # half street width (m)
    bh = 26.0      # building height
    v, u = np.mgrid[0:H, 0:W].astype(float)
    u = (u - W / 2) / (W / 2) * np.tan(fov / 2); v = -(v - H / 2) / (W / 2) * np.tan(fov / 2)
    d = np.stack([u, v * np.cos(pitch) + np.sin(pitch), -v * np.sin(pitch) + np.cos(pitch)], -1)
    d = np.stack([d[...,0]*np.cos(yaw)+d[...,2]*np.sin(yaw), d[...,1], -d[...,0]*np.sin(yaw)+d[...,2]*np.cos(yaw)], -1)
    d /= np.linalg.norm(d, axis=-1, keepdims=True)
    ox = cam_x
    INF = 1e9
    tg = np.where(d[..., 1] < 0, -cam_h / np.minimum(d[..., 1], -1e-9), INF)
    tl = np.where(d[..., 0] < 0, (-hw - ox) / np.minimum(d[..., 0], -1e-9), INF)
    tr = np.where(d[..., 0] > 0, (hw - ox) / np.maximum(d[..., 0], 1e-9), INF)
    tl = np.where(tl > 0, tl, INF); tr = np.where(tr > 0, tr, INF)
    t = np.minimum(tg, np.minimum(tl, tr))
    P = d * t[..., None]; P[..., 0] += ox; P[..., 1] += cam_h; P[..., 2] += cam_z
    zz = P[..., 2]
    blk = np.floor(P[..., 2] / 8).astype(int); side = (P[..., 0] > 0).astype(int)
    hts = 10 + 30 * np.random.default_rng(11).random((200, 2))
    bhp = hts[np.clip(blk, 0, 199), side]
    wall = (t < tg) & (P[..., 1] < bhp)
    # rays passing over a building top: fall to ground plane; outside street = rooftop (dark)
    over = (t < tg) & ~wall
    t = np.where(over, tg, t)
    P = d * t[..., None]; P[..., 0] += ox; P[..., 1] += cam_h; P[..., 2] += cam_z
    roof = (t == tg) & (np.abs(P[..., 0]) > hw)
    ground = (t == tg) & ~roof
    sky = ~(wall | ground | roof) | (tg >= 1e8) & ~wall
    lam = 380 + zz
    onstreet = (zz >= 0) & (zz <= 370) & (zz <= reveal)
    col = mono_rgb(np.clip(lam, 380, 750))
    # lamps every 5 m on both curbs at height 6 m; light pools
    zl = np.round(zz / 5) * 5
    dz = zz - zl
    img = np.zeros((H, W, 3))
    # ground: pools under lamps near both curbs + center lane glow
    xg = P[..., 0]
    pool = np.exp(-(dz ** 2) / 6) * (np.exp(-((np.abs(xg) - hw + 1.2) ** 2) / 10) * 1.0 + 0.25)
    wet = 0.18 + 0.82 * pool
    img += (ground & onstreet)[..., None] * col * wet[..., None] * 0.9
    # walls: lit windows grid
    wx = zz; wy = P[..., 1]
    win = ((np.mod(wx, 3.0) > 0.7) & (np.mod(wy, 3.4) > 1.2) & (wy > 1.0)).astype(float)
    rng = np.random.default_rng(3)
    cellid = (np.floor(wx / 3).astype(int) * 37 + np.floor(wy / 3.4).astype(int) * 11 + (xg > 0) * 5) % 997
    lit = (rng.random(997)[cellid] > 0.35).astype(float)
    wallc = 0.06 + 0.9 * win * lit + 0.25 * np.exp(-(dz ** 2) / 4) * np.exp(-((wy - 6) ** 2) / 8)
    img += (wall & onstreet)[..., None] * col * wallc[..., None]
    # off-street: invisible light, grey wireframe only
    img += roof[..., None] * np.array([0.0015, 0.0018, 0.003])
    off = ~onstreet & ~sky & ~roof
    grid = ((np.abs(np.mod(zz, 10) - 5) > 4.85) & ground).astype(float)
    img += off[..., None] * np.array([0.04, 0.045, 0.05]) * grid[..., None]
    lum = img @ np.array([0.2126, 0.7152, 0.0722])
    vio = np.clip((450 - lam) / 70, 0, 1) * onstreet
    img = img * (1 - 0.3 * vio[..., None]) + lum[..., None] * 0.3 * vio[..., None]
    # distance fog toward horizon
    fog = np.exp(-np.clip(t, 0, 1e4) / 900)
    img *= fog[..., None]
    img += sky[..., None] * np.array([0.002, 0.0025, 0.005]) * (1 - np.clip(v, 0, 1))[..., None]
    # haze glow above street (light scattering)
    a_ = 2.51; b_ = 0.03; c_ = 2.43; d_ = 0.59; e_ = 0.14
    img = img * 1.6; img = np.clip((img*(a_*img+b_))/(img*(c_*img+d_)+e_), 0, 1)
    # bloom
    base = enc(img)
    im = Image.fromarray((base * 255).astype(np.uint8))
    bl = np.asarray(im.filter(ImageFilter.GaussianBlur(18))).astype(float) / 255
    bl2 = np.asarray(im.filter(ImageFilter.GaussianBlur(60))).astype(float) / 255
    out = 1 - (1 - base) * (1 - bl * 0.45) * (1 - bl2 * 0.35)
    out = np.clip(out + np.random.default_rng(1).normal(0, 0.01, out.shape), 0, 1)
    im = Image.fromarray((out * 255).astype(np.uint8)).convert('RGBA')
    o = Image.new('RGBA', im.size, (0, 0, 0, 0)); dr = ImageDraw.Draw(o)
    # labels at projected points on right curb
    f = ImageFont.truetype(MONO, 26)
    for nm in ([400, 500, 600] if labels else []):
        z = nm - 380 - cam_z; X, Y = hw - 0.5 - cam_x, 0.0 - cam_h
        X, z = X*np.cos(yaw) - z*np.sin(yaw), X*np.sin(yaw) + z*np.cos(yaw)
        yv = Y * np.cos(pitch) - z * np.sin(pitch); zc = Y * np.sin(pitch) + z * np.cos(pitch)
        if zc < 2: continue
        sx = W / 2 + X / zc / np.tan(fov / 2) * W / 2; sy = H / 2 - yv / zc / np.tan(fov / 2) * W / 2
        if 0 < sx < W and 0 < sy < H:
            s = max(12, min(60, int(26 * 40 / (zc + 20))))
            dr.text((sx - 6, sy), f'{nm}', font=ImageFont.truetype(MONO, s), fill=(210, 216, 226, 200), anchor='rb')
    def proj(x, y, nm):
        z = nm - 380 - cam_z; X, Y = x - cam_x, y - cam_h
        X, z = X*np.cos(yaw) - z*np.sin(yaw), X*np.sin(yaw) + z*np.cos(yaw)
        yv = Y * np.cos(pitch) - z * np.sin(pitch); zc = Y * np.sin(pitch) + z * np.cos(pitch)
        if zc < 0.5: return None
        return (W / 2 + X / zc / np.tan(fov / 2) * W / 2, H / 2 - yv / zc / np.tan(fov / 2) * W / 2)
    # gate at 750 nm: end of visible
    g = Image.new('RGBA', im.size, (0, 0, 0, 0)); gd = ImageDraw.Draw(g)
    pts = [proj(-hw, 0, 750), proj(-hw, 22, 750), proj(hw, 22, 750), proj(hw, 0, 750)]
    gate_a = int(255 * np.clip((reveal - 360) / 10, 0, 1))
    if all(p is not None for p in pts) and gate_a > 0: gd.line(pts, fill=(255, 120, 90, gate_a), width=max(3, int(5 * W / 1920)))
    # dim rails into the dark: infrared continues
    for x in (-3.0, 3.0):
        for a in range(750, 1400, 12):
            p1, p2 = proj(x, 0, a), proj(x, 0, a + 6)
            if p1 and p2 and gate_a > 0: gd.line([p1, p2], fill=(150, 60, 50, int(150 * gate_a / 255)), width=2)
    gl = g.filter(ImageFilter.GaussianBlur(10))
    im = Image.alpha_composite(Image.alpha_composite(im, gl), g)
    dr = ImageDraw.Draw(o)
    gp = proj(hw, 22, 750)
    if gate_label and gp and 0 < gp[0] < W - 300 and 0 < gp[1] < H:
      gx, gy = gp
      dr.text((gx + 120, gy + 10), '750 nm · 人眼到此为止，往前是红外', font=ImageFont.truetype(CJK_R, 28), fill=(255, 180, 160, 230), anchor='lm')
    up = proj(hw, 0, 380)
    if uv_label and up:
      ux, uy = up
      dr.text((min(ux, W - 260) + 20, min(uy, H - 40) - 30), '紫外在身后 ↓', font=ImageFont.truetype(CJK_R, 24), fill=(150, 120, 230, 200), anchor='lm')
    dr.text((W - 80, 70), '第七观测站 · E006', font=ImageFont.truetype(CJK_R, 30), fill=(200, 208, 218, 150), anchor='ra')
    if title: dr.text((120, H - 175), title, font=ImageFont.truetype(CJK_B, 64), fill=(242, 245, 248, 240))
    if sub: dr.text((124, H - 85), sub, font=ImageFont.truetype(CJK_R, 32), fill=(200, 208, 218, 220))
    out_im = Image.alpha_composite(im, o).convert('RGB')
    if save: out_im.save(path, quality=93)
    return out_im

if __name__ == '__main__':
    render(f'{D}/kf_street_A.jpg')
