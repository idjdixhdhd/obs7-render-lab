#!/usr/bin/env python3
"""第七观测站 · 渲染管线测试片
竖屏 1080x1920 @30fps, 6 秒: 粒子流场 -> 汇成「第七观测站」字样。
目的: 验证 GitHub Actions 渲染环境 (算力/字体/编码) 与文字锐度。
"""
import os, sys, time, platform, subprocess
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

W, H, FPS, DUR = 1080, 1920, 30, 6
N_FRAMES = FPS * DUR
OUT_DIR = "out"
os.makedirs(OUT_DIR, exist_ok=True)

FONT_CANDIDATES = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Bold.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
]

def load_font(size):
    for p in FONT_CANDIDATES:
        if os.path.exists(p):
            return ImageFont.truetype(p, size), p
    return ImageFont.load_default(), "default"

def env_report():
    lines = []
    lines.append(f"platform: {platform.platform()}")
    lines.append(f"python: {sys.version.split()[0]}")
    lines.append(f"cpu_count: {os.cpu_count()}")
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal"):
                    lines.append("mem_total: " + line.split(":")[1].strip())
                    break
    except Exception:
        pass
    lines.append(f"numpy: {np.__version__}")
    import PIL
    lines.append(f"pillow: {PIL.__version__}")
    try:
        v = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True).stdout.splitlines()[0]
        lines.append(v)
    except Exception as e:
        lines.append(f"ffmpeg: unavailable ({e})")
    return "\n".join(lines)

def main():
    t0 = time.time()
    rng = np.random.default_rng(7)

    # ---------- 文字目标点 ----------
    font_main, font_path = load_font(196)
    print("font:", font_path, flush=True)
    mask_img = Image.new("L", (W, H), 0)
    md = ImageDraw.Draw(mask_img)
    text = "第七观测站"
    bbox = md.textbbox((0, 0), text, font=font_main)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    md.text(((W - tw) / 2 - bbox[0], H * 0.40 - th / 2 - bbox[1]), text, fill=255, font=font_main)
    mask = np.array(mask_img) > 0
    ys, xs = np.nonzero(mask)
    print("mask pixels:", len(xs), flush=True)

    N = 2600
    idx = rng.integers(0, len(xs), N)
    tx, ty = xs[idx].astype(np.float32), ys[idx].astype(np.float32)

    # ---------- 粒子 ----------
    px = rng.uniform(0, W, N).astype(np.float32)
    py = rng.uniform(0, H, N).astype(np.float32)
    phase = rng.uniform(0, 2 * np.pi, N).astype(np.float32)

    canvas = np.zeros((H, W), dtype=np.float32)
    # 背景: 深空渐变 + 微光星点
    grad = np.linspace(0.05, 0.16, H, dtype=np.float32)[:, None] * np.ones((1, W), dtype=np.float32)
    stars = (rng.random((H, W)) > 0.9993).astype(np.float32) * 0.35

    # ---------- ffmpeg 管道 ----------
    ff = subprocess.Popen(
        ["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
         "-r", str(FPS), "-i", "-", "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p",
         "-crf", "18", "-preset", "medium", os.path.join(OUT_DIR, "obs7_render_test.mp4")],
        stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    font_sub, _ = load_font(44)
    t_render0 = time.time()
    for f in range(N_FRAMES):
        t = f / FPS
        # 流场相位
        ang = 2.2 * np.sin(px * 0.0035 + t * 0.35 + phase) + 2.2 * np.cos(py * 0.0030 - t * 0.27)
        vx, vy = np.cos(ang) * 2.6, np.sin(ang) * 2.6
        px += vx; py += vy
        # 汇聚阶段: 3.0s 起 smoothstep 拉向目标点
        if t > 3.0:
            k = min(1.0, (t - 3.0) / 1.8)
            k = k * k * (3 - 2 * k)
            px += (tx - px) * 0.085 * k
            py += (ty - py) * 0.085 * k
            px += rng.normal(0, 0.35 * (1 - k) + 0.05, N).astype(np.float32)
            py += rng.normal(0, 0.35 * (1 - k) + 0.05, N).astype(np.float32)
        # 环绕边界
        px %= W; py %= H
        # 轨迹衰减 + 绘制
        canvas *= 0.86
        xi = np.clip(px.astype(np.int32), 0, W - 1)
        yi = np.clip(py.astype(np.int32), 0, H - 1)
        np.add.at(canvas, (yi, xi), 0.85)
        frame = np.clip(grad * 255 + stars * 255 + canvas[..., None] * np.array([190, 215, 255], dtype=np.float32), 0, 255)
        img = Image.fromarray(frame.astype(np.uint8))
        # 轻微辉光
        glow = img.filter(ImageFilter.GaussianBlur(6))
        img = Image.fromarray(np.clip(np.array(img, dtype=np.float32) + np.array(glow, dtype=np.float32) * 0.35, 0, 255).astype(np.uint8))
        # 5 秒后淡入底部字幕 (真实文字渲染, 非贴图缩放)
        if t > 5.0:
            a = min(1.0, (t - 5.0) / 0.7)
            overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
            od = ImageDraw.Draw(overlay)
            sub = "渲染管线测试 · GitHub Actions"
            sb = od.textbbox((0, 0), sub, font=font_sub)
            od.text(((W - (sb[2] - sb[0])) / 2 - sb[0], H * 0.80), sub,
                    fill=(200, 215, 235, int(255 * a)), font=font_sub)
            img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
        ff.stdin.write(img.tobytes())
        if f % 30 == 0:
            print(f"frame {f}/{N_FRAMES} elapsed {time.time()-t_render0:.1f}s", flush=True)
    ff.stdin.close(); ff.wait()
    t_total = time.time() - t0

    report = env_report() + f"\nfont_used: {font_path}\nrender_total_seconds: {t_total:.1f}\nframes: {N_FRAMES} @ {W}x{H} {FPS}fps\n"
    with open(os.path.join(OUT_DIR, "env_report.txt"), "w") as f:
        f.write(report)
    print(report, flush=True)

if __name__ == "__main__":
    main()
