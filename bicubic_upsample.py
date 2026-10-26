"""
Bicubic 插值放大脚本
将 LR (128x128) 图像放大到 HR (512x512) 并保存结果
"""

import os
import argparse
from pathlib import Path
from PIL import Image


def bicubic_upsample(
    lr_dir: str,
    output_dir: str,
    scale: int = 4,
    target_size: tuple = (512, 512),
):
    """
    对 LR 图像目录中的所有图像做 Bicubic 插值放大并保存。

    Args:
        lr_dir     : LR 图像所在目录
        output_dir : 放大后图像的保存目录
        scale      : 放大倍数（与 target_size 二选一，优先使用 target_size）
        target_size: 目标尺寸 (W, H)，设为 None 则依据 scale 自动计算
    """
    lr_dir = Path(lr_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 支持的图像后缀
    extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

    image_paths = sorted([p for p in lr_dir.iterdir() if p.suffix.lower() in extensions])

    if not image_paths:
        print(f"[警告] 在 {lr_dir} 中未找到任何图像文件。")
        return

    print(f"共找到 {len(image_paths)} 张 LR 图像，开始处理...\n")

    for idx, img_path in enumerate(image_paths, 1):
        img = Image.open(img_path).convert("RGB")
        w, h = img.size

        # 计算目标尺寸
        if target_size is not None:
            out_w, out_h = target_size
        else:
            out_w, out_h = w * scale, h * scale

        # Bicubic 插值放大
        img_bicubic = img.resize((out_w, out_h), Image.BICUBIC)

        # 保存，保留原始文件名
        save_path = output_dir / img_path.name
        img_bicubic.save(save_path)

        print(f"[{idx:>4}/{len(image_paths)}] {img_path.name}  "
              f"{w}x{h}  -->  {out_w}x{out_h}  =>  {save_path}")

    print(f"\n全部完成！放大后的图像已保存至: {output_dir.resolve()}")


def verify_pairs(lr_dir: str, hr_dir: str, bicubic_dir: str):
    """
    可选验证：对比 HR 与 Bicubic 放大图的尺寸是否一致，
    并计算简单的像素均值差异（无需安装额外库）。
    """
    lr_dir = Path(lr_dir)
    hr_dir = Path(hr_dir)
    bicubic_dir = Path(bicubic_dir)

    extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
    files = sorted([p.name for p in bicubic_dir.iterdir() if p.suffix.lower() in extensions])

    if not files:
        print("[验证] Bicubic 目录为空，跳过验证。")
        return

    print(f"\n{'='*60}")
    print(f"{'文件名':<30} {'HR尺寸':<12} {'Bicubic尺寸':<12} {'尺寸匹配'}")
    print(f"{'='*60}")

    match_count = 0
    for fname in files:
        hr_path = hr_dir / fname
        bc_path = bicubic_dir / fname
        if not hr_path.exists():
            print(f"{fname:<30} HR文件不存在，跳过")
            continue

        hr_img = Image.open(hr_path)
        bc_img = Image.open(bc_path)
        match = "✓" if hr_img.size == bc_img.size else "✗"
        if hr_img.size == bc_img.size:
            match_count += 1
        print(f"{fname:<30} {str(hr_img.size):<12} {str(bc_img.size):<12} {match}")

    print(f"{'='*60}")
    print(f"尺寸匹配: {match_count}/{len(files)}\n")


# ─────────────────────────────────────────────
# 命令行入口
# ─────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="Bicubic 插值将 LR 图像放大至 HR 尺寸"
    )
    parser.add_argument("--lr_dir",      type=str, required=True,
                        help="LR 图像目录路径")
    parser.add_argument("--output_dir",  type=str, required=True,
                        help="放大后图像的保存目录")
    parser.add_argument("--hr_dir",      type=str, default=None,
                        help="（可选）HR 图像目录，用于尺寸验证")
    parser.add_argument("--scale",       type=int, default=4,
                        help="放大倍数，默认 4（当 --target_size 未指定时生效）")
    parser.add_argument("--target_size", type=int, nargs=2, default=[512, 512],
                        metavar=("W", "H"),
                        help="目标尺寸，默认 512 512")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    bicubic_upsample(
        lr_dir=args.lr_dir,
        output_dir=args.output_dir,
        scale=args.scale,
        target_size=tuple(args.target_size) if args.target_size else None,
    )

    if args.hr_dir:
        verify_pairs(args.lr_dir, args.hr_dir, args.output_dir)