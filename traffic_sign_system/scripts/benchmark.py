"""推理性能基准测试。

测量维度
--------
1. 单图 ``predict`` 耗时（关闭/开启缓存两种状态各 100 次取平均）。
2. 批量 ``predict_batch`` 在 batch_size = {1, 10, 50, 100} 下的耗时，
   与 ``predict`` 循环对比加速比。
3. ``VideoRecognizer`` 在本地视频上的 FPS（skip_frames = 0/1/2）。
4. （可选）缓存命中率（视频相邻帧去重）。

产出
----
- 控制台打印汇总表格
- ``models/artifacts/benchmark.json`` 全量明细
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import time
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np

from traffic_sign_system.config.settings import MODEL_ARTIFACTS_DIR
from traffic_sign_system.recognition.predictor import Predictor
from traffic_sign_system.recognition.video_recognizer import VideoRecognizer

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 计时工具
# ─────────────────────────────────────────────────────────────────────────────
def _time_call(fn, repeat: int) -> tuple[float, float, list[float]]:
    """调用 fn repeat 次，返回 (mean_ms, total_seconds, per_call_ms_list)。"""
    samples: list[float] = []
    started = time.perf_counter()
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1000.0)
    elapsed = time.perf_counter() - started
    return float(np.mean(samples)), elapsed, samples


def _synth_image(size: tuple[int, int] = (128, 128), seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.integers(0, 256, size=(size[1], size[0], 3), dtype=np.uint8))


# ─────────────────────────────────────────────────────────────────────────────
# 基准测试场景
# ─────────────────────────────────────────────────────────────────────────────
def bench_single(predictor: Predictor, repeats: int) -> dict[str, Any]:
    """单图 predict 耗时：冷热缓存对比。"""
    # 用 100 张不同图（关缓存）vs 同一张图（开缓存）测两种状态
    predictor.clear_cache()
    predictor.use_cache = False
    unique_imgs = [_synth_image(seed=i) for i in range(repeats)]
    cold_mean_ms, cold_total, _ = _time_call(
        lambda: [predictor.predict(im) for im in unique_imgs],
        1,  # 已经包含 repeats 张
    )
    # 等价写法：跑 repeats 次，但每次 predict 都用新图（与上面重复）
    cold_per_call = cold_mean_ms / repeats  # 单张平均

    # 热缓存：同一张图反复 predict
    predictor.use_cache = True
    predictor.cache_hits = predictor.cache_misses = 0
    same_img = _synth_image(seed=999)
    # 第一次 warm-up 触发 miss
    predictor.predict(same_img)
    warm_mean_ms, _, _ = _time_call(
        lambda: predictor.predict(same_img),
        repeats,
    )
    cache_stats = predictor.cache_stats()
    predictor.use_cache = True

    return {
        "repeats": repeats,
        "single_no_cache_ms_per_call": round(cold_per_call, 3),
        "single_no_cache_total_s": round(cold_total, 3),
        "single_warm_cache_ms_per_call": round(warm_mean_ms, 3),
        "speedup_with_cache": round(cold_per_call / max(warm_mean_ms, 1e-9), 2),
        "cache": cache_stats,
    }


def bench_batch(predictor: Predictor, batch_sizes: list[int], repeats: int) -> dict[str, Any]:
    """批量 predict_batch 与 循环 predict 对比加速比。"""
    rows: dict[str, Any] = {}
    for bs in batch_sizes:
        images = [_synth_image(seed=10_000 + i) for i in range(bs)]
        # 批量调用
        batch_mean_ms, batch_total, _ = _time_call(
            lambda: predictor.predict_batch(images),
            repeats,
        )
        # 循环 predict（等价对照）
        loop_mean_ms, _, _ = _time_call(
            lambda: [predictor.predict(im) for im in images],
            repeats,
        )
        rows[str(bs)] = {
            "batch_size": bs,
            "predict_batch_ms_per_call": round(batch_mean_ms, 3),
            "predict_batch_total_s": round(batch_total, 3),
            "predict_loop_ms_per_call": round(loop_mean_ms, 3),
            "speedup": round(loop_mean_ms / max(batch_mean_ms, 1e-9), 2),
        }
    return rows


def _build_synth_video(
    out_path: Path,
    *,
    width: int = 320,
    height: int = 240,
    n_frames: int = 60,
    fps: float = 25.0,
    seed: int = 7,
) -> Path:
    """生成一段测试视频（OpenCV VideoWriter）。"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
    rng = np.random.default_rng(seed)
    for i in range(n_frames):
        # 相邻帧差异很小 → 缓存/帧间跳过效果最好
        frame = (rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8))
        if i > 0:
            # 在前几帧基础上加少量噪声，让相邻帧高度相似但不完全相同
            prev = writer.get(cv2.CAP_PROP_POS_FRAMES)
            frame = (frame.astype(np.int16) + i * 2).clip(0, 255).astype(np.uint8)
        writer.write(frame)
    writer.release()
    return out_path


def bench_video(
    predictor: Predictor,
    *,
    video_path: Path | None,
    width: int,
    height: int,
    n_frames: int,
    fps: float,
    skip_options: list[int],
    repeats: int = 1,
) -> dict[str, Any]:
    """对比不同 skip_frames 下的 FPS。"""
    if video_path is None or not Path(video_path).exists():
        video_path = _build_synth_video(
            MODEL_ARTIFACTS_DIR / "_benchmark_synthetic.mp4",
            width=width, height=height, n_frames=n_frames, fps=fps,
        )
    video_path = Path(video_path)

    results: dict[str, Any] = {}
    for skip in skip_options:
        fps_samples: list[float] = []
        reuse_samples: list[float] = []
        cache_hit_samples: list[float] = []
        last_n = 0
        last_total = 0.0
        for _ in range(max(1, repeats)):
            predictor.clear_cache()
            recognizer = VideoRecognizer(predictor, str(video_path), skip_frames=skip)
            if not recognizer.open():
                raise RuntimeError(f"无法打开视频：{video_path}")
            started = time.perf_counter()
            while True:
                ok, _, _ = recognizer.read()
                if not ok:
                    break
            total = time.perf_counter() - started
            n = recognizer.frame_index
            last_n, last_total = n, total
            if n == 0:
                recognizer.release()
                continue
            fps_samples.append(n / total)
            stats = recognizer.skip_stats()
            reuse_samples.append(stats["reuse_rate"])
            cache_hit_samples.append(_safe_hit_rate(predictor))
            recognizer.release()
        results[f"skip_{skip}"] = {
            "frames": last_n,
            "wall_seconds": round(last_total, 3),
            "fps_mean": round(float(np.mean(fps_samples)), 2) if fps_samples else 0.0,
            "fps_max": round(float(np.max(fps_samples)), 2) if fps_samples else 0.0,
            "reuse_rate_mean": round(float(np.mean(reuse_samples)), 3) if reuse_samples else 0.0,
            "cache_hit_rate_mean": round(float(np.mean(cache_hit_samples)), 3) if cache_hit_samples else 0.0,
        }
    results["video_path"] = str(video_path)
    return results


def _safe_hit_rate(predictor: Predictor) -> float:
    """安全读取 hit_rate（避免 cache 字段缺失）。"""
    try:
        return float(predictor.cache_stats().get("hit_rate", 0.0))
    except Exception:  # noqa: BLE001
        return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def run(args: argparse.Namespace) -> dict[str, Any]:
    predictor = Predictor(
        args.bundle,
        use_cache=True,
        cache_maxsize=args.cache_maxsize,
    )
    logger.info("已加载 bundle: %s", args.bundle)
    logger.info("feature_dim=%d, mode=%s", predictor.feature_dim, predictor.feature_config.get("mode"))

    # 1. 单图
    single = bench_single(predictor, repeats=args.repeats_single)

    # 2. 批量
    batch = bench_batch(predictor, batch_sizes=args.batch_sizes, repeats=args.repeats_batch)

    # 3. 视频
    video = bench_video(
        predictor,
        video_path=Path(args.video) if args.video else None,
        width=args.video_width,
        height=args.video_height,
        n_frames=args.video_frames,
        fps=args.video_fps,
        skip_options=args.skip_options,
        repeats=args.repeats_video,
    )

    summary = {
        "bundle": str(args.bundle),
        "feature_dim": predictor.feature_dim,
        "feature_mode": predictor.feature_config.get("mode"),
        "cache_maxsize": args.cache_maxsize,
        "single": single,
        "batch": batch,
        "video": video,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("benchmark.json 已写入：%s", out_path)

    # 控制台汇总
    print("\n========== BENCHMARK SUMMARY ==========")
    print(f"feature_dim={predictor.feature_dim}  mode={predictor.feature_config.get('mode')}")
    print(f"\n[1] Single predict")
    print(f"  cold (no cache)   : {single['single_no_cache_ms_per_call']:.2f} ms/call")
    print(f"  warm (with cache) : {single['single_warm_cache_ms_per_call']:.2f} ms/call")
    print(f"  speedup           : {single['speedup_with_cache']}x")
    print(f"  cache hit_rate    : {single['cache']['hit_rate']:.3f} ({single['cache']['hits']} hits / {single['cache']['misses']} misses)")
    print(f"\n[2] Batch predict")
    for bs, row in batch.items():
        print(f"  batch={bs:>4}  loop={row['predict_loop_ms_per_call']:7.2f} ms  "
              f"batch={row['predict_batch_ms_per_call']:7.2f} ms  speedup={row['speedup']}x")
    print(f"\n[3] Video FPS")
    for key, row in video.items():
        if key == "video_path":
            continue
        print(f"  {key:>8}  fps_mean={row['fps_mean']:5.1f}  fps_max={row['fps_max']:5.1f}  "
              f"reuse_rate={row['reuse_rate_mean']:.2f}  cache_hit={row['cache_hit_rate_mean']:.2f}")
    print(f"\n→ benchmark.json : {out_path}")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="推理性能基准测试（单图/批量/视频）")
    parser.add_argument("--bundle", type=Path, required=True, help=".joblib 模型包路径")
    parser.add_argument(
        "--output",
        type=Path,
        default=MODEL_ARTIFACTS_DIR / "benchmark.json",
        help="输出 JSON 路径",
    )
    parser.add_argument("--cache-maxsize", type=int, default=512)
    parser.add_argument("--repeats-single", type=int, default=100)
    parser.add_argument(
        "--batch-sizes", type=int, nargs="+", default=[1, 10, 50, 100],
    )
    parser.add_argument("--repeats-batch", type=int, default=5)
    parser.add_argument(
        "--video", type=Path, default=None,
        help="可选的视频文件；不提供则生成合成视频",
    )
    parser.add_argument("--video-width", type=int, default=320)
    parser.add_argument("--video-height", type=int, default=240)
    parser.add_argument("--video-frames", type=int, default=60)
    parser.add_argument("--video-fps", type=float, default=25.0)
    parser.add_argument(
        "--skip-options", type=int, nargs="+", default=[0, 1, 2],
        help="VideoRecognizer 的 skip_frames 取值列表",
    )
    parser.add_argument("--repeats-video", type=int, default=1)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    args = build_parser().parse_args(argv)
    if args.repeats_single < 1:
        raise ValueError("--repeats-single 必须 >= 1")
    if any(b < 1 for b in args.batch_sizes):
        raise ValueError("--batch-sizes 中所有值必须 >= 1")
    run(args)


if __name__ == "__main__":
    main()