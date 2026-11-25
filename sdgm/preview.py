"""Reproducible visual evidence. Edge/phase outputs are screening proxies."""

from pathlib import Path

import cv2
import numpy as np
import rasterio
from PIL import Image, ImageDraw, ImageOps

from .io import load_datum, read_flo, save_csv, save_json


def stretch(image):
    image = np.asarray(image)
    finite = image[np.isfinite(image)]
    if not finite.size:
        return np.zeros(image.shape, np.uint8)
    lo, hi = np.percentile(finite, [1, 99])
    if hi <= lo:
        return np.zeros(image.shape, np.uint8)
    return (np.clip((image.astype(float) - lo) / (hi - lo), 0, 1) * 255).astype(
        np.uint8
    )


def rgb(image):
    image = np.asarray(image)
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=2)
    return image[..., :3].astype(np.uint8)


def panel(image, label, size=(512, 512)):
    im = ImageOps.contain(Image.fromarray(rgb(image)), size)
    result = Image.new("RGB", (size[0], size[1] + 36), "white")
    result.paste(im, ((size[0] - im.width) // 2, 36 + (size[1] - im.height) // 2))
    ImageDraw.Draw(result).text((8, 10), label, fill="black")
    return result


def montage(panels, columns, path):
    width = max(x.width for x in panels)
    height = max(x.height for x in panels)
    result = Image.new(
        "RGB",
        (width * columns, height * ((len(panels) + columns - 1) // columns)),
        "#e0e0e0",
    )
    for i, im in enumerate(panels):
        result.paste(im, ((i % columns) * width, (i // columns) * height))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    result.save(path)


def structural(image):
    image = stretch(image).astype(np.float32) / 255
    image = cv2.GaussianBlur(image, (0, 0), 1.2)
    return cv2.magnitude(
        cv2.Sobel(image, cv2.CV_32F, 1, 0), cv2.Sobel(image, cv2.CV_32F, 0, 1)
    )


def edge_overlay(optical, sar):
    first, second = structural(optical), structural(sar)
    first = first > np.percentile(first, 85)
    second = second > np.percentile(second, 85)
    out = np.zeros((*first.shape, 3), np.uint8)
    out[..., 0] = first * 255
    out[..., 1] = second * 255
    return out


def phase_shift(optical, sar):
    first, second = structural(optical), structural(sar)
    window = cv2.createHanningWindow((first.shape[1], first.shape[0]), cv2.CV_32F)
    shift, response = cv2.phaseCorrelate(first, second, window)
    return dict(dx=float(shift[0]), dy=float(shift[1]), response=float(response))


def select_whu(records, count, seed):
    """Spread selected patch IDs across both provided splits."""
    rng = np.random.default_rng(seed)
    selected = []
    splits = sorted({r["split"] for r in records})
    for index, split in enumerate(splits):
        pool = [r for r in records if r["split"] == split and r["status"] == "ok"]
        quota = count // len(splits) + int(index < count % len(splits))
        for chunk in np.array_split(np.arange(len(pool)), min(quota, len(pool))):
            if len(chunk):
                selected.append(pool[int(rng.choice(chunk))])
    return selected


def whu_diagnostics(records, output, count=12, seed=20261125):
    output = Path(output) / "whu" / "diagnostics"
    output.mkdir(parents=True, exist_ok=True)
    selected = select_whu(records, count, seed)
    proxies = []
    overview = []
    previews = []
    for record in selected:
        paths = record["paths"]
        datum = load_datum(paths["datum"])
        optical = datum[0][..., 0]
        warped = datum[0][..., 1]
        base = datum[4]
        flow = read_flo(paths["flow"])
        yy, xx = np.indices(flow.shape[:2])
        recovered = cv2.remap(
            warped,
            (xx + flow[..., 0]).astype(np.float32),
            (yy + flow[..., 1]).astype(np.float32),
            cv2.INTER_LINEAR,
        )
        sample = record["id"].replace("/", "_")
        proxy = phase_shift(optical, base)
        proxy.update(
            id=record["id"],
            kind="gradient_phase_correlation_proxy",
            is_ground_truth=False,
        )
        proxies.append(proxy)
        local = []
        vector = rgb(stretch(optical)).copy()
        for row in range(3):
            for col in range(3):
                x0 = col * 170
                y0 = row * 170
                part = phase_shift(
                    optical[y0 : y0 + 170, x0 : x0 + 170],
                    base[y0 : y0 + 170, x0 : x0 + 170],
                )
                part.update(
                    id=record["id"],
                    kind="local_gradient_phase_correlation_proxy",
                    cell_row=row,
                    cell_col=col,
                    x0=x0,
                    y0=y0,
                    width=170,
                    height=170,
                    is_ground_truth=False,
                )
                local.append(part)
                center = (x0 + 85, y0 + 85)
                cv2.circle(vector, center, 4, (255, 255, 0), -1)
                if part["response"] > 0.05 and np.hypot(part["dx"], part["dy"]) < 40:
                    end = (round(center[0] + part["dx"]), round(center[1] + part["dy"]))
                    cv2.arrowedLine(vector, center, end, (255, 0, 0), 2, tipLength=0.2)
                cv2.putText(
                    vector,
                    f"r={part['response']:.2f}",
                    (x0 + 4, y0 + 18),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (255, 255, 0),
                    1,
                )
        proxies.extend(local)
        montage(
            [
                panel(stretch(optical), "Optical (datum[0][:,:,0])"),
                panel(stretch(base), "Base SAR (datum[4])"),
                panel(
                    edge_overlay(optical, base), "Base edges: optical red / SAR green"
                ),
                panel(stretch(warped), "Artificially warped SAR"),
                panel(
                    stretch(recovered),
                    "SAR recovered with forward flow (valid overlap only)",
                ),
                panel(
                    vector,
                    f"Shift proxies, global ({proxy['dx']:.1f},{proxy['dy']:.1f}), r={proxy['response']:.2f}",
                ),
            ],
            3,
            output / f"{sample}_diagnostic.png",
        )
        montage(
            [
                panel(stretch(optical), record["id"] + " optical"),
                panel(stretch(base), "Base SAR, same pixel grid"),
            ],
            2,
            output / f"{sample}_base_pair.png",
        )
        overview.extend(
            [
                panel(stretch(optical), sample + " OPT", (256, 256)),
                panel(stretch(base), "Base SAR", (256, 256)),
            ]
        )
        previews.append(
            dict(
                id=record["id"],
                diagnostic=f"{sample}_diagnostic.png",
                base_pair=f"{sample}_base_pair.png",
                source_scene_id=None,
                base_alignment_status="requires_visual_checkpoints",
                interpretation_uncertainty="Grayscale optical, SAR scattering and possible temporal changes; source scene identity unavailable",
            )
        )
    montage(overview, 4, output / "overview.png")
    save_json(output / "selected_samples.json", previews)
    save_csv(output / "shift_proxies.csv", proxies)
    return selected


def other_previews(srif_records, spacenet_records, output, srif_root):
    output = Path(output)
    panels = []
    for record in [r for r in srif_records if r["status"] == "ok"][::33]:
        index = int(record["id"])
        a = cv2.imread(
            str(Path(srif_root) / f"pair{index}_1.jpg"), cv2.IMREAD_GRAYSCALE
        )
        b = cv2.imread(
            str(Path(srif_root) / f"pair{index}_2.jpg"), cv2.IMREAD_GRAYSCALE
        )
        warped = cv2.warpAffine(a, np.array(record["matrix"]), (b.shape[1], b.shape[0]))
        panels.extend(
            [
                panel(a, f"SRIF {index} image 1", (320, 320)),
                panel(b, "Image 2", (320, 320)),
                panel(edge_overlay(warped, b), "Forward GT warp edges", (320, 320)),
            ]
        )
    if panels:
        montage(panels, 3, output / "srif" / "sample_preview.png")
    panels = []
    for record in spacenet_records:
        if record["status"] != "ok":
            continue
        for key in ["optical", "sar"]:
            with rasterio.open(record[key]["path"]) as r:
                scale = 640 / max(r.height, r.width)
                height, width = max(1, round(r.height * scale)), max(
                    1, round(r.width * scale)
                )
                array = r.read(out_shape=(r.count, height, width))
                image = np.moveaxis(array, 0, -1)
                if r.count == 1:
                    image = stretch(np.log1p(np.maximum(image[..., 0], 0)))
                else:
                    image = stretch(image)
                panels.append(panel(image, record["id"] + " " + key, (640, 640)))
    if panels:
        montage(panels, 2, output / "spacenet9" / "sample_preview.png")
