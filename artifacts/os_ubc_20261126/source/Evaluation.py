# -*- coding: utf-8 -*-
"""
GCP-based rectification with GDAL (no DEM/RPC).
Inputs:
  - ref_tif:   reference GeoTIFF (defines world CRS, resolution, bounds)
  - mov_tif:   moving GeoTIFF to be rectified
  - matches:   text file lines: x1 y1 x2 y2  (x1,y1 in ref, x2,y2 in mov)  <-- ALL used
  - gtruth:    ground truth points for evaluation
Outputs:
  - out_tif:   rectified GeoTIFF aligned to the reference raster grid
  - eval_csv:  evaluation report in CSV format with pixel-based residuals only
Notes:
  - Coordinates convention: x=column, y=row (0-based). If your txt uses
    pixel centers or 1-based indexing, set offset_mode accordingly.
  - Transform model:
      * polynomial order=1 (affine), 2, 3
      * or TPS (thin plate spline)
"""

import os
import math
import numpy as np
from osgeo import gdal

# ---------- User config ----------
ref_tif   = r"../OSDataset2.0/Scene-level Subset/optical.tif"
mov_tif   = r"../OSDataset2.0/Scene-level Subset/sar.tif"
matches   = r"../OSDataset2.0/Scene-level Subset/match_result.txt"  # x1 y1 x2 y2 (ref -> mov)
gtruth    = r"../OSDataset2.0/Scene-level Subset/GT.txt" # Ground truth (for evaluation)
out_tif   = r"../OSDataset2.0/Scene-level Subset/registered_norm_Pol.tif"
eval_csv  = r"../OSDataset2.0/Scene-level Subset/eval_report.csv"  # Output Excel file

# Transform model
use_tps    = False     # True: Thin Plate Spline; False: Polynomial
poly_order = 1         # 1=affine, 2 or 3 (ignored if use_tps=True)

# resample
resample_alg = gdal.GRA_Bilinear  # NearestNeighbour / Bilinear / Cubic / Lanczos

# offset
offset_mode = "none"   # "none" | "+0.5" | "-0.5"

# ==========================================

def transform_points_robust(transformer, pts):
    res = transformer.TransformPoints(False, pts)

    # Helper to extract (ok, pts_out) from two items in any weird packaging
    def _normalize2(a, b):
        # a: ok / [ok] / points
        # b: ok / [ok] / points
        def is_ok_val(x):
            return isinstance(x, (int, bool))
        def list_is_ok(x):
            return isinstance(x, list) and len(x) == 1 and isinstance(x[0], (int, bool))
        def is_pts(x):
            return isinstance(x, list) and len(x) >= 1 and isinstance(x[0], (tuple, list)) and len(x[0]) >= 2

        # (ok, pts) or ([ok], pts)
        if (is_ok_val(a) or list_is_ok(a)) and is_pts(b):
            ok = a if is_ok_val(a) else a[0]
            return int(ok), b
        # (pts, ok) or (pts, [ok])
        if is_pts(a) and (is_ok_val(b) or list_is_ok(b)):
            ok = b if is_ok_val(b) else b[0]
            return int(ok), a
        return None

    # --- Cases ---
    if isinstance(res, tuple) and len(res) == 2:
        norm = _normalize2(res[0], res[1])
        if norm is not None:
            return norm
        a, b = res
        if isinstance(a, list) and len(a) == 1 and isinstance(a[0], list):
            norm = _normalize2(a[0], b)
            if norm is not None:
                return norm
        if isinstance(b, list) and len(b) == 1 and isinstance(b[0], list):
            norm = _normalize2(a, b[0])
            if norm is not None:
                return norm


    if isinstance(res, list):
        return 1, res


    if isinstance(res, (int, bool)):
        return int(res), pts


    if res is None:
        return 1, pts


    raise RuntimeError(f"Unexpected return from TransformPoints: {type(res)} -> {res!r}")

def read_xyxy(path):
    arr = []
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln or ln.startswith("#"):
                continue
            parts = ln.replace(",", " ").split()
            if len(parts) < 4:
                continue
            x1, y1, x2, y2 = map(float, parts[:4])
            arr.append((x1, y1, x2, y2))
    if not arr:
        raise RuntimeError("No valid pairs in: " + path)
    return np.array(arr, dtype=float)

def apply_offset(x, y, mode):
    if mode == "none":
        return x, y
    if mode == "+0.5":
        return x + 0.5, y + 0.5
    if mode == "-0.5":
        return x - 0.5, y - 0.5
    raise ValueError("Unknown offset_mode: " + mode)

def pixel_to_world(gt, x, y):
    # works for rotated geotransform
    X = gt[0] + x * gt[1] + y * gt[2]
    Y = gt[3] + x * gt[4] + y * gt[5]
    return X, Y

def bounds_from_gt(ds):
    gt = ds.GetGeoTransform(can_return_null=True)
    w = ds.RasterXSize; h = ds.RasterYSize
    XY = [pixel_to_world(gt, px, py) for (px,py) in [(0,0),(w,0),(w,h),(0,h)]]
    xs = [p[0] for p in XY]; ys = [p[1] for p in XY]
    return (min(xs), min(ys), max(xs), max(ys))

def open_raster_strict(path):
    gdal.UseExceptions()
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    ds = gdal.OpenEx(path, gdal.OF_RASTER,
                     allowed_drivers=["GTiff","COG","JP2OpenJPEG","HFA","EHdr","ENVI","BMP","PNG","JPEG","VRT"])
    if ds is None:
        raise RuntimeError("Failed to open raster: " + path)
    return ds

def invert_gt(gt):
    """Return inverse geotransform as a 6-tuple, compatible across GDAL versions."""
    res = gdal.InvGeoTransform(gt)

    if isinstance(res, tuple) and len(res) == 2 and isinstance(res[1], (tuple, list)):
        ok, inv = res
        if not ok:
            raise RuntimeError("Cannot invert GeoTransform.")
        return tuple(inv)

    if isinstance(res, (tuple, list)) and len(res) == 6:
        return tuple(res)
    raise RuntimeError(f"Unexpected return from InvGeoTransform: {res!r}")

def apply_gt(gt, X, Y):
    x = gt[0] + gt[1]*X + gt[2]*Y
    y = gt[3] + gt[4]*X + gt[5]*Y
    return x, y

def metric_scale_from_gt(gt):
    # approximate pixel size for rotated GT
    if abs(gt[2]) < 1e-12 and abs(gt[4]) < 1e-12:
        return abs(gt[1]), abs(gt[5])
    else:
        return math.hypot(gt[1], gt[4]), math.hypot(gt[2], gt[5])

def main():
    # ---------- open datasets ----------
    ref_ds = open_raster_strict(ref_tif)
    mov_ds = open_raster_strict(mov_tif)

    ref_gt = ref_ds.GetGeoTransform(can_return_null=True)
    if ref_gt is None:
        raise RuntimeError("Reference has no GeoTransform.")
    ref_srs = ref_ds.GetProjection()
    if not ref_srs:
        raise RuntimeError("Reference has no projection WKT.")

    # target grid aligned to reference
    out_bounds = bounds_from_gt(ref_ds)
    x_res, y_res = metric_scale_from_gt(ref_gt)

    # ---------- read pairs ----------
    MS = read_xyxy(matches)
    GT = read_xyxy(gtruth)

    # split & offset
    x1_m, y1_m = MS[:,0], MS[:,1]  # ref pixels
    x2_m, y2_m = MS[:,2], MS[:,3]  # mov pixels
    x1_m, y1_m = apply_offset(x1_m, y1_m, offset_mode)
    x2_m, y2_m = apply_offset(x2_m, y2_m, offset_mode)

    # ref pixel -> world
    worldX = np.empty_like(x1_m); worldY = np.empty_like(y1_m)
    for i in range(len(x1_m)):
        X, Y = pixel_to_world(ref_gt, x1_m[i], y1_m[i])
        worldX[i] = X; worldY[i] = Y

    # Build GCPs (mov pixel -> world)
    gcps = [gdal.GCP(float(worldX[i]), float(worldY[i]), 0.0, float(x2_m[i]), float(y2_m[i]))
            for i in range(len(x2_m))]

    # Wrap moving with these GCPs into a VRT
    vrt_path = "/vsimem/mov_with_gcps.vrt"
    vrt = gdal.Translate(vrt_path, mov_ds, format="VRT", GCPs=gcps, outputSRS=ref_srs)
    if vrt is None:
        raise RuntimeError("Failed to create VRT with GCPs")

    # ---------- Warp to reference grid ----------
    if use_tps:
        warp_opts = gdal.WarpOptions(
            format="GTiff", dstSRS=ref_srs,
            outputBounds=out_bounds, xRes=x_res, yRes=y_res, targetAlignedPixels=True,
            resampleAlg=resample_alg, tps=True
        )
    else:
        warp_opts = gdal.WarpOptions(
            format="GTiff", dstSRS=ref_srs,
            outputBounds=out_bounds, xRes=x_res, yRes=y_res, targetAlignedPixels=True,
            resampleAlg=resample_alg, polynomialOrder=int(poly_order)
        )
    out_ds = gdal.Warp(out_tif, vrt, options=warp_opts)
    if out_ds is None:
        raise RuntimeError("gdal.Warp failed.")
    out_ds = None
    print("[OK] rectified image written:", out_tif)

    # ---------- Evaluation on GT (PIXEL UNITS ONLY) ----------
    # Prepare a GCP transformer for pixel(line)->world using SAME model as warp
    if use_tps:
        transformer = gdal.Transformer(vrt, None, ["METHOD=GCP_TPS"])
    else:
        transformer = gdal.Transformer(vrt, None, ["METHOD=GCP_POLYNOMIAL", f"ORDER={int(poly_order)}"])

    # Invert reference geotransform to map world->ref pixel
    inv_ref_gt = invert_gt(ref_gt)

    # Read GT and offset
    x1_g, y1_g = GT[:,0], GT[:,1]   # ref pix (truth)
    x2_g, y2_g = GT[:,2], GT[:,3]   # mov pix (to be transformed)
    x1_g, y1_g = apply_offset(x1_g, y1_g, offset_mode)
    x2_g, y2_g = apply_offset(x2_g, y2_g, offset_mode)

    pred_ref = np.zeros((len(x2_g), 2), dtype=float)
    for i in range(len(x2_g)):
        # src pixel/line -> world via GCP transformer
        pts = [(float(x2_g[i]), float(y2_g[i]), 0.0)]
        ok, out_pts = transform_points_robust(transformer, pts)
        if not ok:
            raise RuntimeError(f"GCP transform failed at idx {i}")
        Xw, Yw = float(out_pts[0][0]), float(out_pts[0][1])
        px, py = gdal.ApplyGeoTransform(inv_ref_gt, Xw, Yw)
        pred_ref[i,0] = px
        pred_ref[i,1] = py

    # pixel residuals
    dpx = pred_ref[:,0] - x1_g
    dpy = pred_ref[:,1] - y1_g
    d_pix = np.hypot(dpx, dpy)

    def stats(v):
        return dict(
            count = int(v.size),
            mean  = float(np.mean(v)),
            median= float(np.median(v)),
            std   = float(np.std(v, ddof=0)),
            rmse  = float(np.sqrt(np.mean(v**2))),
            p90   = float(np.percentile(v, 90)),
            max   = float(np.max(v))
        )

    S_pix = stats(d_pix)
    print("=== Evaluation (PIXEL UNITS ONLY) ===")
    print("Pixel RMSE = %.6f px | mean=%.6f | median=%.6f | std=%.6f | P90=%.6f | max=%.6f"
          % (S_pix["rmse"], S_pix["mean"], S_pix["median"], S_pix["std"], S_pix["p90"], S_pix["max"]))

    # ---------- write per-point CSV (PIXEL UNITS ONLY) ----------
    # Columns: id, ref_x, ref_y, mov_x, mov_y, pred_x, pred_y, dpx, dpy, d_pix
    with open(eval_csv, "w", encoding="utf-8") as f:
        f.write("id,ref_x,ref_y,mov_x,mov_y,pred_x,pred_y,dpx,dpy,d_pix\n")
        for i in range(len(x1_g)):
            f.write(("%d,%.10f,%.10f,%.10f,%.10f,%.10f,%.10f,%.10f,%.10f,%.10f\n") %
                    (i+1, x1_g[i], y1_g[i], x2_g[i], y2_g[i],
                     pred_ref[i,0], pred_ref[i,1],
                     dpx[i], dpy[i], d_pix[i]))
    print("[OK] eval CSV written :", eval_csv)

if __name__ == "__main__":
    main()