"""Conservative native-base screening, separate from synthetic label accuracy."""

import numpy as np


def checkpoint_measurement(point, roi):
    optical = np.asarray(point["optical_xy"], dtype=float)
    sar = np.asarray(point["sar_xy"], dtype=float)
    if (
        optical.shape != (2,)
        or sar.shape != (2,)
        or not np.isfinite([optical, sar]).all()
    ):
        raise ValueError("Checkpoint coordinates must be finite XY pairs")
    if not (
        (optical >= 0).all()
        and (sar >= 0).all()
        and (optical < [roi["width"], roi["height"]]).all()
        and (sar < [roi["width"], roi["height"]]).all()
    ):
        raise ValueError("Checkpoint outside native audit window")
    uo, us = float(point["optical_uncertainty_px"]), float(point["sar_uncertainty_px"])
    if not np.isfinite([uo, us]).all() or min(uo, us) < 0:
        raise ValueError("Invalid localization uncertainty")
    residual = sar - optical
    norm = float(np.linalg.norm(residual))
    origin = np.array([roi["x"], roi["y"]])
    return dict(
        **point,
        parent_image_id=roi["parent_image_id"],
        geo_group_id=roi["geo_group_id"],
        split=roi["split"],
        optical_original_xy=(origin + optical).tolist(),
        sar_original_xy=(origin + sar).tolist(),
        sar_minus_optical_xy=residual.tolist(),
        nominal_residual_px=norm,
        conservative_residual_plus_uncertainty_px=norm + uo + us,
        lower_possible_residual_px=max(0, norm - uo - us),
        use="audit_only_not_fit",
        independent_human_review=False,
    )


def region_admission(points, tolerance=8.0, fine_target=2.0):
    """Passing sparse point bounds alone must never admit an entire region."""
    coarse_points = [
        p for p in points if p["conservative_residual_plus_uncertainty_px"] <= tolerance
    ]
    return dict(
        coarse_allowed=False,
        fine_allowed=False,
        base_optical_to_sar_matrix=np.eye(3).tolist(),
        base_relation_status="unverified_nominal_identity_not_admitted",
        checkpoint_ids=[p["id"] for p in points],
        checkpoint_count=len(points),
        point_bounds_within_coarse_tolerance=len(coarse_points),
        coarse_tolerance_px=tolerance,
        fine_target_px=fine_target,
        region_spatial_coverage_verified=False,
        systematic_bias_assessed=False,
        reason="No region-level accuracy certification: sparse tentative points and/or residual-plus-uncertainty bound exceeds tolerance. Fine precision and systematic bias remain unverified.",
    )
