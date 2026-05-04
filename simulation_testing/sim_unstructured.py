import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Tuple


CELL_TYPES = np.array(["A", "B", "C", "D", "E"], dtype=object)


# Notes:
# - In the QUICHE paper's 2D unstructured simulation, one grid cell out of a g x g grid
#   is selected as the planted niche region. Its area fraction is 1 / g^2.
# - In this 3D version, one voxel out of a g x g x g grid is selected. Its volume fraction is 1 / g^3.
# - Therefore, g = 5 in 3D corresponds to a much smaller planted region (0.8%) than g = 5 in 2D (4%).
# - For a first 3D baseline that is roughly analogous to the paper's 2D g = 5 example,
#   g = 3 is a better starting point because 1 / 3^3 = 3.7%.
#
# Domain size:
# - We use domain_size = 300 as a practical 3D simulation domain that yields a reasonable
#   occupancy regime for local neighborhoods and keeps the simulation easy to debug.
# - This is not a direct "density match" to the original 2D simulation, since area density
#   and volume density are not directly comparable.


@dataclass
class SimConfig:
    n_patients: int = 20
    n_conditions: int = 2
    cells_per_patient: int = 5000
    domain_size: float = 300.0
    grid_size: int = 3
    prevalence: float = 1.0
    random_state: int = 0
    preserve_global_counts: bool = True


def make_patient_table(n_patients: int = 20, n_conditions: int = 2) -> pd.DataFrame:
    if n_conditions != 2:
        raise ValueError("This simulator currently expects exactly 2 conditions.")
    if n_patients % n_conditions != 0:
        raise ValueError("n_patients must be divisible by n_conditions for balanced groups.")

    n_per_condition = n_patients // n_conditions
    patient_ids = [f"P{i:02d}" for i in range(n_patients)]
    conditions = np.repeat(np.arange(n_conditions), n_per_condition)

    return pd.DataFrame({
        "patient_id": patient_ids,
        "condition": conditions,
    })


def assign_initial_cell_types(cells_per_patient: int, rng: np.random.Generator) -> np.ndarray:
    if cells_per_patient % len(CELL_TYPES) != 0:
        raise ValueError("cells_per_patient must be divisible by the number of cell types.")

    n_each = cells_per_patient // len(CELL_TYPES)
    labels = np.repeat(CELL_TYPES, n_each).astype(object)
    rng.shuffle(labels)
    return labels


def sample_coordinates_3d(
    cells_per_patient: int,
    domain_size: float,
    rng: np.random.Generator
) -> np.ndarray:
    return rng.uniform(0.0, domain_size, size=(cells_per_patient, 3))


def pick_niche_positive_patients(
    patient_df: pd.DataFrame,
    prevalence: float,
    rng: np.random.Generator
) -> pd.DataFrame:
    if not (0.0 <= prevalence <= 1.0):
        raise ValueError("prevalence must be between 0 and 1.")

    out = patient_df.copy()
    out["has_niche"] = False

    for cond in sorted(out["condition"].unique()):
        idx = out.index[out["condition"] == cond].to_numpy()
        n_pos = int(round(len(idx) * prevalence))
        if n_pos > 0:
            chosen = rng.choice(idx, size=n_pos, replace=False)
            out.loc[chosen, "has_niche"] = True

    return out


def voxel_bounds(
    grid_size: int,
    domain_size: float,
    voxel_idx: Tuple[int, int, int]
) -> Tuple[float, float, float, float, float, float]:
    step = domain_size / grid_size
    ix, iy, iz = voxel_idx

    x0, x1 = ix * step, (ix + 1) * step
    y0, y1 = iy * step, (iy + 1) * step
    z0, z1 = iz * step, (iz + 1) * step

    return x0, x1, y0, y1, z0, z1


def random_voxel(grid_size: int, rng: np.random.Generator) -> Tuple[int, int, int]:
    return tuple(rng.integers(0, grid_size, size=3).tolist())


def cells_in_voxel(
    coords: np.ndarray,
    bounds: Tuple[float, float, float, float, float, float]
) -> np.ndarray:
    x0, x1, y0, y1, z0, z1 = bounds
    mask = (
        (coords[:, 0] >= x0) & (coords[:, 0] < x1) &
        (coords[:, 1] >= y0) & (coords[:, 1] < y1) &
        (coords[:, 2] >= z0) & (coords[:, 2] < z1)
    )
    return np.where(mask)[0]


def niche_target_types(condition: int) -> np.ndarray:
    if condition == 0:
        return np.array(["A", "C", "E"], dtype=object)
    if condition == 1:
        return np.array(["B", "D"], dtype=object)
    raise ValueError("condition must be 0 or 1")


def value_counts_array(labels: np.ndarray) -> pd.Series:
    return pd.Series(labels, dtype="object").value_counts().reindex(CELL_TYPES, fill_value=0)


def plant_niche_without_preserving_counts(
    labels: np.ndarray,
    niche_idx: np.ndarray,
    targets: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    new_labels = labels.copy()
    if len(niche_idx) == 0:
        return new_labels

    new_labels[niche_idx] = rng.choice(targets, size=len(niche_idx), replace=True)
    return new_labels


def plant_niche_preserving_counts(
    labels: np.ndarray,
    niche_idx: np.ndarray,
    targets: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Relabel cells in the planted region using target cell types, then compensate outside
    the region so global cell-type counts remain unchanged.

    Important:
    - This preserves global abundance exactly.
    - The ground-truth niche remains a spatial-region label, not a "target-cell identity" label.
    """
    new_labels = labels.copy()
    if len(niche_idx) == 0:
        return new_labels

    orig_counts = value_counts_array(labels)

    niche_new = rng.choice(targets, size=len(niche_idx), replace=True)
    new_labels[niche_idx] = niche_new

    new_counts = value_counts_array(new_labels)
    diff = (new_counts - orig_counts).to_dict()

    niche_mask = np.zeros(len(labels), dtype=bool)
    niche_mask[niche_idx] = True
    outside_idx = np.where(~niche_mask)[0]

    surplus = {ct: max(0, diff[ct]) for ct in CELL_TYPES}
    deficit = {ct: max(0, -diff[ct]) for ct in CELL_TYPES}

    deficit_pool: List[str] = []
    for ct in CELL_TYPES:
        deficit_pool.extend([ct] * deficit[ct])

    deficit_pool = np.array(deficit_pool, dtype=object)
    rng.shuffle(deficit_pool)

    pool_ptr = 0
    for from_ct in CELL_TYPES:
        n_surplus = surplus[from_ct]
        if n_surplus == 0:
            continue

        candidate_idx = outside_idx[new_labels[outside_idx] == from_ct]
        if len(candidate_idx) < n_surplus:
            raise RuntimeError(
                f"Not enough outside-region cells of type {from_ct} to preserve counts."
            )

        chosen_from = rng.choice(candidate_idx, size=n_surplus, replace=False)
        assign_to = deficit_pool[pool_ptr: pool_ptr + n_surplus]
        if len(assign_to) != n_surplus:
            raise RuntimeError("Deficit bookkeeping failed during global count preservation.")

        new_labels[chosen_from] = assign_to
        pool_ptr += n_surplus

    final_counts = value_counts_array(new_labels)
    if not np.all(final_counts.values == orig_counts.values):
        raise RuntimeError("Global cell-type counts were not preserved.")

    return new_labels


def summarize_region_counts(labels: np.ndarray, idx: np.ndarray) -> Dict[str, int]:
    if len(idx) == 0:
        return {ct: 0 for ct in CELL_TYPES}
    counts = value_counts_array(labels[idx])
    return {ct: int(counts[ct]) for ct in CELL_TYPES}


def simulate_patient(
    patient_id: str,
    condition: int,
    has_niche: bool,
    cfg: SimConfig,
    rng: np.random.Generator,
) -> Tuple[pd.DataFrame, Dict]:
    coords = sample_coordinates_3d(cfg.cells_per_patient, cfg.domain_size, rng)
    labels_before = assign_initial_cell_types(cfg.cells_per_patient, rng)

    voxel_idx = random_voxel(cfg.grid_size, rng)
    bounds = voxel_bounds(cfg.grid_size, cfg.domain_size, voxel_idx)
    niche_idx = cells_in_voxel(coords, bounds)

    labels_after = labels_before.copy()
    gt_region_label = np.array(["background"] * cfg.cells_per_patient, dtype=object)

    if has_niche:
        targets = niche_target_types(condition)
        if cfg.preserve_global_counts:
            labels_after = plant_niche_preserving_counts(labels_before, niche_idx, targets, rng)
        else:
            labels_after = plant_niche_without_preserving_counts(labels_before, niche_idx, targets, rng)

        gt_region_label[niche_idx] = "ACE_region" if condition == 0 else "BD_region"

    niche_mask = np.zeros(cfg.cells_per_patient, dtype=bool)
    niche_mask[niche_idx] = True
    outside_idx = np.where(~niche_mask)[0]

    counts_before_total = summarize_region_counts(labels_before, np.arange(cfg.cells_per_patient))
    counts_after_total = summarize_region_counts(labels_after, np.arange(cfg.cells_per_patient))
    counts_after_inside = summarize_region_counts(labels_after, niche_idx)
    counts_after_outside = summarize_region_counts(labels_after, outside_idx)

    cell_df = pd.DataFrame({
        "patient_id": patient_id,
        "condition": condition,
        "cell_id": [f"{patient_id}_cell_{i}" for i in range(cfg.cells_per_patient)],
        "x": coords[:, 0],
        "y": coords[:, 1],
        "z": coords[:, 2],
        "cell_type": labels_after,
        "gt_region_label": gt_region_label,
        "in_niche_voxel": niche_mask,
    })

    voxel_fraction = 1.0 / (cfg.grid_size ** 3)
    realized_fraction = len(niche_idx) / cfg.cells_per_patient

    meta = {
        "patient_id": patient_id,
        "condition": int(condition),
        "has_niche": bool(has_niche),
        "grid_size": int(cfg.grid_size),
        "domain_size": float(cfg.domain_size),
        "voxel_ix": int(voxel_idx[0]),
        "voxel_iy": int(voxel_idx[1]),
        "voxel_iz": int(voxel_idx[2]),
        "theoretical_region_fraction": float(voxel_fraction),
        "n_cells_in_voxel": int(len(niche_idx)),
        "realized_region_fraction": float(realized_fraction),
        "gt_niche_type": (
            "ACE_region" if (has_niche and condition == 0)
            else "BD_region" if (has_niche and condition == 1)
            else "none"
        ),
        "counts_before_total": counts_before_total,
        "counts_after_total": counts_after_total,
        "counts_after_inside_region": counts_after_inside,
        "counts_after_outside_region": counts_after_outside,
    }

    return cell_df, meta


def simulate_cohort(cfg: SimConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(cfg.random_state)

    patient_df = make_patient_table(cfg.n_patients, cfg.n_conditions)
    patient_df = pick_niche_positive_patients(patient_df, cfg.prevalence, rng)

    all_cells = []
    all_meta = []

    for row in patient_df.itertuples(index=False):
        cell_df, meta = simulate_patient(
            patient_id=row.patient_id,
            condition=row.condition,
            has_niche=row.has_niche,
            cfg=cfg,
            rng=rng,
        )
        all_cells.append(cell_df)
        all_meta.append(meta)

    cells = pd.concat(all_cells, ignore_index=True)
    meta = pd.DataFrame(all_meta)

    return cells, meta


def run_parameter_sweep(
    grid_sizes: List[int] = [3, 4, 5, 6, 7, 8, 9, 10],
    prevalences: List[float] = [0.2, 0.4, 0.6, 0.8, 1.0],
    n_trials: int = 5,
    base_random_state: int = 0,
    **kwargs,
) -> pd.DataFrame:
    rows = []

    for g in grid_sizes:
        for prev in prevalences:
            for trial in range(n_trials):
                cfg = SimConfig(
                    grid_size=g,
                    prevalence=prev,
                    random_state=base_random_state + trial,
                    **kwargs,
                )
                cells, meta = simulate_cohort(cfg)

                rows.append({
                    "grid_size": g,
                    "prevalence": prev,
                    "trial": trial,
                    "n_patients": cfg.n_patients,
                    "cells_per_patient": cfg.cells_per_patient,
                    "domain_size": cfg.domain_size,
                    "n_total_cells": len(cells),
                    "n_niche_positive_patients": int(meta["has_niche"].sum()),
                    "theoretical_region_fraction": 1.0 / (g ** 3),
                    "mean_realized_region_fraction": float(meta["realized_region_fraction"].mean()),
                    "mean_cells_in_voxel": float(meta["n_cells_in_voxel"].mean()),
                    "sd_cells_in_voxel": float(meta["n_cells_in_voxel"].std(ddof=0)),
                })

    return pd.DataFrame(rows)


def print_cohort_summary(cells: pd.DataFrame, meta: pd.DataFrame) -> None:
    print("=== Cohort summary ===")
    print(f"Total cells: {len(cells)}")
    print(f"Patients: {meta.shape[0]}")
    print(f"Niche-positive patients: {int(meta['has_niche'].sum())}")
    print()

    print("Mean realized region fraction:")
    print(meta["realized_region_fraction"].mean())
    print()

    print("Mean cells in planted voxel:")
    print(meta["n_cells_in_voxel"].mean())
    print()

    print("Cell types across entire cohort:")
    print(cells["cell_type"].value_counts().sort_index())
    print()

    print("Ground-truth region labels across entire cohort:")
    print(cells["gt_region_label"].value_counts())
    print()


if __name__ == "__main__":
    # First 3D baseline:
    # - g = 3 gives ~3.7% planted region, close to the paper's 2D g = 5 example (~4%)
    # - preserve_global_counts=True isolates spatial organization from global abundance shifts
    cfg = SimConfig(
        n_patients=20,
        n_conditions=2,
        cells_per_patient=5000,
        domain_size=300.0,
        grid_size=3,
        prevalence=1.0,
        random_state=1,
        preserve_global_counts=True,
    )

    cells, meta = simulate_cohort(cfg)

    print(cells.head())
    print()
    print(meta[[
        "patient_id",
        "condition",
        "has_niche",
        "gt_niche_type",
        "n_cells_in_voxel",
        "theoretical_region_fraction",
        "realized_region_fraction",
    ]].head())
    print()

    print_cohort_summary(cells, meta)

    first_patient = meta.loc[0, "patient_id"]
    print(f"=== First patient: {first_patient} ===")
    print("Final cell-type counts:")
    print(cells.loc[cells["patient_id"] == first_patient, "cell_type"].value_counts().sort_index())
    print()

    print("Region membership counts:")
    print(cells.loc[cells["patient_id"] == first_patient, "gt_region_label"].value_counts())