import anndata as ad
import pandas as pd
import numpy as np
import quiche as qu

def cells_to_adata(cells):
    cells = cells.copy()

    cells["condition"] = cells["condition"].astype(str)
    cells["fov"] = cells["patient_id"]
    cells["cell_cluster"] = pd.Categorical(cells["cell_type"])
    cells["label"] = cells["cell_id"].astype(str)

    X_df = pd.get_dummies(cells["cell_cluster"], dtype=float)
    X = X_df.to_numpy()

    adata = ad.AnnData(X=X)
    adata.obs_names = cells["cell_id"].astype(str).values

    obs_cols = [
        "patient_id",
        "condition",
        "fov",
        "label",
        "cell_cluster",
        "cell_type",
        "gt_region_label",
        "in_niche_voxel",
    ]

    adata.obs = cells[obs_cols].copy()
    adata.obs.index = adata.obs_names
    adata.obs = adata.obs.rename(columns={"patient_id": "Patient_ID"})

    adata.obs["Patient_ID"] = adata.obs["Patient_ID"].astype("category")
    adata.obs["fov"] = adata.obs["fov"].astype("category")
    adata.obs["condition"] = adata.obs["condition"].astype("category")
    adata.obs["cell_cluster"] = pd.Categorical(adata.obs["cell_cluster"])

    adata.obsm["spatial3d"] = cells[["x", "y", "z"]].to_numpy().astype(float)
    adata.var_names = X_df.columns.astype(str)

    return adata


# Classify annotations
def classify_quiche_annotation(label):
    if pd.isna(label) or label == "":
        return "unidentified"

    parts = set(str(label).split("__"))

    ace = {"A", "C", "E"}
    bd = {"B", "D"}

    has_ace = len(parts.intersection(ace))
    has_bd = len(parts.intersection(bd))

    if has_ace >= 2 and has_bd == 0:
        return "ACE_like"
    elif has_bd == 2:
        return "BD_like"
    elif has_ace >= 2 and has_bd > 0:
        return "ACE_mixed"
    elif has_bd > 0 and has_ace > 0:
        return "BD_mixed"
    else:
        return "other"
    



    # Compute recovery metrics
def compute_recovery_metrics(q, fdr_threshold=0.05):
    var = q.mdata["quiche"].var.copy()

    var["predicted_class"] = var["quiche_niche"].apply(classify_quiche_annotation)
    var["is_sig"] = var["SpatialFDR"] < fdr_threshold

    sig = var[var["is_sig"]]

    # Direction-aware recall
    ace_sig = sig[
        (sig["gt_region_label"] == "ACE_region") &
        (sig["logFC"] < 0)
    ]

    bd_sig = sig[
        (sig["gt_region_label"] == "BD_region") &
        (sig["logFC"] > 0)
    ]

    ace_total = (var["gt_region_label"] == "ACE_region").sum()
    bd_total = (var["gt_region_label"] == "BD_region").sum()

    ace_recall = len(ace_sig) / ace_total if ace_total > 0 else np.nan
    bd_recall = len(bd_sig) / bd_total if bd_total > 0 else np.nan

    # Purity
    ace_like_sig = sig[
        sig["predicted_class"].isin(["ACE_like", "ACE_mixed"]) &
        (sig["logFC"] < 0)
    ]

    bd_like_sig = sig[
        sig["predicted_class"].isin(["BD_like", "BD_mixed"]) &
        (sig["logFC"] > 0)
    ]

    ace_purity = (
        (ace_like_sig["gt_region_label"] == "ACE_region").mean()
        if len(ace_like_sig) > 0 else np.nan
    )

    bd_purity = (
        (bd_like_sig["gt_region_label"] == "BD_region").mean()
        if len(bd_like_sig) > 0 else np.nan
    )

    # Neighbor stats
    conn = q.adata.obsp["spatial_connectivities"].tocsr()
    neighbor_counts = np.diff(conn.indptr)

    return {
        "n_total_niches": var.shape[0],
        "n_sig": len(sig),
        "frac_sig": len(sig) / var.shape[0],
        "ace_recall": ace_recall,
        "bd_recall": bd_recall,
        "ace_purity": ace_purity,
        "bd_purity": bd_purity,
        "mean_neighbors": float(np.mean(neighbor_counts)),
        "frac_lt3_neighbors": float(np.mean(neighbor_counts < 3)),
    }



def run_quiche_once(
    adata,
    radius,
    n_neighbors,
    k_sim,
    sketch_size,
):
    adata = adata.copy()

    q = qu.tl.QUICHE(
        adata=adata,
        labels_key="cell_cluster",
        spatial_key="spatial3d",
        fov_key="fov",
        patient_key="Patient_ID",
        segmentation_label_key="label",
    )

    q.compute_spatial_niches(
        radius=radius,
        n_neighbors=n_neighbors,
        khop=None,
        min_cell_threshold=3,
        coord_type="generic",
        delaunay=False,
    )

    q.subsample(sketch_size=sketch_size)

    q.differential_enrichment(
        k_sim=k_sim,
        design="~condition",
        model_contrasts="condition1-condition0",
    )

    q.annotate_niches(
        annotation_scheme="neighborhood",
        annotation_key="quiche_niche",
        nlargest=3,
        min_perc=0.1,
    )

    return q

