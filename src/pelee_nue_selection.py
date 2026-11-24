"""Pandora PeLEE nue selections (1eNp0pi and 1e0p0pi), ported from the public PeLEE analysis code.

Source: https://github.com/ubneutrinos/PELEE (fetched 2026-11-23)
  * microfit/selections.py: the selection queries (PRESQ, NPPRESQ, NPLCUTQ, NPTCUTQ, BDTCQ, ZPBOXCUTS,
    ZPLOOSESEL, ZPBDTLOOSE)
  * data_loading.py: the derived variables the queries use, computed in the same order as load_sample:
    process_uproot_shower_variables -> process_uproot_recoveryvars -> post_process_shower_vars (only the
    pieces the selections need are ported; the recovery step, which re-assigns shower/track variables for
    split showers, spurious leading tracks, etc., is ported in full since it changes selection variables)

Differences from the PeLEE code, all deliberate:
  * the "nslice == 1" preselection cut uses slice_orig_pass_id == 1, as the MCC9.10 PeLEE ntuple notes say
    (PeLEE_Ntuples.pdf: "In MCC9.10 slice_orig_pass_id==1 replaces nslice==1");
  * the optical-filter part of the preselection, (_opfilter_pe_beam > 0 and _opfilter_pe_veto < 20), is not
    applied: those branches are not in our ntuples;
  * the BDT selections use the BDT score branches stored in the ntuples (bdt_pi0_np, bdt_nonpi0_np,
    bdt_bkg_0p) in place of PeLEE's offline xgboost scores (pi0_score, nonpi0_score, bkg_score), which come
    from pickled models outside the public repo.  That these branches are the same scores is NOT verified.

Works on the per-chunk pandas DataFrame in create_df.py / create_detvar_df.py (Pandora branches prefixed with
"pandora_", vector branches as per-event arrays) before the vector branches are removed.
"""

import numpy as np
import pandas as pd

from ntuple_variables.pandora_variables import pandora_pelee_scalar_vars as PELEE_SCALAR_BRANCHES
from ntuple_variables.pandora_variables import pandora_pelee_vector_vars as PELEE_VECTOR_BRANCHES

# output columns
PELEE_SELECTION_COLUMNS = [
    "pelee_nue_presel",     # PRESQ (without the optical filter)
    "pelee_1eNp_loose_box",  # NPLCUTQ
    "pelee_1eNp_tight_box",  # NPTCUTQ
    "pelee_1eNp_bdt",       # BDTCQ (with the ntuple BDT scores)
    "pelee_1e0p_box",       # ZPBOXCUTS
    "pelee_1e0p_bdt",       # ZPBDTLOOSE (with the ntuple BDT score)
]


# ----------------------------------------------------------------------------------------------------------
# PeLEE helpers (data_loading.py), on per-event arrays instead of awkward arrays
# ----------------------------------------------------------------------------------------------------------

def _is_vec(v):
    return isinstance(v, (np.ndarray, list, tuple))


def get_elm_from_vec_idx(vecs, idx, fillval=np.nan):
    """Element idx of each event's vector; fillval if idx is out of bounds (or the vector is missing)."""
    return np.array([v[int(i)] if _is_vec(v) and np.isfinite(i) and 0 <= i < len(v) else fillval
                     for v, i in zip(vecs, idx)], dtype=float)


def get_idx_from_vec_sort(argidx, vecsort, masks):
    """Index (in the unmasked vector) of the element at position argidx of the masked vector sorted by
    vecsort; -1 if there are not enough elements."""
    sizecheck = argidx if argidx >= 0 else abs(argidx) - 1
    out = []
    for v, m in zip(vecsort, masks):
        if not _is_vec(v):
            out.append(-1)
            continue
        m = np.asarray(m, dtype=bool)
        vid = np.asarray(v)[m]
        if len(vid) > sizecheck:
            out.append(int(np.flatnonzero(m)[vid.argsort()[argidx]]))
        else:
            out.append(-1)
    return np.array(out)


def distance(x1, y1, z1, x2, y2, z2):
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2)


def cos_angle_two_vecs(vx1, vy1, vz1, vx2, vy2, vz2):
    return (vx1 * vx2 + vy1 * vy2 + vz1 * vz2) / (
        np.sqrt(vx1 ** 2 + vy1 ** 2 + vz1 ** 2) * np.sqrt(vx2 ** 2 + vy2 ** 2 + vz2 ** 2))


# ----------------------------------------------------------------------------------------------------------
# derived variables
# ----------------------------------------------------------------------------------------------------------

def _shower_variables(df, up):
    """process_uproot_shower_variables (selection-relevant part)."""
    trk_id = df["trk_id"] - 1
    shr_id = df["shr_id"] - 1
    df["trkpid"] = get_elm_from_vec_idx(up["trk_llr_pid_score_v"], trk_id)
    for c in "xyz":
        df[f"shr_trk_sce_start_{c}"] = get_elm_from_vec_idx(up[f"trk_sce_start_{c}_v"], shr_id)
        df[f"shr_trk_sce_end_{c}"] = get_elm_from_vec_idx(up[f"trk_sce_end_{c}_v"], shr_id)
    df["shr_trk_len"] = distance(df["shr_trk_sce_start_x"], df["shr_trk_sce_start_y"], df["shr_trk_sce_start_z"],
                                 df["shr_trk_sce_end_x"], df["shr_trk_sce_end_y"], df["shr_trk_sce_end_z"])

    trk_mask = [np.asarray(v) > 0.5 if _is_vec(v) else np.zeros(0, bool) for v in up["trk_score_v"]]
    shr_mask = [np.asarray(v) < 0.5 if _is_vec(v) else np.zeros(0, bool) for v in up["trk_score_v"]]
    df["n_tracks_tot"] = [m.sum() for m in trk_mask]
    df["n_showers_tot"] = [m.sum() for m in shr_mask]

    trk_id_all = get_idx_from_vec_sort(-1, up["pfnhits"], trk_mask)  # includes uncontained tracks
    for c in "xyz":
        df[f"shr_start_{c}"] = get_elm_from_vec_idx(up[f"shr_start_{c}_v"], shr_id)
        df[f"trk1_start_{c}_alltk"] = get_elm_from_vec_idx(up[f"trk_start_{c}_v"], trk_id_all)
        df[f"trk1_dir_{c}_alltk"] = get_elm_from_vec_idx(up[f"trk_dir_{c}_v"], trk_id_all)
    df["tk1sh1_angle_alltk"] = np.where(
        df["n_tracks_tot"] == 0, np.nan,
        cos_angle_two_vecs(df["trk1_dir_x_alltk"], df["trk1_dir_y_alltk"], df["trk1_dir_z_alltk"],
                           df["shr_px"], df["shr_py"], df["shr_pz"]))

    # the 'subcluster' bug fix: take the leading shower's subclusters from the per-PFP vectors
    for k, plane in enumerate("UVY"):
        df[f"shrsubclusters{k}"] = get_elm_from_vec_idx(up[f"pfpplanesubclusters_{plane}"], shr_id, 0)

    shr2_id_corr = df["shr2_id"] - 1
    shr2_id_appr = get_idx_from_vec_sort(-2, up["pfnhits"], shr_mask)
    shr2_id = np.where((shr2_id_corr >= 0) & (shr2_id_corr < df["n_showers_tot"]), shr2_id_corr, shr2_id_appr)
    for k, plane in enumerate("UVY"):
        df[f"shr2subclusters{k}"] = get_elm_from_vec_idx(up[f"pfpplanesubclusters_{plane}"], shr2_id, 0)
    df["subcluster2tmp"] = df["shr2subclusters0"] + df["shr2subclusters1"] + df["shr2subclusters2"]
    for c in "xyz":
        df[f"shr2_start_{c}"] = get_elm_from_vec_idx(up[f"shr_start_{c}_v"], shr2_id)
        df[f"trk1_start_{c}"] = get_elm_from_vec_idx(up[f"trk_start_{c}_v"], trk_id)
    df["tk1sh2_distance"] = np.where(
        (df["n_showers_contained"] > 1) & (df["n_tracks_contained"] > 0),
        distance(df["shr2_start_x"], df["shr2_start_y"], df["shr2_start_z"],
                 df["trk1_start_x"], df["trk1_start_y"], df["trk1_start_z"]),
        np.nan)
    df["shr2pid"] = get_elm_from_vec_idx(up["trk_llr_pid_score_v"], shr2_id)


def _recovery_variables(df, up):
    """process_uproot_recoveryvars (the non-buggy-energy-estimator version), in full."""
    trk_id = df["trk_id"] - 1
    shr_id = df["shr_id"] - 1
    trk2_id = df["trk2_id"] - 1
    shr2_id = df["shr2_id"] - 1

    df["trk2_energy"] = get_elm_from_vec_idx(up["shr_energy_y_v"], trk2_id, 0.0)
    df["shr2_energy"] = get_elm_from_vec_idx(up["shr_energy_y_v"], shr2_id, 0.0)
    for c in "xyz":
        df[f"shr1_start_{c}"] = get_elm_from_vec_idx(up[f"shr_start_{c}_v"], shr_id)
        df[f"shr2_start_{c}"] = get_elm_from_vec_idx(up[f"shr_start_{c}_v"], shr2_id)
        df[f"shr12_start_d{c}"] = df[f"shr2_start_{c}"] - df[f"shr1_start_{c}"]
    df["shr12_cos_p1_dstart"] = np.where(
        (df["n_showers_contained"] < 2) | (df["shr2_energy"] < 0) | (df["shr12_start_dx"] == 0), np.nan,
        cos_angle_two_vecs(df["shr12_start_dx"], df["shr12_start_dy"], df["shr12_start_dz"],
                           df["shr_px"], df["shr_py"], df["shr_pz"]))
    df["trk1_len"] = get_elm_from_vec_idx(up["trk_len_v"], trk_id)
    df["trk2_len"] = get_elm_from_vec_idx(up["trk_len_v"], trk2_id)
    df["trk1_distance"] = get_elm_from_vec_idx(up["trk_distance_v"], trk_id)
    df["trk2_distance"] = get_elm_from_vec_idx(up["trk_distance_v"], trk2_id)
    df["trk1_llr_pid"] = get_elm_from_vec_idx(up["trk_llr_pid_score_v"], trk_id)
    df["trk2_llr_pid"] = get_elm_from_vec_idx(up["trk_llr_pid_score_v"], trk2_id)
    df["trk1_nhits"] = get_elm_from_vec_idx(up["pfnhits"], trk_id)
    df["trk2_nhits"] = get_elm_from_vec_idx(up["pfnhits"], trk2_id)
    df["shr1_nhits"] = get_elm_from_vec_idx(up["pfnhits"], shr_id)
    df["shr2_nhits"] = get_elm_from_vec_idx(up["pfnhits"], shr2_id)
    for c in "xyz":
        df[f"trk1_start_{c}"] = get_elm_from_vec_idx(up[f"trk_start_{c}_v"], trk_id)
        df[f"trk2_start_{c}"] = get_elm_from_vec_idx(up[f"trk_start_{c}_v"], trk2_id)
        df[f"trk1_dir_{c}"] = get_elm_from_vec_idx(up[f"trk_dir_{c}_v"], trk_id)
        df[f"trk2_dir_{c}"] = get_elm_from_vec_idx(up[f"trk_dir_{c}_v"], trk2_id)
    df["tk1sh1_distance"] = np.where(
        (df["n_showers_contained"] > 0) & (df["n_tracks_contained"] > 0),
        distance(df["shr_start_x"], df["shr_start_y"], df["shr_start_z"],
                 df["trk1_start_x"], df["trk1_start_y"], df["trk1_start_z"]), np.nan)
    df["tk2sh1_distance"] = np.where(
        (df["n_showers_contained"] > 0) & (df["n_tracks_contained"] > 1),
        distance(df["shr_start_x"], df["shr_start_y"], df["shr_start_z"],
                 df["trk2_start_x"], df["trk2_start_y"], df["trk2_start_z"]), np.nan)
    df["tk1tk2_distance"] = np.where(
        df["n_tracks_contained"] > 1,
        distance(df["trk1_start_x"], df["trk1_start_y"], df["trk1_start_z"],
                 df["trk2_start_x"], df["trk2_start_y"], df["trk2_start_z"]), np.nan)
    df["tk1sh1_angle"] = np.where(
        (df["n_tracks_contained"] < 1) | (df["n_showers_contained"] < 1), np.nan,
        cos_angle_two_vecs(df["trk1_dir_x"], df["trk1_dir_y"], df["trk1_dir_z"], df["shr_px"], df["shr_py"], df["shr_pz"]))
    df["tk2sh1_angle"] = np.where(
        (df["n_tracks_contained"] < 2) | (df["n_showers_contained"] < 1), np.nan,
        cos_angle_two_vecs(df["trk2_dir_x"], df["trk2_dir_y"], df["trk2_dir_z"], df["shr_px"], df["shr_py"], df["shr_pz"]))
    df["tk1tk2_angle"] = np.where(
        df["n_tracks_contained"] < 2, np.nan,
        cos_angle_two_vecs(df["trk1_dir_x"], df["trk1_dir_y"], df["trk1_dir_z"],
                           df["trk2_dir_x"], df["trk2_dir_y"], df["trk2_dir_z"]))

    # 2nd shower split from the main one
    df["is_shr2splt"] = np.zeros_like(df["n_tracks_contained"])
    shr2splt = ((df["n_tracks_contained"] > 0) & (df["n_showers_contained"] > 1) & (df["shr12_cos_p1_dstart"] > 0.95)
                & (df["tk1sh2_distance"] > 60) & (df["shr_score"] < 0.1)
                & ((df["shrsubclusters0"] + df["shrsubclusters1"] + df["shrsubclusters2"]) > 3))
    df.loc[shr2splt, "is_shr2splt"] = 1
    df.loc[shr2splt, "n_showers_contained"] = 1  # PeLEE: "assume this happens to nues only!"
    for k, plane in enumerate("UVY"):
        df[f"shr2subclusters{k}"] = get_elm_from_vec_idx(up[f"pfpplanesubclusters_{plane}"], shr2_id, 0)
    for k in range(3):
        df.loc[shr2splt, f"shrsubclusters{k}"] = df[f"shrsubclusters{k}"] + df[f"shr2subclusters{k}"]
    df.loc[shr2splt, "subcluster"] = df["shrsubclusters0"] + df["shrsubclusters1"] + df["shrsubclusters2"]
    df.loc[shr2splt & (df["shr1shr2moliereavg"] > 0), "shrmoliereavg"] = df["shr1shr2moliereavg"]

    # spurious leading track (more than 30 cm from the vertex)
    df["is_trk1bad"] = np.zeros_like(df["n_tracks_contained"])
    trk1bad = (df["n_tracks_contained"] > 1) & (df["trk_distance"] > 30.0) & (df["is_shr2splt"] == 0)
    df.loc[trk1bad, "is_trk1bad"] = 1
    df.loc[trk1bad, "trkpid"] = df["trk2_llr_pid"]
    df.loc[trk1bad, "tksh_distance"] = df["tk2sh1_distance"]
    df.loc[trk1bad, "tksh_angle"] = df["tk2sh1_angle"]
    df.loc[trk1bad, "hits_ratio"] = df["shr_hits_tot"] / (df["shr_hits_tot"] + df["trk_hits_tot"] - df["trk1_nhits"])
    df.loc[trk1bad, "trk_len"] = df["trk2_len"]
    df.loc[trk1bad, "trk_distance"] = df["trk2_distance"]
    df["trk2_score"] = get_elm_from_vec_idx(up["trk_score_v"], trk2_id)
    df["trk2_protonenergy"] = get_elm_from_vec_idx(up["trk_energy_proton_v"], trk2_id)
    df["trk2_theta"] = get_elm_from_vec_idx(up["trk_theta_v"], trk2_id)
    df["trk2_phi"] = get_elm_from_vec_idx(up["trk_phi_v"], trk2_id)
    df.loc[trk1bad, "trk_score"] = df["trk2_score"]
    df.loc[trk1bad, "trk_theta"] = df["trk2_theta"]
    df.loc[trk1bad, "trk_phi"] = df["trk2_phi"]
    df.loc[trk1bad, "trkshrhitdist2"] = df["trk2shrhitdist2"]
    df.loc[trk1bad, "n_tracks_contained"] = df["n_tracks_contained"] - 1
    df.loc[trk1bad, "trk_energy_tot"] = df["trk_energy_tot"] - df["trk_energy"]

    # 2nd track is actually the start of the shower
    df["is_trk2srtshr"] = np.zeros_like(df["n_tracks_contained"])
    trk2srtshr = ((df["n_tracks_contained"] > 1) & (df["tk2sh1_angle"] > 0.98)
                  & (df["tk1tk2_distance"] < df["tksh_distance"]) & (df["shr_score"] < 0.1)
                  & (df["is_shr2splt"] == 0) & (df["is_trk1bad"] == 0))
    df.loc[trk2srtshr, "is_trk2srtshr"] = 1
    for p in "uvy":
        df[f"trk2_tkfit_dedx_{p}"] = get_elm_from_vec_idx(up[f"shr_tkfit_dedx_{p}_v"], trk2_id)
        df[f"trk2_tkfit_nhits_{p}"] = get_elm_from_vec_idx(up[f"shr_tkfit_dedx_nhits_{p}_v"], trk2_id)
    df["trk2_tkfit_nhits_tot"] = df["trk2_tkfit_nhits_u"] + df["trk2_tkfit_nhits_v"] + df["trk2_tkfit_nhits_y"]
    for k, plane in enumerate("UVY"):
        df[f"trk2subclusters{k}"] = get_elm_from_vec_idx(up[f"pfpplanesubclusters_{plane}"], trk2_id, 0)
    df.loc[trk2srtshr, "tksh_distance"] = df["tk1tk2_distance"]
    df.loc[trk2srtshr, "tksh_angle"] = df["tk1tk2_angle"]
    has_fit = trk2srtshr & (df["trk2_tkfit_nhits_tot"] > 0)
    for P, p in zip("UVY", "uvy"):
        df.loc[has_fit, f"shr_tkfit_dedx_{P}"] = df[f"trk2_tkfit_dedx_{p}"]
        df.loc[has_fit, f"shr_tkfit_nhits_{P}"] = df[f"trk2_tkfit_nhits_{p}"]
    df.loc[trk2srtshr, "hits_ratio"] = (df["shr_hits_tot"] + df["trk2_nhits"]) / (df["shr_hits_tot"] + df["trk_hits_tot"])
    df.loc[trk2srtshr, "shr_tkfit_npointsvalid"] = df["shr_tkfit_npointsvalid"] + df["trk2_nhits"]
    df.loc[trk2srtshr & (df["trk1trk2hitdist2"] > 0) & (df["trkshrhitdist2"] > 0)
           & (df["trk1trk2hitdist2"] < df["trkshrhitdist2"]), "trkshrhitdist2"] = df["trk1trk2hitdist2"]
    df.loc[trk2srtshr & (df["trk1trk2hitdist2"] > 0) & (df["trkshrhitdist2"] < 0), "trkshrhitdist2"] = df["trk1trk2hitdist2"]
    for k in range(3):
        df.loc[trk2srtshr, f"shrsubclusters{k}"] = df[f"shrsubclusters{k}"] + df[f"trk2subclusters{k}"]
    df.loc[trk2srtshr, "subcluster"] = df["shrsubclusters0"] + df["shrsubclusters1"] + df["shrsubclusters2"]
    df.loc[trk2srtshr & (df["shr1trk2moliereavg"] > 0), "shrmoliereavg"] = df["shr1trk2moliereavg"]
    df.loc[trk2srtshr, "n_tracks_contained"] = df["n_tracks_contained"] - 1
    df.loc[trk2srtshr, "trk_energy_tot"] = df["trk_energy_tot"] - df["trk2_protonenergy"]
    df.loc[trk2srtshr & (df["trk2_energy"] < 0.0), "trk2_energy"] = 0.0
    df["trk2_energy_cali"] = 0.001 * df["trk2_energy"] * df["shr_energy_tot_cali"] / df["shr_energy_tot"]

    # 2nd shower is actually an attached proton
    shr2prtn = ((df["n_showers_contained"] > 1) & (df["tk1sh2_distance"] < 6.0) & (df["subcluster2tmp"] <= 4)
                & (df["shr2pid"] < 0.02))
    df.loc[shr2prtn, "n_showers_contained"] = df["n_showers_contained"] - 1
    df.loc[shr2prtn, "n_tracks_contained"] = df["n_tracks_contained"] + 1
    df.loc[shr2prtn & (df["shr2_energy"] < 0.0), "shr2_energy"] = 0.0
    # (PeLEE subtracts trk2_energy_cali here, not shr2_energy_cali; kept as in the source)
    df.loc[shr2prtn, "shr_energy_tot_cali"] = df["shr_energy_tot_cali"] - df["trk2_energy_cali"]


def _post_process_shower_vars(df):
    """post_process_shower_vars (selection-relevant part); after the recovery step."""
    df["subcluster"] = df["shrsubclusters0"] + df["shrsubclusters1"] + df["shrsubclusters2"]
    df["trkfit"] = df["shr_tkfit_npointsvalid"] / df["shr_tkfit_npoints"]
    df["anglediff_Y"] = np.abs(df["secondshower_Y_dir"] - df["shrclusdir2"])
    df["shr_tkfit_nhits_tot"] = df["shr_tkfit_nhits_Y"] + df["shr_tkfit_nhits_U"] + df["shr_tkfit_nhits_V"]
    for kind in ["", "2cm_", "gap10_"]:
        col = f"shr_tkfit_{kind}dedx_max"
        df[col] = df[f"shr_tkfit_{kind}dedx_Y"]
        u_most = df[f"shr_tkfit_{kind}nhits_U"] > df[f"shr_tkfit_{kind}nhits_Y"]
        df.loc[u_most, col] = df[f"shr_tkfit_{kind}dedx_U"]
        v_most = ((df[f"shr_tkfit_{kind}nhits_V"] > df[f"shr_tkfit_{kind}nhits_Y"])
                  & (df[f"shr_tkfit_{kind}nhits_V"] > df[f"shr_tkfit_{kind}nhits_U"]))
        df.loc[v_most, col] = df[f"shr_tkfit_{kind}dedx_V"]


# ----------------------------------------------------------------------------------------------------------
# selections (microfit/selections.py; NaN comparisons are False, as in pandas queries)
# ----------------------------------------------------------------------------------------------------------

def _selections(d):
    presel = (d["slice_orig_pass_id"] == 1) & (d["selected"] == 1) & (d["shr_energy_tot_cali"] > 0.07)

    np_presel = presel & (d["n_tracks_contained"] > 0)
    np_loose = (np_presel & (d["CosmicIPAll3D"] > 10.) & (d["trkpid"] < 0.02) & (d["hits_ratio"] > 0.50)
                & (d["shrmoliereavg"] < 9) & (d["subcluster"] > 4) & (d["trkfit"] < 0.65) & (d["tksh_distance"] < 6.0)
                & (d["shr_tkfit_nhits_tot"] > 1) & (d["shr_tkfit_dedx_max"] > 0.5) & (d["shr_tkfit_dedx_max"] < 5.5)
                & (d["tksh_angle"] > -0.9) & (d["shr_trk_len"] < 300.))
    np_very_loose = (np_presel & (d["CosmicIPAll3D"] > 10.) & (d["trkpid"] < 0.25) & (d["hits_ratio"] > 0.5)
                     & (d["trkfit"] < 0.90) & (d["tksh_distance"] < 10.0) & (d["tksh_angle"] > -0.9))
    np_tight = (np_very_loose & (d["CosmicIPAll3D"] > 30.) & (d["CosmicDirAll3D"] > -0.98) & (d["CosmicDirAll3D"] < 0.98)
                & (d["trkpid"] < 0.02) & (d["hits_ratio"] > 0.65) & (d["shr_score"] < 0.25)
                & (d["shrmoliereavg"] > 2) & (d["shrmoliereavg"] < 10) & (d["subcluster"] > 7) & (d["trkfit"] < 0.70)
                & (d["tksh_distance"] < 4.0) & (d["trkshrhitdist2"] < 1.5)
                & (d["shr_tkfit_nhits_tot"] > 1) & (d["shr_tkfit_dedx_max"] > 1.0) & (d["shr_tkfit_dedx_max"] < 3.8)
                & ((d["secondshower_Y_nhit"] <= 8) | (d["secondshower_Y_dot"] <= 0.8) | (d["anglediff_Y"] <= 40)
                   | (d["secondshower_Y_vtxdist"] >= 100))
                & (d["tksh_angle"] > -0.9) & (d["tksh_angle"] < 0.70) & (d["shr_trk_len"] < 300.))
    one_shower = d["n_showers_contained"] == 1
    np_bdt = np_loose & (d["bdt_pi0_np"] > 0.67) & (d["bdt_nonpi0_np"] > 0.70)

    zp_box = (presel & one_shower & (d["shrmoliereavg"] > 1) & (d["shrmoliereavg"] < 8) & (d["shr_score"] < 0.05)
              & (d["CosmicIPAll3D"] > 20.) & (d["CosmicDirAll3D"] < 0.75) & (d["CosmicDirAll3D"] > -0.75)
              & (d["trkfit"] < 0.4) & (d["subcluster"] > 6)
              & (d["shr_tkfit_gap10_dedx_Y"] > 1.5) & (d["shr_tkfit_gap10_dedx_Y"] < 2.5)
              & (d["shr_tkfit_gap10_dedx_U"] > 1.5) & (d["shr_tkfit_gap10_dedx_U"] < 3.75)
              & (d["shr_tkfit_gap10_dedx_V"] > 1.5) & (d["shr_tkfit_gap10_dedx_V"] < 3.75)
              & (d["shr_tkfit_2cm_dedx_max"] > 1.) & (d["shr_tkfit_2cm_dedx_max"] < 4.) & (d["shr_trk_len"] < 300.)
              & (d["n_tracks_contained"] == 0))
    alltk_angle_ok = (d["n_tracks_tot"] == 0) | ((d["n_tracks_tot"] > 0) & (d["tk1sh1_angle_alltk"] > -0.9))
    zp_loose = (presel & one_shower & (d["CosmicIPAll3D"] > 10.) & (d["CosmicDirAll3D"] > -0.9)
                & (d["CosmicDirAll3D"] < 0.9) & (d["shrmoliereavg"] < 15) & (d["subcluster"] > 4) & (d["trkfit"] < 0.65)
                & (d["secondshower_Y_nhit"] < 50) & (d["shr_trk_sce_start_y"] > -100) & (d["shr_trk_sce_start_y"] < 80)
                & (d["shr_trk_sce_end_y"] > -100) & (d["shr_trk_sce_end_y"] < 100) & (d["shr_trk_len"] < 300.)
                & alltk_angle_ok)
    zp_bdt = zp_loose & (d["bdt_bkg_0p"] > 0.72) & (d["n_tracks_contained"] == 0) & alltk_angle_ok

    return {
        "pelee_nue_presel": presel,
        "pelee_1eNp_loose_box": np_loose & one_shower,
        "pelee_1eNp_tight_box": np_tight & one_shower,
        "pelee_1eNp_bdt": np_bdt & one_shower,
        "pelee_1e0p_box": zp_box,
        "pelee_1e0p_bdt": zp_bdt,
    }


def add_pelee_nue_selections(df):
    """Append the PELEE_SELECTION_COLUMNS booleans to a create_df chunk (pandas, pandora_-prefixed columns,
    vector branches still present).  The derived variables are computed on a private copy, so the existing
    pandora_* columns are not modified."""
    d = pd.DataFrame({b: pd.to_numeric(df[f"pandora_{b}"], errors="coerce").to_numpy(dtype=float)
                      for b in PELEE_SCALAR_BRANCHES})
    up = {b: df[f"pandora_{b}"].to_numpy() for b in PELEE_VECTOR_BRANCHES}
    _shower_variables(d, up)
    _recovery_variables(d, up)
    _post_process_shower_vars(d)
    for col, sel in _selections(d).items():
        df[col] = sel.fillna(False).to_numpy(dtype=bool)
    return df
