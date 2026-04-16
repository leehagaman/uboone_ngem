import numpy as np
from tqdm import tqdm
import pandas as pd


# MicroBooNE TPC active volume boundaries [cm]
TPC_X_MIN, TPC_X_MAX = -1., 254.3
TPC_Y_MIN, TPC_Y_MAX = -115., 117.
TPC_Z_MIN, TPC_Z_MAX = 0.6, 1036.4

SPHERE_RADIUS_CM = 75.0           # sphere radius around WC shower vertex [cm]
FORWARD_CONE_HALF_ANGLE_DEG = 45.0  # half-angle of forward shower cone to exclude [deg]


def is_blip_in_tpc(x, y, z):
    """Check whether a blip position is within the MicroBooNE TPC active volume."""
    return (TPC_X_MIN < x < TPC_X_MAX and
            TPC_Y_MIN < y < TPC_Y_MAX and
            TPC_Z_MIN < z < TPC_Z_MAX)


def is_within_sphere_outside_conic(sh_vtx, sh_mom_unit, blip_xyz,
                                    radius=SPHERE_RADIUS_CM,
                                    cone_half_angle_deg=FORWARD_CONE_HALF_ANGLE_DEG):
    """
    Returns True if the blip is within the sphere AND outside the forward shower cone.

    Parameters
    ----------
    sh_vtx : array-like, shape (3,)       WC shower vertex position [cm]
    sh_mom_unit : array-like, shape (3,)  unit vector of shower momentum direction
    blip_xyz : array-like, shape (3,)     blip position [cm]
    radius : float                        sphere radius [cm]
    cone_half_angle_deg : float           half-angle of the forward cone to exclude [deg]
    """
    vtx_to_blip = blip_xyz - sh_vtx
    dist = np.linalg.norm(vtx_to_blip)
    if dist < 1e-10 or dist >= radius:
        return False
    cos_angle = np.dot(vtx_to_blip / dist, sh_mom_unit)
    cos_cone = np.cos(np.radians(cone_half_angle_deg))
    # outside cone: angle to shower momentum > cone_half_angle_deg
    return cos_angle < cos_cone


def is_in_cone(apex, direction_unit, blip_xyz,
               half_angle_deg=FORWARD_CONE_HALF_ANGLE_DEG):
    """
    Returns True if blip_xyz is within the cone defined by apex, direction_unit,
    and half_angle_deg.
    """
    vtx_to_blip = blip_xyz - apex
    dist = np.linalg.norm(vtx_to_blip)
    if dist < 1e-10:
        return True
    cos_angle = np.dot(vtx_to_blip / dist, direction_unit)
    return cos_angle >= np.cos(np.radians(half_angle_deg))


def is_backtracked_blip(dist2vtx, cos_angle_sh):
    """
    Returns True if the blip is in the backtrack cone (Region B).

    Three nested cones around the backward shower direction:
      - within 75 cm  and within  4 deg of backward direction
      - within 50 cm  and within  8 deg of backward direction
      - within 25 cm  and within 11 deg of backward direction

    Parameters
    ----------
    dist2vtx : float    blip distance from WC shower vertex [cm]
    cos_angle_sh : float  cos(angle between vtx→blip vector and shower momentum direction)
    """
    cos_4  = np.cos(np.radians(4.0))
    cos_8  = np.cos(np.radians(8.0))
    cos_11 = np.cos(np.radians(11.0))
    # cos(angle from backward direction) = -cos_angle_sh
    bwd_cos = -cos_angle_sh
    return ((dist2vtx < 75.0 and bwd_cos > cos_4) or
            (dist2vtx < 50.0 and bwd_cos > cos_8) or
            (dist2vtx < 25.0 and bwd_cos > cos_11))


def reco_proton_blip(blip_energy, blip_dx, blip_dw, filetype):
    if filetype == "data" or filetype == "ext":
        a1, a2 = 1.72, 2.08
    else:
        a1, a2 = 1.69, 2.64
    blip_ds = np.sqrt(blip_dx**2 + blip_dw**2)
    return blip_energy / blip_ds > a1 * np.log(a2 * blip_energy)


def do_blip_postprocessing(df):

    all_blip_x = df["blip_x"].to_numpy()
    all_blip_y = df["blip_y"].to_numpy()
    all_blip_z = df["blip_z"].to_numpy()
    all_blip_energy = df["blip_energy"].to_numpy()
    all_blip_dx = df["blip_dx"].to_numpy()
    all_blip_dw = df["blip_dw"].to_numpy()
    all_wc_reco_shower_momentum = df["wc_reco_showerMomentum"].to_numpy()
    all_wc_reco_shower_vtx_x = df["wc_reco_showervtxX"].to_numpy()
    all_wc_reco_shower_vtx_y = df["wc_reco_showervtxY"].to_numpy()
    all_wc_reco_shower_vtx_z = df["wc_reco_showervtxZ"].to_numpy()
    filetype = df["filetype"].to_numpy()

    # -----------------------------------------
    # Part A: closest upstream blip
    # -----------------------------------------
    closest_upstream_blip_distance = []
    closest_upstream_blip_angle = []
    closest_upstream_blip_impact_parameter = []
    closest_upstream_blip_energy = []
    closest_upstream_blip_dx = []
    closest_upstream_blip_dw = []
    for event_index in tqdm(range(len(df)), desc="Finding closest upstream blip", mininterval=10):

        curr_closest_upstream_blip_distance = np.inf
        curr_closest_upstream_blip_angle = np.nan
        curr_closest_upstream_blip_impact_parameter = np.nan
        curr_closest_upstream_blip_energy = np.nan
        curr_closest_upstream_blip_dx = np.nan
        curr_closest_upstream_blip_dw = np.nan

        blip_xs = all_blip_x[event_index]
        blip_ys = all_blip_y[event_index]
        blip_zs = all_blip_z[event_index]
        blip_energies = all_blip_energy[event_index]
        blip_dxs = all_blip_dx[event_index]
        blip_dws = all_blip_dw[event_index]

        # Check if wc_reco_shower_momentum is a float/NaN (not subscriptable)
        if isinstance(all_wc_reco_shower_momentum[event_index], float) or not hasattr(all_wc_reco_shower_momentum[event_index], '__getitem__'):
            closest_upstream_blip_distance.append(curr_closest_upstream_blip_distance)
            closest_upstream_blip_angle.append(curr_closest_upstream_blip_angle)
            closest_upstream_blip_impact_parameter.append(curr_closest_upstream_blip_impact_parameter)
            closest_upstream_blip_energy.append(curr_closest_upstream_blip_energy)
            closest_upstream_blip_dx.append(curr_closest_upstream_blip_dx)
            closest_upstream_blip_dw.append(curr_closest_upstream_blip_dw)
            continue

        wc_reco_shower_momentum = np.array([all_wc_reco_shower_momentum[event_index][0],
                                            all_wc_reco_shower_momentum[event_index][1],
                                            all_wc_reco_shower_momentum[event_index][2]])
        wc_reco_shower_momentum_unit = wc_reco_shower_momentum / np.linalg.norm(wc_reco_shower_momentum)
        wc_reco_shower_vtx_x = all_wc_reco_shower_vtx_x[event_index]
        wc_reco_shower_vtx_y = all_wc_reco_shower_vtx_y[event_index]
        wc_reco_shower_vtx_z = all_wc_reco_shower_vtx_z[event_index]

        for blip_index in range(len(blip_xs)):
            dist_to_WC_shower_vtx = np.sqrt((blip_xs[blip_index] - wc_reco_shower_vtx_x)**2
                                          + (blip_ys[blip_index] - wc_reco_shower_vtx_y)**2
                                          + (blip_zs[blip_index] - wc_reco_shower_vtx_z)**2)
            threshold_dist = 2 # cm, must be at least 2 cm away to be considered upstream
            shower_vtx_to_blip_vector = np.array([blip_xs[blip_index] - wc_reco_shower_vtx_x,
                                        blip_ys[blip_index] - wc_reco_shower_vtx_y,
                                        blip_zs[blip_index] - wc_reco_shower_vtx_z])
            shower_vtx_to_blip_vector_unit = shower_vtx_to_blip_vector / np.linalg.norm(shower_vtx_to_blip_vector)
            dot_product = np.dot(shower_vtx_to_blip_vector_unit, -wc_reco_shower_momentum_unit)
            dot_product = np.clip(dot_product, -1.0, 1.0) # accounting for floating point errors near 1 or -1
            blip_angle = np.arccos(dot_product) * 180 / np.pi # angle between vtx to blip vector and backwards shower momentum vector
            if threshold_dist < dist_to_WC_shower_vtx < curr_closest_upstream_blip_distance and blip_angle < 90:
                curr_closest_upstream_blip_distance = dist_to_WC_shower_vtx
                curr_closest_upstream_blip_angle = blip_angle
                curr_projected_vec = np.dot(shower_vtx_to_blip_vector, wc_reco_shower_momentum_unit) * wc_reco_shower_momentum_unit # vtx to blip vector projected along shower axis
                curr_closest_upstream_blip_impact_parameter = np.linalg.norm(shower_vtx_to_blip_vector - curr_projected_vec)
                curr_closest_upstream_blip_energy = blip_energies[blip_index]
                curr_closest_upstream_blip_dx = blip_dxs[blip_index]
                curr_closest_upstream_blip_dw = blip_dws[blip_index]
        closest_upstream_blip_distance.append(curr_closest_upstream_blip_distance)
        closest_upstream_blip_angle.append(curr_closest_upstream_blip_angle)
        closest_upstream_blip_impact_parameter.append(curr_closest_upstream_blip_impact_parameter)
        closest_upstream_blip_energy.append(curr_closest_upstream_blip_energy)
        closest_upstream_blip_dx.append(curr_closest_upstream_blip_dx)
        closest_upstream_blip_dw.append(curr_closest_upstream_blip_dw)

    df["blip_closest_upstream_distance"] = closest_upstream_blip_distance
    df["blip_closest_upstream_angle"] = closest_upstream_blip_angle
    df["blip_closest_upstream_impact_parameter"] = closest_upstream_blip_impact_parameter
    df["blip_closest_upstream_energy"] = closest_upstream_blip_energy
    df["blip_closest_upstream_dx"] = closest_upstream_blip_dx
    df["blip_closest_upstream_dw"] = closest_upstream_blip_dw


    # -----------------------------------------
    # Part B: vertex → blips (multiple vertices, multiple radii)
    # -----------------------------------------
    radii_cm = [5, 10, 30, 50, 75, 100]

    vertices = {
        "wc": ("wc_reco_nuvtxX", "wc_reco_nuvtxY", "wc_reco_nuvtxZ"),
        "pandora_pelee": ("pandora_reco_nu_vtx_sce_x", "pandora_reco_nu_vtx_sce_y", "pandora_reco_nu_vtx_sce_z"),
        "pandora_glee": ("glee_reco_vertex_x", "glee_reco_vertex_y", "glee_reco_vertex_z"),
        "lantern": ("lantern_vtxX", "lantern_vtxY", "lantern_vtxZ"),
    }

    # pull vertex arrays once
    vtx_arrays = {}
    for key, (cx, cy, cz) in vertices.items():
        vtx_arrays[key] = (df[cx].to_numpy(), df[cy].to_numpy(), df[cz].to_numpy())

    # outputs: per vertex
    out_minDist = {k: [] for k in vertices}
    out_minE = {k: [] for k in vertices}
    out_nWithin = {k: {R: [] for R in radii_cm} for k in vertices}

    for event_index in tqdm(range(len(df)), desc="Vertex → blips (WC/Pandora/LANTERN)", mininterval=10):

        blip_xs = all_blip_x[event_index]
        blip_ys = all_blip_y[event_index]
        blip_zs = all_blip_z[event_index]
        blip_energies = all_blip_energy[event_index]

        has_blips = (blip_xs is not None) and (len(blip_xs) > 0)

        # If no blips, fill defaults for all vertices
        if not has_blips:
            for key in vertices:
                out_minDist[key].append(np.inf)
                out_minE[key].append(np.nan)
                for R in radii_cm:
                    out_nWithin[key][R].append(0)
            continue

        bx = np.asarray(blip_xs, dtype=float)
        by = np.asarray(blip_ys, dtype=float)
        bz = np.asarray(blip_zs, dtype=float)
        be = np.asarray(blip_energies, dtype=float)

        for key in vertices:
            vx_arr, vy_arr, vz_arr = vtx_arrays[key]
            vx, vy, vz = vx_arr[event_index], vy_arr[event_index], vz_arr[event_index]

            # bad vertex
            if not (np.isfinite(vx) and np.isfinite(vy) and np.isfinite(vz)):
                out_minDist[key].append(np.inf)
                out_minE[key].append(np.nan)
                for R in radii_cm:
                    out_nWithin[key][R].append(0)
                continue

            d = np.sqrt((bx - vx)**2 + (by - vy)**2 + (bz - vz)**2)

            jmin = int(np.argmin(d))
            out_minDist[key].append(float(d[jmin]))
            out_minE[key].append(float(be[jmin]))

            for R in radii_cm:
                out_nWithin[key][R].append(int(np.sum(d < R)))

    # attach columns
    for key in vertices:
        df[f"{key}_blip_minDist"] = out_minDist[key]
        df[f"{key}_blip_minDist_energy"] = out_minE[key]
        for R in radii_cm:
            df[f"{key}_blip_nWithin_{R}cm"] = out_nWithin[key][R]


    # -----------------------------------------
    # Part C: sphere / no_shower_cone / backtrack_cones / no_shower_cone_no_backtrack_cones
    #         (originally named: sphere / signal / Region B / Region A in the C++ anamacro)
    #
    # All regions use the WC shower vertex and shower momentum direction.
    # Quality cuts (sphere selection, mirroring the C++ anamacro):
    #   blip_nplanes > 1          (3D blips matched on 2+ planes)
    #   blip_touchtrk == 0        (not touching a track)
    #   blip_pl2_bydeadwire == 0  (not adjacent to dead wire on collection plane)
    #   blip_proxtrkdist > 15 cm  (at least 15 cm from nearest track)
    #   dist to shower vertex < SPHERE_RADIUS_CM
    #   blip inside TPC active volume
    #
    # Regions (applied on top of sphere quality cuts):
    #   no_shower_cone:                    is_within_sphere_outside_conic — outside the 45-deg forward cone
    #   backtrack_cones:                   is_backtracked_blip — within the backward cones
    #   no_shower_cone_no_backtrack_cones: no_shower_cone blips NOT in backtrack_cones
    # -----------------------------------------
    all_blip_nplanes       = df["blip_nplanes"].to_numpy()
    all_blip_touchtrk      = df["blip_touchtrk"].to_numpy()
    all_blip_pl2_bydeadwire = df["blip_pl2_bydeadwire"].to_numpy()
    all_blip_proxtrkdist   = df["blip_proxtrkdist"].to_numpy()

    sphere_n                            = []
    sphere_sumE                         = []
    no_shower_cone_n                    = []
    no_shower_cone_sumE                 = []
    no_shower_cone_no_backtrack_cones_n    = []
    no_shower_cone_no_backtrack_cones_sumE = []
    backtrack_cones_n                   = []
    backtrack_cones_sumE                = []

    sphere_proton_n                            = []
    sphere_proton_sumE                         = []
    no_shower_cone_proton_n                    = []
    no_shower_cone_proton_sumE                 = []
    no_shower_cone_no_backtrack_cones_proton_n    = []
    no_shower_cone_no_backtrack_cones_proton_sumE = []
    backtrack_cones_proton_n                   = []
    backtrack_cones_proton_sumE                = []

    sphere_nonproton_n                            = []
    sphere_nonproton_sumE                         = []
    no_shower_cone_nonproton_n                    = []
    no_shower_cone_nonproton_sumE                 = []
    no_shower_cone_no_backtrack_cones_nonproton_n    = []
    no_shower_cone_no_backtrack_cones_nonproton_sumE = []
    backtrack_cones_nonproton_n                   = []
    backtrack_cones_nonproton_sumE                = []

    for event_index in tqdm(range(len(df)), desc="Counting blips in sphere, no-shower-cone, and backtrack-cone regions", mininterval=10):

        blip_xs       = all_blip_x[event_index]
        blip_ys       = all_blip_y[event_index]
        blip_zs       = all_blip_z[event_index]
        blip_energies = all_blip_energy[event_index]
        blip_dxs_ev   = all_blip_dx[event_index]
        blip_dws_ev   = all_blip_dw[event_index]
        blip_nplanes_ev       = all_blip_nplanes[event_index]
        blip_touchtrk_ev      = all_blip_touchtrk[event_index]
        blip_pl2_bydeadwire_ev = all_blip_pl2_bydeadwire[event_index]
        blip_proxtrkdist_ev   = all_blip_proxtrkdist[event_index]
        ev_filetype = filetype[event_index]

        ev_sphere_n                            = 0;  ev_sphere_sumE                         = 0.0
        ev_no_shower_cone_n                    = 0;  ev_no_shower_cone_sumE                 = 0.0
        ev_no_shower_cone_no_backtrack_cones_n = 0;  ev_no_shower_cone_no_backtrack_cones_sumE = 0.0
        ev_backtrack_cones_n                   = 0;  ev_backtrack_cones_sumE                = 0.0

        ev_sphere_proton_n                            = 0;  ev_sphere_proton_sumE                         = 0.0
        ev_no_shower_cone_proton_n                    = 0;  ev_no_shower_cone_proton_sumE                 = 0.0
        ev_no_shower_cone_no_backtrack_cones_proton_n = 0;  ev_no_shower_cone_no_backtrack_cones_proton_sumE = 0.0
        ev_backtrack_cones_proton_n                   = 0;  ev_backtrack_cones_proton_sumE                = 0.0

        ev_sphere_nonproton_n                            = 0;  ev_sphere_nonproton_sumE                         = 0.0
        ev_no_shower_cone_nonproton_n                    = 0;  ev_no_shower_cone_nonproton_sumE                 = 0.0
        ev_no_shower_cone_no_backtrack_cones_nonproton_n = 0;  ev_no_shower_cone_no_backtrack_cones_nonproton_sumE = 0.0
        ev_backtrack_cones_nonproton_n                   = 0;  ev_backtrack_cones_nonproton_sumE                = 0.0

        def _append_defaults():
            sphere_n.append(ev_sphere_n);                            sphere_sumE.append(ev_sphere_sumE)
            no_shower_cone_n.append(ev_no_shower_cone_n);            no_shower_cone_sumE.append(ev_no_shower_cone_sumE)
            no_shower_cone_no_backtrack_cones_n.append(ev_no_shower_cone_no_backtrack_cones_n); no_shower_cone_no_backtrack_cones_sumE.append(ev_no_shower_cone_no_backtrack_cones_sumE)
            backtrack_cones_n.append(ev_backtrack_cones_n);          backtrack_cones_sumE.append(ev_backtrack_cones_sumE)
            sphere_proton_n.append(ev_sphere_proton_n);              sphere_proton_sumE.append(ev_sphere_proton_sumE)
            no_shower_cone_proton_n.append(ev_no_shower_cone_proton_n); no_shower_cone_proton_sumE.append(ev_no_shower_cone_proton_sumE)
            no_shower_cone_no_backtrack_cones_proton_n.append(ev_no_shower_cone_no_backtrack_cones_proton_n); no_shower_cone_no_backtrack_cones_proton_sumE.append(ev_no_shower_cone_no_backtrack_cones_proton_sumE)
            backtrack_cones_proton_n.append(ev_backtrack_cones_proton_n); backtrack_cones_proton_sumE.append(ev_backtrack_cones_proton_sumE)
            sphere_nonproton_n.append(ev_sphere_nonproton_n);        sphere_nonproton_sumE.append(ev_sphere_nonproton_sumE)
            no_shower_cone_nonproton_n.append(ev_no_shower_cone_nonproton_n); no_shower_cone_nonproton_sumE.append(ev_no_shower_cone_nonproton_sumE)
            no_shower_cone_no_backtrack_cones_nonproton_n.append(ev_no_shower_cone_no_backtrack_cones_nonproton_n); no_shower_cone_no_backtrack_cones_nonproton_sumE.append(ev_no_shower_cone_no_backtrack_cones_nonproton_sumE)
            backtrack_cones_nonproton_n.append(ev_backtrack_cones_nonproton_n); backtrack_cones_nonproton_sumE.append(ev_backtrack_cones_nonproton_sumE)

        # Need valid shower momentum and shower vertex
        if (isinstance(all_wc_reco_shower_momentum[event_index], float) or
                not hasattr(all_wc_reco_shower_momentum[event_index], '__getitem__') or
                blip_xs is None or len(blip_xs) == 0):
            _append_defaults()
            continue

        sh_mom = np.array([all_wc_reco_shower_momentum[event_index][0],
                           all_wc_reco_shower_momentum[event_index][1],
                           all_wc_reco_shower_momentum[event_index][2]])
        sh_mom_mag = np.linalg.norm(sh_mom)
        if sh_mom_mag < 1e-10:
            _append_defaults()
            continue

        sh_mom_unit = sh_mom / sh_mom_mag
        sh_vtx = np.array([all_wc_reco_shower_vtx_x[event_index],
                           all_wc_reco_shower_vtx_y[event_index],
                           all_wc_reco_shower_vtx_z[event_index]])

        for blip_index in range(len(blip_xs)):
            energy = blip_energies[blip_index]
            x = blip_xs[blip_index]
            y = blip_ys[blip_index]
            z = blip_zs[blip_index]

            # Sphere quality cuts
            if blip_nplanes_ev[blip_index] <= 1:
                continue
            if blip_touchtrk_ev[blip_index] != 0:
                continue
            if blip_pl2_bydeadwire_ev[blip_index] != 0:
                continue
            if blip_proxtrkdist_ev[blip_index] <= 15.0:
                continue
            if not is_blip_in_tpc(x, y, z):
                continue

            vtx_to_blip = np.array([x - sh_vtx[0], y - sh_vtx[1], z - sh_vtx[2]])
            dist2vtx = np.linalg.norm(vtx_to_blip)
            if dist2vtx >= SPHERE_RADIUS_CM:
                continue

            # Passed sphere selection
            is_proton = reco_proton_blip(energy, blip_dxs_ev[blip_index], blip_dws_ev[blip_index], ev_filetype)
            ev_sphere_n    += 1
            ev_sphere_sumE += energy
            if is_proton:
                ev_sphere_proton_n    += 1
                ev_sphere_proton_sumE += energy
            else:
                ev_sphere_nonproton_n    += 1
                ev_sphere_nonproton_sumE += energy

            blip_xyz = np.array([x, y, z])
            if not is_within_sphere_outside_conic(sh_vtx, sh_mom_unit, blip_xyz):
                continue

            # Passed no_shower_cone selection
            ev_no_shower_cone_n    += 1
            ev_no_shower_cone_sumE += energy
            if is_proton:
                ev_no_shower_cone_proton_n    += 1
                ev_no_shower_cone_proton_sumE += energy
            else:
                ev_no_shower_cone_nonproton_n    += 1
                ev_no_shower_cone_nonproton_sumE += energy

            cos_angle_sh = np.dot(vtx_to_blip / dist2vtx, sh_mom_unit)
            if is_backtracked_blip(dist2vtx, cos_angle_sh):
                ev_backtrack_cones_n    += 1
                ev_backtrack_cones_sumE += energy
                if is_proton:
                    ev_backtrack_cones_proton_n    += 1
                    ev_backtrack_cones_proton_sumE += energy
                else:
                    ev_backtrack_cones_nonproton_n    += 1
                    ev_backtrack_cones_nonproton_sumE += energy
            else:
                ev_no_shower_cone_no_backtrack_cones_n    += 1
                ev_no_shower_cone_no_backtrack_cones_sumE += energy
                if is_proton:
                    ev_no_shower_cone_no_backtrack_cones_proton_n    += 1
                    ev_no_shower_cone_no_backtrack_cones_proton_sumE += energy
                else:
                    ev_no_shower_cone_no_backtrack_cones_nonproton_n    += 1
                    ev_no_shower_cone_no_backtrack_cones_nonproton_sumE += energy

        _append_defaults()

    df["blip_sphere_n"]                            = sphere_n
    df["blip_sphere_sumE"]                         = sphere_sumE
    df["blip_no_shower_cone_n"]                    = no_shower_cone_n
    df["blip_no_shower_cone_sumE"]                 = no_shower_cone_sumE
    df["blip_no_shower_cone_no_backtrack_cones_n"]    = no_shower_cone_no_backtrack_cones_n
    df["blip_no_shower_cone_no_backtrack_cones_sumE"] = no_shower_cone_no_backtrack_cones_sumE
    df["blip_backtrack_cones_n"]                   = backtrack_cones_n
    df["blip_backtrack_cones_sumE"]                = backtrack_cones_sumE

    df["blip_sphere_proton_n"]                            = sphere_proton_n
    df["blip_sphere_proton_sumE"]                         = sphere_proton_sumE
    df["blip_no_shower_cone_proton_n"]                    = no_shower_cone_proton_n
    df["blip_no_shower_cone_proton_sumE"]                 = no_shower_cone_proton_sumE
    df["blip_no_shower_cone_no_backtrack_cones_proton_n"]    = no_shower_cone_no_backtrack_cones_proton_n
    df["blip_no_shower_cone_no_backtrack_cones_proton_sumE"] = no_shower_cone_no_backtrack_cones_proton_sumE
    df["blip_backtrack_cones_proton_n"]                   = backtrack_cones_proton_n
    df["blip_backtrack_cones_proton_sumE"]                = backtrack_cones_proton_sumE

    df["blip_sphere_nonproton_n"]                            = sphere_nonproton_n
    df["blip_sphere_nonproton_sumE"]                         = sphere_nonproton_sumE
    df["blip_no_shower_cone_nonproton_n"]                    = no_shower_cone_nonproton_n
    df["blip_no_shower_cone_nonproton_sumE"]                 = no_shower_cone_nonproton_sumE
    df["blip_no_shower_cone_no_backtrack_cones_nonproton_n"]    = no_shower_cone_no_backtrack_cones_nonproton_n
    df["blip_no_shower_cone_no_backtrack_cones_nonproton_sumE"] = no_shower_cone_no_backtrack_cones_nonproton_sumE
    df["blip_backtrack_cones_nonproton_n"]                   = backtrack_cones_nonproton_n
    df["blip_backtrack_cones_nonproton_sumE"]                = backtrack_cones_nonproton_sumE

    # -----------------------------------------
    # Part D: Two-shower event variables (multiple vertex hypotheses)
    #
    # Identifies the two highest-energy reco showers (wc_reco_pdg == 11) per event.
    # For each vertex hypothesis {wcvtx, lanternvtx, gleevtx, pandoravtx}:
    #   Exclusion cones: apex at the given vertex, direction = (shower_start - vtx) normalized
    #   (falls back to shower momentum direction if that distance is ~zero), 45-deg half-angle.
    # Same blip quality cuts as Part C.
    #
    # Outputs (default -1 if < 2 showers found or vertex is invalid):
    #   blip_2shwr_{vtx}_no_shower_cones_{n,sumE}           — 75cm sphere around vtx, outside both cones (apex at vtx)
    #   blip_2shwr_{vtx}_no_shower_cones_{n,sumE}_notwithin_{1,3,10}cm — same, excluding blips within R cm of vtx
    #   blip_2shwr_{vtx}_no_shower_cones_{proton,nonproton}_{n,sumE}
    #   blip_2shwr_{vtx}_no_gapped_shower_cones_{n,sumE}   — 75cm sphere, outside both cones (apex at each shower start)
    #   blip_2shwr_{vtx}_no_gapped_shower_cones_{n,sumE}_notwithin_{1,3,10}cm — same, excluding blips within R cm of vtx
    #   blip_2shwr_{vtx}_no_gapped_shower_cones_{proton,nonproton}_{n,sumE}
    #   blip_2shwr_{vtx}_{nWithin,sumE}_{1,3,10}cm          — small spheres, no cone exclusion
    #   blip_2shwr_{vtx}_{proton,nonproton}_{nWithin,sumE}_{1,3,10}cm
    #   blip_2shwr_min_{vtx}_to_shower_vtx_dist
    # where {vtx} in {wcvtx, lanternvtx, gleevtx, pandoravtx}
    # -----------------------------------------
    all_wc_reco_pdg           = df["wc_reco_pdg"].to_numpy()
    all_wc_reco_startMomentum = df["wc_reco_startMomentum"].to_numpy()
    all_wc_reco_startXYZT     = df["wc_reco_startXYZT"].to_numpy()

    vtx_defs_2shwr = {
        "wcvtx":    ("wc_reco_nuvtxX",            "wc_reco_nuvtxY",            "wc_reco_nuvtxZ"),
        "wcshwrvtx":  ("wc_reco_showervtxX",         "wc_reco_showervtxY",        "wc_reco_showervtxZ"),
        "wc2shwvtx":  ("wc_2shw_vtx_x",             "wc_2shw_vtx_y",             "wc_2shw_vtx_z"),
        "lanternvtx": ("lantern_vtxX",               "lantern_vtxY",              "lantern_vtxZ"),
        "gleevtx":    ("glee_reco_vertex_x",         "glee_reco_vertex_y",        "glee_reco_vertex_z"),
        "pandoravtx": ("pandora_reco_nu_vtx_sce_x",  "pandora_reco_nu_vtx_sce_y", "pandora_reco_nu_vtx_sce_z"),
    }
    vtx_arrays_2shwr = {k: (df[cx].to_numpy(), df[cy].to_numpy(), df[cz].to_numpy())
                        for k, (cx, cy, cz) in vtx_defs_2shwr.items()}

    proton_radii_2shwr = [1, 3, 10]

    two_shwr_no_shower_cones_n           = {k: [] for k in vtx_defs_2shwr}
    two_shwr_no_shower_cones_sumE        = {k: [] for k in vtx_defs_2shwr}
    two_shwr_no_shower_cones_proton_n    = {k: [] for k in vtx_defs_2shwr}
    two_shwr_no_shower_cones_proton_sumE = {k: [] for k in vtx_defs_2shwr}
    two_shwr_no_shower_cones_nonproton_n    = {k: [] for k in vtx_defs_2shwr}
    two_shwr_no_shower_cones_nonproton_sumE = {k: [] for k in vtx_defs_2shwr}
    two_shwr_no_shower_cones_notwithin_n    = {k: {R: [] for R in proton_radii_2shwr} for k in vtx_defs_2shwr}
    two_shwr_no_shower_cones_notwithin_sumE = {k: {R: [] for R in proton_radii_2shwr} for k in vtx_defs_2shwr}

    two_shwr_no_gapped_shower_cones_n           = {k: [] for k in vtx_defs_2shwr}
    two_shwr_no_gapped_shower_cones_sumE        = {k: [] for k in vtx_defs_2shwr}
    two_shwr_no_gapped_shower_cones_proton_n    = {k: [] for k in vtx_defs_2shwr}
    two_shwr_no_gapped_shower_cones_proton_sumE = {k: [] for k in vtx_defs_2shwr}
    two_shwr_no_gapped_shower_cones_nonproton_n    = {k: [] for k in vtx_defs_2shwr}
    two_shwr_no_gapped_shower_cones_nonproton_sumE = {k: [] for k in vtx_defs_2shwr}
    two_shwr_no_gapped_shower_cones_notwithin_n    = {k: {R: [] for R in proton_radii_2shwr} for k in vtx_defs_2shwr}
    two_shwr_no_gapped_shower_cones_notwithin_sumE = {k: {R: [] for R in proton_radii_2shwr} for k in vtx_defs_2shwr}

    two_shwr_nWithin           = {k: {R: [] for R in proton_radii_2shwr} for k in vtx_defs_2shwr}
    two_shwr_sumE              = {k: {R: [] for R in proton_radii_2shwr} for k in vtx_defs_2shwr}
    two_shwr_proton_nWithin    = {k: {R: [] for R in proton_radii_2shwr} for k in vtx_defs_2shwr}
    two_shwr_proton_sumE       = {k: {R: [] for R in proton_radii_2shwr} for k in vtx_defs_2shwr}
    two_shwr_nonproton_nWithin = {k: {R: [] for R in proton_radii_2shwr} for k in vtx_defs_2shwr}
    two_shwr_nonproton_sumE    = {k: {R: [] for R in proton_radii_2shwr} for k in vtx_defs_2shwr}

    two_shwr_min_vtx_to_shower_vtx_dist = {k: [] for k in vtx_defs_2shwr}

    def _append_2shwr_defaults_for_vtx(k):
        two_shwr_no_shower_cones_n[k].append(-1)
        two_shwr_no_shower_cones_sumE[k].append(-1)
        two_shwr_no_shower_cones_proton_n[k].append(-1)
        two_shwr_no_shower_cones_proton_sumE[k].append(-1)
        two_shwr_no_shower_cones_nonproton_n[k].append(-1)
        two_shwr_no_shower_cones_nonproton_sumE[k].append(-1)
        for R in proton_radii_2shwr:
            two_shwr_no_shower_cones_notwithin_n[k][R].append(-1)
            two_shwr_no_shower_cones_notwithin_sumE[k][R].append(-1)
        two_shwr_no_gapped_shower_cones_n[k].append(-1)
        two_shwr_no_gapped_shower_cones_sumE[k].append(-1)
        two_shwr_no_gapped_shower_cones_proton_n[k].append(-1)
        two_shwr_no_gapped_shower_cones_proton_sumE[k].append(-1)
        two_shwr_no_gapped_shower_cones_nonproton_n[k].append(-1)
        two_shwr_no_gapped_shower_cones_nonproton_sumE[k].append(-1)
        for R in proton_radii_2shwr:
            two_shwr_no_gapped_shower_cones_notwithin_n[k][R].append(-1)
            two_shwr_no_gapped_shower_cones_notwithin_sumE[k][R].append(-1)
        for R in proton_radii_2shwr:
            two_shwr_nWithin[k][R].append(-1)
            two_shwr_sumE[k][R].append(-1)
            two_shwr_proton_nWithin[k][R].append(-1)
            two_shwr_proton_sumE[k][R].append(-1)
            two_shwr_nonproton_nWithin[k][R].append(-1)
            two_shwr_nonproton_sumE[k][R].append(-1)
        two_shwr_min_vtx_to_shower_vtx_dist[k].append(-1)

    for event_index in tqdm(range(len(df)), desc="Two-shower blip variables", mininterval=10):

        pdg_arr = all_wc_reco_pdg[event_index]
        if isinstance(pdg_arr, float) or not hasattr(pdg_arr, '__getitem__'):
            for k in vtx_defs_2shwr:
                _append_2shwr_defaults_for_vtx(k)
            continue

        mom_arr  = all_wc_reco_startMomentum[event_index]
        xyzt_arr = all_wc_reco_startXYZT[event_index]

        # Find the two highest-energy showers (pdg == 11)
        shower_indices = [j for j in range(len(pdg_arr)) if pdg_arr[j] == 11]
        if len(shower_indices) < 2:
            for k in vtx_defs_2shwr:
                _append_2shwr_defaults_for_vtx(k)
            continue

        shower_indices.sort(key=lambda j: mom_arr[j][3], reverse=True)
        top2 = shower_indices[:2]

        # Shower start positions (same for all vertex hypotheses)
        shower_start_xyzs = [np.array([xyzt_arr[j][0], xyzt_arr[j][1], xyzt_arr[j][2]], dtype=np.float64) for j in top2]
        if not all(np.all(np.isfinite(xyz)) for xyz in shower_start_xyzs):
            for k in vtx_defs_2shwr:
                _append_2shwr_defaults_for_vtx(k)
            continue

        # Pre-filter blips by quality cuts (independent of vertex)
        blip_xs           = all_blip_x[event_index]
        blip_ys           = all_blip_y[event_index]
        blip_zs           = all_blip_z[event_index]
        blip_energies     = all_blip_energy[event_index]
        blip_dxs_ev       = all_blip_dx[event_index]
        blip_dws_ev       = all_blip_dw[event_index]
        blip_nplanes_ev        = all_blip_nplanes[event_index]
        blip_touchtrk_ev       = all_blip_touchtrk[event_index]
        blip_pl2_bydeadwire_ev = all_blip_pl2_bydeadwire[event_index]
        blip_proxtrkdist_ev    = all_blip_proxtrkdist[event_index]
        ev_filetype = filetype[event_index]

        passing_blips = []  # list of (blip_xyz, energy, is_proton)
        if blip_xs is not None and len(blip_xs) > 0:
            for blip_index in range(len(blip_xs)):
                if blip_nplanes_ev[blip_index] <= 1:
                    continue
                if blip_touchtrk_ev[blip_index] != 0:
                    continue
                if blip_pl2_bydeadwire_ev[blip_index] != 0:
                    continue
                if blip_proxtrkdist_ev[blip_index] <= 15.0:
                    continue
                bx = blip_xs[blip_index]
                by = blip_ys[blip_index]
                bz = blip_zs[blip_index]
                if not is_blip_in_tpc(bx, by, bz):
                    continue
                energy = blip_energies[blip_index]
                is_proton = reco_proton_blip(energy, blip_dxs_ev[blip_index], blip_dws_ev[blip_index], ev_filetype)
                passing_blips.append((np.array([bx, by, bz], dtype=np.float64), energy, is_proton))

        # Process each vertex hypothesis
        for k, (vx_arr, vy_arr, vz_arr) in vtx_arrays_2shwr.items():
            vx = vx_arr[event_index]
            vy = vy_arr[event_index]
            vz = vz_arr[event_index]
            if not (np.isfinite(vx) and np.isfinite(vy) and np.isfinite(vz)):
                _append_2shwr_defaults_for_vtx(k)
                continue

            nu_vtx = np.array([vx, vy, vz], dtype=np.float64)

            # Cone directions from this vertex to each shower start
            cone_dirs = []
            for idx, j in enumerate(top2):
                nu_to_sh = shower_start_xyzs[idx] - nu_vtx
                dist_nu_to_sh = np.linalg.norm(nu_to_sh)
                if dist_nu_to_sh > 1e-10:
                    cone_dirs.append(nu_to_sh / dist_nu_to_sh)
                else:
                    sh_mom = np.array([mom_arr[j][0], mom_arr[j][1], mom_arr[j][2]], dtype=np.float64)
                    sh_mom_mag = np.linalg.norm(sh_mom) if np.all(np.isfinite(sh_mom)) else 0.0
                    if sh_mom_mag > 1e-10:
                        cone_dirs.append(sh_mom / sh_mom_mag)
                    else:
                        cone_dirs.append(np.array([0.0, 0.0, 1.0]))  # degenerate fallback

            min_dist = min(np.linalg.norm(shower_start_xyzs[0] - nu_vtx),
                           np.linalg.norm(shower_start_xyzs[1] - nu_vtx))

            ev_no_shower_cones_n          = 0; ev_no_shower_cones_sumE          = 0.0
            ev_no_shower_cones_proton_n   = 0; ev_no_shower_cones_proton_sumE   = 0.0
            ev_no_shower_cones_nonproton_n = 0; ev_no_shower_cones_nonproton_sumE = 0.0
            ev_no_shower_cones_notwithin_n    = {R: 0   for R in proton_radii_2shwr}
            ev_no_shower_cones_notwithin_sumE = {R: 0.0 for R in proton_radii_2shwr}
            ev_no_gapped_shower_cones_n          = 0; ev_no_gapped_shower_cones_sumE          = 0.0
            ev_no_gapped_shower_cones_proton_n   = 0; ev_no_gapped_shower_cones_proton_sumE   = 0.0
            ev_no_gapped_shower_cones_nonproton_n = 0; ev_no_gapped_shower_cones_nonproton_sumE = 0.0
            ev_no_gapped_shower_cones_notwithin_n    = {R: 0   for R in proton_radii_2shwr}
            ev_no_gapped_shower_cones_notwithin_sumE = {R: 0.0 for R in proton_radii_2shwr}
            ev_nWithin           = {R: 0   for R in proton_radii_2shwr}
            ev_sumE              = {R: 0.0 for R in proton_radii_2shwr}
            ev_proton_nWithin    = {R: 0   for R in proton_radii_2shwr}
            ev_proton_sumE       = {R: 0.0 for R in proton_radii_2shwr}
            ev_nonproton_nWithin = {R: 0   for R in proton_radii_2shwr}
            ev_nonproton_sumE    = {R: 0.0 for R in proton_radii_2shwr}

            for blip_xyz, energy, is_proton in passing_blips:
                dist_to_nu = np.linalg.norm(blip_xyz - nu_vtx)

                # Small spheres around vertex (no cone exclusion)
                for R in proton_radii_2shwr:
                    if dist_to_nu < R:
                        ev_nWithin[R] += 1
                        ev_sumE[R] += energy
                        if is_proton:
                            ev_proton_nWithin[R] += 1
                            ev_proton_sumE[R] += energy
                        else:
                            ev_nonproton_nWithin[R] += 1
                            ev_nonproton_sumE[R] += energy

                # 75cm sphere cuts (shared prerequisite for both cone variants)
                if dist_to_nu < SPHERE_RADIUS_CM:
                    # Outside both shower cones (apex at nu vertex)
                    if not (is_in_cone(nu_vtx, cone_dirs[0], blip_xyz) or
                            is_in_cone(nu_vtx, cone_dirs[1], blip_xyz)):
                        ev_no_shower_cones_n    += 1
                        ev_no_shower_cones_sumE += energy
                        if is_proton:
                            ev_no_shower_cones_proton_n    += 1
                            ev_no_shower_cones_proton_sumE += energy
                        else:
                            ev_no_shower_cones_nonproton_n    += 1
                            ev_no_shower_cones_nonproton_sumE += energy
                        for R in proton_radii_2shwr:
                            if dist_to_nu >= R:
                                ev_no_shower_cones_notwithin_n[R]    += 1
                                ev_no_shower_cones_notwithin_sumE[R] += energy

                    # Outside both gapped shower cones (apex at each shower start)
                    if not (is_in_cone(shower_start_xyzs[0], cone_dirs[0], blip_xyz) or
                            is_in_cone(shower_start_xyzs[1], cone_dirs[1], blip_xyz)):
                        ev_no_gapped_shower_cones_n    += 1
                        ev_no_gapped_shower_cones_sumE += energy
                        if is_proton:
                            ev_no_gapped_shower_cones_proton_n    += 1
                            ev_no_gapped_shower_cones_proton_sumE += energy
                        else:
                            ev_no_gapped_shower_cones_nonproton_n    += 1
                            ev_no_gapped_shower_cones_nonproton_sumE += energy
                        for R in proton_radii_2shwr:
                            if dist_to_nu >= R:
                                ev_no_gapped_shower_cones_notwithin_n[R]    += 1
                                ev_no_gapped_shower_cones_notwithin_sumE[R] += energy

            two_shwr_no_shower_cones_n[k].append(ev_no_shower_cones_n)
            two_shwr_no_shower_cones_sumE[k].append(ev_no_shower_cones_sumE)
            two_shwr_no_shower_cones_proton_n[k].append(ev_no_shower_cones_proton_n)
            two_shwr_no_shower_cones_proton_sumE[k].append(ev_no_shower_cones_proton_sumE)
            two_shwr_no_shower_cones_nonproton_n[k].append(ev_no_shower_cones_nonproton_n)
            two_shwr_no_shower_cones_nonproton_sumE[k].append(ev_no_shower_cones_nonproton_sumE)
            for R in proton_radii_2shwr:
                two_shwr_no_shower_cones_notwithin_n[k][R].append(ev_no_shower_cones_notwithin_n[R])
                two_shwr_no_shower_cones_notwithin_sumE[k][R].append(ev_no_shower_cones_notwithin_sumE[R])
            two_shwr_no_gapped_shower_cones_n[k].append(ev_no_gapped_shower_cones_n)
            two_shwr_no_gapped_shower_cones_sumE[k].append(ev_no_gapped_shower_cones_sumE)
            two_shwr_no_gapped_shower_cones_proton_n[k].append(ev_no_gapped_shower_cones_proton_n)
            two_shwr_no_gapped_shower_cones_proton_sumE[k].append(ev_no_gapped_shower_cones_proton_sumE)
            two_shwr_no_gapped_shower_cones_nonproton_n[k].append(ev_no_gapped_shower_cones_nonproton_n)
            two_shwr_no_gapped_shower_cones_nonproton_sumE[k].append(ev_no_gapped_shower_cones_nonproton_sumE)
            for R in proton_radii_2shwr:
                two_shwr_no_gapped_shower_cones_notwithin_n[k][R].append(ev_no_gapped_shower_cones_notwithin_n[R])
                two_shwr_no_gapped_shower_cones_notwithin_sumE[k][R].append(ev_no_gapped_shower_cones_notwithin_sumE[R])
            for R in proton_radii_2shwr:
                two_shwr_nWithin[k][R].append(ev_nWithin[R])
                two_shwr_sumE[k][R].append(ev_sumE[R])
                two_shwr_proton_nWithin[k][R].append(ev_proton_nWithin[R])
                two_shwr_proton_sumE[k][R].append(ev_proton_sumE[R])
                two_shwr_nonproton_nWithin[k][R].append(ev_nonproton_nWithin[R])
                two_shwr_nonproton_sumE[k][R].append(ev_nonproton_sumE[R])
            two_shwr_min_vtx_to_shower_vtx_dist[k].append(min_dist)

    new_cols = {}
    for k in vtx_defs_2shwr:
        new_cols[f"blip_2shwr_{k}_no_shower_cones_n"]             = two_shwr_no_shower_cones_n[k]
        new_cols[f"blip_2shwr_{k}_no_shower_cones_sumE"]           = two_shwr_no_shower_cones_sumE[k]
        new_cols[f"blip_2shwr_{k}_no_shower_cones_proton_n"]       = two_shwr_no_shower_cones_proton_n[k]
        new_cols[f"blip_2shwr_{k}_no_shower_cones_proton_sumE"]    = two_shwr_no_shower_cones_proton_sumE[k]
        new_cols[f"blip_2shwr_{k}_no_shower_cones_nonproton_n"]    = two_shwr_no_shower_cones_nonproton_n[k]
        new_cols[f"blip_2shwr_{k}_no_shower_cones_nonproton_sumE"] = two_shwr_no_shower_cones_nonproton_sumE[k]
        for R in proton_radii_2shwr:
            new_cols[f"blip_2shwr_{k}_no_shower_cones_n_notwithin_{R}cm"]    = two_shwr_no_shower_cones_notwithin_n[k][R]
            new_cols[f"blip_2shwr_{k}_no_shower_cones_sumE_notwithin_{R}cm"] = two_shwr_no_shower_cones_notwithin_sumE[k][R]
        new_cols[f"blip_2shwr_{k}_no_gapped_shower_cones_n"]             = two_shwr_no_gapped_shower_cones_n[k]
        new_cols[f"blip_2shwr_{k}_no_gapped_shower_cones_sumE"]          = two_shwr_no_gapped_shower_cones_sumE[k]
        new_cols[f"blip_2shwr_{k}_no_gapped_shower_cones_proton_n"]      = two_shwr_no_gapped_shower_cones_proton_n[k]
        new_cols[f"blip_2shwr_{k}_no_gapped_shower_cones_proton_sumE"]   = two_shwr_no_gapped_shower_cones_proton_sumE[k]
        new_cols[f"blip_2shwr_{k}_no_gapped_shower_cones_nonproton_n"]   = two_shwr_no_gapped_shower_cones_nonproton_n[k]
        new_cols[f"blip_2shwr_{k}_no_gapped_shower_cones_nonproton_sumE"] = two_shwr_no_gapped_shower_cones_nonproton_sumE[k]
        for R in proton_radii_2shwr:
            new_cols[f"blip_2shwr_{k}_no_gapped_shower_cones_n_notwithin_{R}cm"]    = two_shwr_no_gapped_shower_cones_notwithin_n[k][R]
            new_cols[f"blip_2shwr_{k}_no_gapped_shower_cones_sumE_notwithin_{R}cm"] = two_shwr_no_gapped_shower_cones_notwithin_sumE[k][R]
        for R in proton_radii_2shwr:
            new_cols[f"blip_2shwr_{k}_nWithin_{R}cm"]           = two_shwr_nWithin[k][R]
            new_cols[f"blip_2shwr_{k}_sumE_{R}cm"]              = two_shwr_sumE[k][R]
            new_cols[f"blip_2shwr_{k}_proton_nWithin_{R}cm"]    = two_shwr_proton_nWithin[k][R]
            new_cols[f"blip_2shwr_{k}_proton_sumE_{R}cm"]       = two_shwr_proton_sumE[k][R]
            new_cols[f"blip_2shwr_{k}_nonproton_nWithin_{R}cm"] = two_shwr_nonproton_nWithin[k][R]
            new_cols[f"blip_2shwr_{k}_nonproton_sumE_{R}cm"]    = two_shwr_nonproton_sumE[k][R]
        new_cols[f"blip_2shwr_min_{k}_to_shower_vtx_dist"] = two_shwr_min_vtx_to_shower_vtx_dist[k]
    df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

    return df
