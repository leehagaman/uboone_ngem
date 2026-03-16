import numpy as np
from tqdm import tqdm


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

    return df
