import numpy as np
import pandas as pd
from tqdm import tqdm


def do_orthogonalization_and_POT_weighting(df, pot_dic, normalizing_POT=1.11e21):

    summed_POT_nc_1pi0 = pot_dic['nc_pi0_overlay'] + pot_dic['nu_overlay']

    # deleting out-FV or otherwise weird events from the NC Pi0 overlay
    nc_pi0_overlay_true_nc_1pi0_df = df.query("filetype == 'nc_pi0_overlay' and wc_truth_isCC==0 and wc_truth_NprimPio==1 and wc_truth_vtxInside == 1")
    nu_overlay_true_nc_1pi0_df = df.query("filetype == 'nu_overlay' and wc_truth_isCC==0 and wc_truth_NprimPio==1 and wc_truth_vtxInside == 1")
    nu_overlay_other_df = df.query("filetype == 'nu_overlay' and not (wc_truth_isCC==0 and wc_truth_NprimPio==1 and wc_truth_vtxInside == 1)")
    dirt_df = df.query("filetype == 'dirt_overlay'")

    # setting the POTs in order to combine the NC Pi0 overlay and nu overlay files without throwing away MC statistics
    nc_pi0_overlay_true_nc_1pi0_df["wc_file_POT"] = summed_POT_nc_1pi0
    nu_overlay_true_nc_1pi0_df["wc_file_POT"] = summed_POT_nc_1pi0

    df = pd.concat([nc_pi0_overlay_true_nc_1pi0_df, nu_overlay_true_nc_1pi0_df, nu_overlay_other_df, dirt_df])

    weight_cv_arr = df["wc_weight_cv"].to_numpy()
    weight_spline_arr = df["wc_weight_spline"].to_numpy()
    file_POTs = df["wc_file_POT"].to_numpy()
    net_weights = []
    for i in tqdm(range(len(weight_cv_arr)), desc="Adding POT weighting"):
        file_POT = file_POTs[i]
        weight_temp = weight_cv_arr[i] * weight_spline_arr[i]
        if weight_temp <= 0. or weight_temp > 30. or np.isnan(weight_temp): # something went wrong with the saved GENIE weights, set it to one
            weight_temp = 1.
        net_weights.append(weight_temp * normalizing_POT / file_POT)
    df["wc_net_weight"] = net_weights


def do_wc_postprocessing(df):

    # Extra reco variables
    proton_nums = []
    other_track_nums = []
    energy_lists = df["wc_kine_energy_particle"].to_numpy()
    pdg_lists = df["wc_kine_particle_type"].to_numpy()
    for i in tqdm(range(df.shape[0]), desc="Adding WC reco particle multiplicity"):

        proton_num = 0
        other_track_num = 0
        energy_list = energy_lists[i]

        if isinstance(energy_list, float) and np.isnan(energy_list):
            proton_nums.append(np.nan)
            other_track_nums.append(np.nan)
            continue

        pdg_list = pdg_lists[i]
        for i in range(len(energy_list)):
            if abs(pdg_list[i]) == 2212:
                if energy_list[i] > 35: # 35 MeV reco proton kinetic energy threshold
                    proton_num += 1
            if abs(pdg_list[i]) == 13 or abs(pdg_list[i]) == 211: # see N_tracks at https://github.com/BNLIF/wcp-uboone-bdt/blob/main/inc/WCPLEEANA/cuts.h
                if energy_list[i] > 10.:
                    other_track_num += 1
        proton_nums.append(proton_num)
        other_track_nums.append(other_track_num)
    df["wc_reco_num_protons"] = proton_nums
    df["wc_reco_num_other_tracks"] = other_track_nums

    # Extra truth variables
    max_prim_proton_energies = []
    truth_pdgs = df["wc_truth_pdg"].to_numpy()
    truth_mothers = df["wc_truth_mother"].to_numpy()
    truth_startMomentums = df["wc_truth_startMomentum"].to_numpy()
    for i in tqdm(range(df.shape[0]), desc="Adding WC truth primary proton energy"):
        max_prim_proton_energy = 0
        truth_pdg_list = truth_pdgs[i]
        truth_mother_list = truth_mothers[i]
        truth_startMomentum_list = truth_startMomentums[i]
        if isinstance(truth_pdg_list, float) and np.isnan(truth_pdg_list): # doesn't have a truth Geant4 particle tree
            max_prim_proton_energies.append(-1)
            continue
        for j in range(len(truth_pdg_list)):
            if truth_mother_list[j] == 0 and truth_pdg_list[j] == 2212: # primary proton
                max_prim_proton_energy = max(truth_startMomentum_list[j][3] * 1000. - 938.272089, max_prim_proton_energy)
        max_prim_proton_energies.append(max_prim_proton_energy)
    df["wc_true_max_prim_proton_energy"] = max_prim_proton_energies

    # extra primary shower position and angle variables
    shower_thetas = []
    shower_phis = []
    distances_to_boundary = []
    backwards_projected_dists = []
    reco_shower_momentum = df["wc_reco_showerMomentum"].to_numpy()
    reco_nu_vtx_x = df["wc_reco_showervtxX"].to_numpy()
    reco_nu_vtx_y = df["wc_reco_showervtxY"].to_numpy()
    reco_nu_vtx_z = df["wc_reco_showervtxZ"].to_numpy()
    for i in tqdm(range(df.shape[0]), desc="Adding WC shower position and angle variables"):

        if isinstance(reco_shower_momentum[i], float) and np.isnan(reco_shower_momentum[i]):
            shower_thetas.append(np.nan)
            shower_phis.append(np.nan)
            distances_to_boundary.append(np.nan)
            backwards_projected_dists.append(np.nan)
            continue

        reco_shower_momentum_0 = reco_shower_momentum[i][0]
        reco_shower_momentum_1 = reco_shower_momentum[i][1]
        reco_shower_momentum_2 = reco_shower_momentum[i][2]
        reco_shower_momentum_3 = reco_shower_momentum[i][3]
        if reco_shower_momentum_3 > 0:
            reco_shower_momentum_perp = np.sqrt(reco_shower_momentum_0 * reco_shower_momentum_0 + reco_shower_momentum_1 * reco_shower_momentum_1)
            shower_thetas.append(np.arctan2(reco_shower_momentum_perp, reco_shower_momentum_2) * 180. / np.pi)
            shower_phis.append(np.arctan2(reco_shower_momentum_0, reco_shower_momentum_1) * 180. / np.pi)

            shower_momentum_total_3d = np.sqrt(reco_shower_momentum_0 * reco_shower_momentum_0 + 
                                                reco_shower_momentum_1 * reco_shower_momentum_1 + 
                                                reco_shower_momentum_2 * reco_shower_momentum_2)
            shower_unit_vector_3d = [reco_shower_momentum_0 / shower_momentum_total_3d, 
                                        reco_shower_momentum_1 / shower_momentum_total_3d, 
                                        reco_shower_momentum_2 / shower_momentum_total_3d]
            center_x = 130.
            center_y = 0.
            center_z = 525.    
            towards_center_length = np.sqrt((reco_nu_vtx_x[i] - center_x) * (reco_nu_vtx_x[i] - center_x) + 
                                            (reco_nu_vtx_y[i] - center_y) * (reco_nu_vtx_y[i] - center_y) + 
                                            (reco_nu_vtx_z[i] - center_z) * (reco_nu_vtx_z[i] - center_z))
            towards_center_unit_vector_3d = [(center_x - reco_nu_vtx_x[i]) / towards_center_length, 
                                                (center_y - reco_nu_vtx_y[i]) / towards_center_length, 
                                                (center_z - reco_nu_vtx_z[i]) / towards_center_length]

            shower_momentum_total_2d = np.sqrt(reco_shower_momentum_0 * reco_shower_momentum_0 + 
                                                reco_shower_momentum_1 * reco_shower_momentum_1)
            shower_unit_vector_2d = [reco_shower_momentum_0 / shower_momentum_total_3d, 
                                        reco_shower_momentum_1 / shower_momentum_total_3d]
            towards_center_length = np.sqrt((reco_nu_vtx_x[i] - center_x) * (reco_nu_vtx_x[i] - center_x) + 
                                            (reco_nu_vtx_y[i] - center_y) * (reco_nu_vtx_y[i] - center_y))
            towards_center_unit_vector_2d = [(center_x - reco_nu_vtx_x[i]) / towards_center_length, 
                                                (center_y - reco_nu_vtx_y[i]) / towards_center_length]
            
            min_backwards_projected_dist = 1e9
                    
            # projecting to x walls
            if shower_unit_vector_3d[0] > 0:
                min_backwards_projected_dist = min(min_backwards_projected_dist, (reco_nu_vtx_x[i] - (-1)) / shower_unit_vector_3d[0])
            elif shower_unit_vector_3d[0] < 0:
                min_backwards_projected_dist = min(min_backwards_projected_dist, (reco_nu_vtx_x[i] - (254.3)) / shower_unit_vector_3d[0])
                
            # projecting to y walls
            if shower_unit_vector_3d[1] > 0:
                min_backwards_projected_dist = min(min_backwards_projected_dist, (reco_nu_vtx_y[i] - (-115.)) / shower_unit_vector_3d[1])
            elif shower_unit_vector_3d[1] < 0:
                min_backwards_projected_dist = min(min_backwards_projected_dist, (reco_nu_vtx_y[i] - (117.)) / shower_unit_vector_3d[1])
                
            # projecting to z walls
            if shower_unit_vector_3d[2] > 0:
                min_backwards_projected_dist = min(min_backwards_projected_dist, (reco_nu_vtx_z[i] - (0.6)) / shower_unit_vector_3d[2])
            elif shower_unit_vector_3d[2] < 0:
                min_backwards_projected_dist = min(min_backwards_projected_dist, (reco_nu_vtx_z[i] - (1036.4)) / shower_unit_vector_3d[2])

            backwards_projected_dists.append(min_backwards_projected_dist)
            
        else:
            shower_thetas.append(np.nan)
            shower_phis.append(np.nan)
            backwards_projected_dists.append(np.nan)
            
        distances_to_boundary.append(np.min([
            abs(reco_nu_vtx_x[i] - (-1.)),
            abs(reco_nu_vtx_x[i] - (254.3)),
            abs(reco_nu_vtx_y[i] - (-115.)),
            abs(reco_nu_vtx_y[i] - (117.)),
            abs(reco_nu_vtx_z[i] - (0.6)),
            abs(reco_nu_vtx_z[i] - (1036.4))
        ]))
    df["wc_reco_shower_theta"] = shower_thetas
    df["wc_reco_shower_phi"] = shower_phis
    df["wc_reco_distance_to_boundary"] = distances_to_boundary
    df["wc_reco_backwards_projected_dist"] = backwards_projected_dists


def do_blip_postprocessing(df):

    # adding separate electron blip, proton, and other blip arrays based on blip_true_pdg
    electron_blips_x = []
    electron_blips_y = []
    electron_blips_z = []
    electron_blips_energy = []
    proton_blips_x = []
    proton_blips_y = []
    proton_blips_z = []
    proton_blips_energy = []
    other_blips_x = []
    other_blips_y = []
    other_blips_z = []
    other_blips_energy = []

    all_blip_pdgs = df["blip_true_pdg"].to_numpy()
    all_blip_x = df["blip_x"].to_numpy()
    all_blip_y = df["blip_y"].to_numpy()
    all_blip_z = df["blip_z"].to_numpy()
    all_blip_energy = df["blip_energy"].to_numpy()
    
    for event_index in tqdm(range(len(df)), desc="Adding separate e/p/other blips"):
        electron_blips_x.append([])
        electron_blips_y.append([])
        electron_blips_z.append([])
        electron_blips_energy.append([])
        proton_blips_x.append([])
        proton_blips_y.append([])
        proton_blips_z.append([])
        proton_blips_energy.append([])
        other_blips_x.append([])
        other_blips_y.append([])
        other_blips_z.append([])
        other_blips_energy.append([])

        curr_blip_pdgs = all_blip_pdgs[event_index]
        curr_blip_x = all_blip_x[event_index]
        curr_blip_y = all_blip_y[event_index]
        curr_blip_z = all_blip_z[event_index]
        curr_blip_energy = all_blip_energy[event_index]

        if isinstance(curr_blip_pdgs, float) and np.isnan(curr_blip_pdgs):
            electron_blips_x[event_index].append([])
            electron_blips_y[event_index].append([])
            electron_blips_z[event_index].append([])
            electron_blips_energy[event_index].append([])
            proton_blips_x[event_index].append([])
            proton_blips_y[event_index].append([])
            proton_blips_z[event_index].append([])
            proton_blips_energy[event_index].append([])
            other_blips_x[event_index].append([])
            other_blips_y[event_index].append([])
            other_blips_z[event_index].append([])
            other_blips_energy[event_index].append([])
            continue

        for blip_index in range(len(curr_blip_pdgs)):
            if curr_blip_pdgs[blip_index] == 11:
                electron_blips_x[event_index].append(curr_blip_x[blip_index])
                electron_blips_y[event_index].append(curr_blip_y[blip_index])
                electron_blips_z[event_index].append(curr_blip_z[blip_index])
                electron_blips_energy[event_index].append(curr_blip_energy[blip_index])
            elif curr_blip_pdgs[blip_index] == 2212:
                proton_blips_x[event_index].append(curr_blip_x[blip_index])
                proton_blips_y[event_index].append(curr_blip_y[blip_index])
                proton_blips_z[event_index].append(curr_blip_z[blip_index])
                proton_blips_energy[event_index].append(curr_blip_energy[blip_index])
            else:
                other_blips_x[event_index].append(curr_blip_x[blip_index])
                other_blips_y[event_index].append(curr_blip_y[blip_index])
                other_blips_z[event_index].append(curr_blip_z[blip_index])
                other_blips_energy[event_index].append(curr_blip_energy[blip_index])

    df["electron_blip_x"] = electron_blips_x
    df["electron_blip_y"] = electron_blips_y
    df["electron_blip_z"] = electron_blips_z
    df["electron_blip_energy"] = electron_blips_energy
    df["proton_blip_x"] = proton_blips_x
    df["proton_blip_y"] = proton_blips_y
    df["proton_blip_z"] = proton_blips_z
    df["proton_blip_energy"] = proton_blips_energy
    df["other_blip_x"] = other_blips_x
    df["other_blip_y"] = other_blips_y
    df["other_blip_z"] = other_blips_z
    df["other_blip_energy"] = other_blips_energy
