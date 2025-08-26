import numpy as np
import pandas as pd
from tqdm import tqdm

from signal_categories import topological_category_queries, topological_category_labels
from signal_categories import physics_category_queries, physics_category_labels

def do_orthogonalization_and_POT_weighting(df, pot_dic, normalizing_POT=1.11e21):

    original_length = df.shape[0]

    summed_POT_nc_1pi0 = pot_dic['nc_pi0_overlay'] + pot_dic['nu_overlay']

    # Get masks for different event types
    nc_pi0_overlay_true_nc_1pi0_mask = (df["filetype"] == 'nc_pi0_overlay') & (df["wc_truth_isCC"] == 0) & (df["wc_truth_NprimPio"] == 1) & (df["wc_truth_vtxInside"] == 1)
    nu_overlay_true_nc_1pi0_mask = (df["filetype"] == 'nu_overlay') & (df["wc_truth_isCC"] == 0) & (df["wc_truth_NprimPio"] == 1) & (df["wc_truth_vtxInside"] == 1)
    nu_overlay_other_mask = (df["filetype"] == 'nu_overlay') & ~((df["wc_truth_isCC"] == 0) & (df["wc_truth_NprimPio"] == 1) & (df["wc_truth_vtxInside"] == 1))
    dirt_mask = df["filetype"] == 'dirt_overlay'
    ext_mask = df["filetype"] == 'ext'

    # setting the POTs in order to combine the NC Pi0 overlay and nu overlay files without throwing away MC statistics
    df.loc[nc_pi0_overlay_true_nc_1pi0_mask, "wc_file_POT"] = summed_POT_nc_1pi0
    df.loc[nu_overlay_true_nc_1pi0_mask, "wc_file_POT"] = summed_POT_nc_1pi0


    # Filter out unwanted events by keeping only the events we want
    combined_mask = nc_pi0_overlay_true_nc_1pi0_mask | nu_overlay_true_nc_1pi0_mask | nu_overlay_other_mask | dirt_mask | ext_mask
    
    # Use boolean indexing instead of drop for more reliable filtering
    df = df[combined_mask].copy()
    df.reset_index(drop=True, inplace=True)

    weight_cv_arr = df["wc_weight_cv"].to_numpy()
    weight_spline_arr = df["wc_weight_spline"].to_numpy()
    file_POTs = df["wc_file_POT"].to_numpy()
    net_weights = []
    for i in tqdm(range(len(weight_cv_arr)), desc="Adding POT weighting"):
        file_POT = file_POTs[i]
        weight_temp = weight_cv_arr[i] * weight_spline_arr[i]
        if weight_temp <= 0. or weight_temp > 30. or np.isnan(weight_temp) or np.isinf(weight_temp): # something went wrong with the saved GENIE weights, set it to one
            weight_temp = 1.
        net_weights.append(weight_temp * normalizing_POT / file_POT)

    df["wc_net_weight"] = net_weights

    final_length = df.shape[0]

    print(f"When combining different file types, went from {original_length} to {final_length} events")

    return df


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

    return df


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

    return df

def add_extra_true_photon_variables(df):

    true_num_gamma = []
    true_gamma_energies = []
    true_gamma_pairconversion_xs = []
    true_gamma_pairconversion_ys = []
    true_gamma_pairconversion_zs = []
    true_num_gamma_pairconvert = []
    true_num_gamma_pairconvert_in_FV = []
    true_num_gamma_pairconvert_in_FV_20_MeV = []

    truth_pdg_arr = df["wc_truth_pdg"].to_numpy()
    truth_id_arr = df["wc_truth_id"].to_numpy()
    truth_mother_arr = df["wc_truth_mother"].to_numpy()
    truth_startMomentum_arr = df["wc_truth_startMomentum"].to_numpy()
    truth_startXYZT_arr = df["wc_truth_startXYZT"].to_numpy()

    num_infinite_loops_broken = 0

    for event_i in tqdm(range(df.shape[0]), desc="Adding true photon variables"):

        if isinstance(truth_id_arr[event_i], float) and np.isnan(truth_id_arr[event_i]):
            true_num_gamma.append(np.nan)
            true_gamma_energies.append(np.nan)
            true_gamma_pairconversion_xs.append(np.nan)
            true_gamma_pairconversion_ys.append(np.nan)
            true_gamma_pairconversion_zs.append(np.nan)
            true_num_gamma_pairconvert.append(np.nan)
            true_num_gamma_pairconvert_in_FV.append(np.nan)
            true_num_gamma_pairconvert_in_FV_20_MeV.append(np.nan)
            continue

        num_particles = len(truth_id_arr[event_i])
        
        curr_true_num_gamma = 0
        curr_true_gamma_energies = []
        curr_true_gamma_pairconversion_xs = []
        curr_true_gamma_pairconversion_ys = []
        curr_true_gamma_pairconversion_zs = []
        curr_true_num_gamma_pairconvert = 0
        curr_true_num_gamma_pairconvert_in_FV = 0
        curr_true_num_gamma_pairconvert_in_FV_20_MeV = 0

        pi0_ids = []
        for i in range(num_particles):
            if truth_pdg_arr[event_i][i] == 111:
                pi0_ids.append(truth_id_arr[event_i][i])

        primary_or_pi0_gamma_ids = []
        for i in range(num_particles):
            if truth_mother_arr[event_i][i] in pi0_ids or truth_mother_arr[event_i][i] == 0: # this is a daughter of a pi0 or a primary particle
                if truth_pdg_arr[event_i][i] == 22: # this is a photon from a pi0 or a primary photon (most likely from an eta or Delta radiative)

                    curr_true_num_gamma += 1
                    curr_true_gamma_energies.append(truth_startMomentum_arr[event_i][i][3])
                    primary_or_pi0_gamma_ids.append(truth_id_arr[event_i][i])

        # looking for the first point where the photon transfers more than half its energy to daughter charged particles
        # should be 100% for pair production, but compton scatters can also effectively cause the start of a shower
        # daughter particles could disappear from the geant tree even if it pair converts, that type of photon won't be included here

        # Create a mapping from photon ID to its position in the energy list
        photon_id_to_position = {photon_id: pos for pos, photon_id in enumerate(primary_or_pi0_gamma_ids)}
        
        # Initialize conversion point lists with None values
        curr_true_gamma_pairconversion_xs = [None] * len(primary_or_pi0_gamma_ids)
        curr_true_gamma_pairconversion_ys = [None] * len(primary_or_pi0_gamma_ids)
        curr_true_gamma_pairconversion_zs = [None] * len(primary_or_pi0_gamma_ids)
        
        # looking for pair conversion points, allowing for the possibility of Compton scattering
        for i in range(num_particles):
            if truth_id_arr[event_i][i] in primary_or_pi0_gamma_ids: # pi0/primary -> gamma, this won't include the manually deleted photon

                original_gamma_energy = truth_startMomentum_arr[event_i][i][3]
                cumulative_deposited_energy = 0

                visited_ids = set()
                iteration_count = 0
                
                while True:
                    curr_id = truth_id_arr[event_i][i]
                    
                    if curr_id in visited_ids:
                        num_infinite_loops_broken += 1
                        break

                    visited_ids.add(curr_id)
                    iteration_count += 1
                    
                    descendants_ids = []
                    descendants_indices = []
                    descendants_pdgs = []
                    for j in range(num_particles):
                        if truth_mother_arr[event_i][j] == curr_id: # pi0/primary -> gamma -> this particle
                            descendants_ids.append(truth_id_arr[event_i][j])
                            descendants_indices.append(j)
                            descendants_pdgs.append(truth_pdg_arr[event_i][j])

                    for descendant_i in range(len(descendants_indices)):
                        if abs(descendants_pdgs[descendant_i]) == 11: # electron/positron daughter
                            cumulative_deposited_energy += truth_startMomentum_arr[event_i][descendants_indices[descendant_i]][3]

                    if cumulative_deposited_energy > original_gamma_energy / 2: # it has deposited enough energy to effectively count as a pair conversion
                        break

                    if 22 in descendants_pdgs: # found a compton scatter, hasn't deposited enough energy yet, loop to consider that next photon
                        curr_id = descendants_ids[descendants_pdgs.index(22)]
                        #print("doing a compton scatter")
                    else: # no compton scatter, we're done, it's either a pair conversion or photoelectric absorption or a Geant tree deletion
                        break

                if cumulative_deposited_energy < original_gamma_energy / 2: # weird event, didn't deposit enough energy to count as a pair conversion
                    #print(f"weird event, no daughter photon, but also deposited less than half the energy: {cumulative_deposited_energy} / {original_gamma_energy}")
                    pass
                else:
                    # Store conversion point in the correct position based on photon ID
                    photon_id = truth_id_arr[event_i][i]
                    position = photon_id_to_position[photon_id]
                    
                    curr_true_gamma_pairconversion_xs[position] = truth_startXYZT_arr[event_i][descendants_indices[0]][0]
                    curr_true_gamma_pairconversion_ys[position] = truth_startXYZT_arr[event_i][descendants_indices[0]][1]
                    curr_true_gamma_pairconversion_zs[position] = truth_startXYZT_arr[event_i][descendants_indices[0]][2]
                    curr_true_num_gamma_pairconvert += 1

                    if -1 < curr_true_gamma_pairconversion_xs[position] <= 254.3 and -115.0 < curr_true_gamma_pairconversion_ys[position] <= 117.0 and 0.6 < curr_true_gamma_pairconversion_zs[position] <= 1036.4:
                        curr_true_num_gamma_pairconvert_in_FV += 1

                        if original_gamma_energy > 0.02:
                            curr_true_num_gamma_pairconvert_in_FV_20_MeV += 1

        # Filter out None values from conversion point lists
        curr_true_gamma_pairconversion_xs = [x for x in curr_true_gamma_pairconversion_xs if x is not None]
        curr_true_gamma_pairconversion_ys = [y for y in curr_true_gamma_pairconversion_ys if y is not None]
        curr_true_gamma_pairconversion_zs = [z for z in curr_true_gamma_pairconversion_zs if z is not None]
        
        true_num_gamma.append(curr_true_num_gamma)
        true_gamma_energies.append(curr_true_gamma_energies)
        true_gamma_pairconversion_xs.append(curr_true_gamma_pairconversion_xs)
        true_gamma_pairconversion_ys.append(curr_true_gamma_pairconversion_ys)
        true_gamma_pairconversion_zs.append(curr_true_gamma_pairconversion_zs)
        true_num_gamma_pairconvert.append(curr_true_num_gamma_pairconvert)
        true_num_gamma_pairconvert_in_FV.append(curr_true_num_gamma_pairconvert_in_FV)
        true_num_gamma_pairconvert_in_FV_20_MeV.append(curr_true_num_gamma_pairconvert_in_FV_20_MeV)

    if num_infinite_loops_broken > 0:
        print(f"Broke infinite loops in the true gamma daughter search {num_infinite_loops_broken} / {df.shape[0]} times")

    df["true_num_gamma"] = true_num_gamma
    df["true_gamma_energies"] = true_gamma_energies
    df["true_gamma_pairconversion_xs"] = true_gamma_pairconversion_xs
    df["true_gamma_pairconversion_ys"] = true_gamma_pairconversion_ys
    df["true_gamma_pairconversion_zs"] = true_gamma_pairconversion_zs
    df["true_num_gamma_pairconvert"] = true_num_gamma_pairconvert
    df["true_num_gamma_pairconvert_in_FV"] = true_num_gamma_pairconvert_in_FV
    df["true_num_gamma_pairconvert_in_FV_20_MeV"] = true_num_gamma_pairconvert_in_FV_20_MeV
    df["true_one_pairconvert_in_FV_20_MeV"] = true_num_gamma_pairconvert_in_FV_20_MeV == 1

    return df


def add_signal_categories(all_df):

    filetype_arr = all_df["filetype"].to_numpy()
    all_df["normal_overlay"] = ~(filetype_arr == "dirt_overlay") & ~(filetype_arr == "ext")

    truth_inFV_arr = all_df["wc_truth_vtxInside"].to_numpy().astype(bool)
    all_df["wc_truth_inFV"] = truth_inFV_arr

    true_num_gamma_pairconvert_in_FV = all_df["true_num_gamma_pairconvert_in_FV"].to_numpy()
    truth_0g_arr = true_num_gamma_pairconvert_in_FV == 0
    truth_1g_arr = true_num_gamma_pairconvert_in_FV == 1
    truth_2g_arr = true_num_gamma_pairconvert_in_FV == 2
    truth_3plusg_arr = true_num_gamma_pairconvert_in_FV >= 3
    all_df["wc_truth_0g"] = truth_0g_arr
    all_df["wc_truth_1g"] = truth_1g_arr
    all_df["wc_truth_2g"] = truth_2g_arr
    all_df["wc_truth_3plusg"] = truth_3plusg_arr

    wc_true_max_prim_proton_energy_arr = all_df["wc_true_max_prim_proton_energy"].to_numpy()
    truth_Np_arr = wc_true_max_prim_proton_energy_arr >= 35
    truth_0p_arr = wc_true_max_prim_proton_energy_arr < 35
    all_df["wc_truth_Np"] = truth_Np_arr
    all_df["wc_truth_0p"] = truth_0p_arr

    truth_isCC_arr = all_df["wc_truth_isCC"].to_numpy().astype(bool)
    truth_nuPdg_arr = all_df["wc_truth_nuPdg"].to_numpy()
    truth_numuCC_arr = truth_isCC_arr & (np.abs(truth_nuPdg_arr) == 14)
    truth_nueCC_arr = truth_isCC_arr & (np.abs(truth_nuPdg_arr) == 12)
    all_df["wc_truth_numuCC"] = truth_numuCC_arr
    all_df["wc_truth_notnumuCC"] = ~truth_numuCC_arr
    all_df["wc_truth_nueCC"] = truth_nueCC_arr
    all_df["wc_truth_notnueCC"] = ~truth_nueCC_arr
    all_df["wc_truth_isCC"] = truth_isCC_arr
    all_df["wc_truth_isNC"] = ~truth_isCC_arr
    all_df["wc_truth_1mu"] = truth_numuCC_arr & truth_inFV_arr
    all_df["wc_truth_0mu"] = ~(truth_numuCC_arr & truth_inFV_arr)
    all_df["wc_truth_1e"] = truth_nueCC_arr & truth_inFV_arr
    all_df["wc_truth_0e"] = ~(truth_nueCC_arr & truth_inFV_arr)

    truth_NprimPio_arr = all_df["wc_truth_NprimPio"].to_numpy()
    all_df["wc_truth_0pi0"] = truth_NprimPio_arr == 0
    all_df["wc_truth_1pi0"] = truth_NprimPio_arr == 1
    all_df["wc_truth_multi_pi0"] = truth_NprimPio_arr > 1

    topological_conditions = [all_df.eval(query) for query in topological_category_queries]
    for i1, condition1 in enumerate(topological_conditions):
        for i2, condition2 in enumerate(topological_conditions):
            if i1 != i2:
                overlap = condition1 & condition2
                if overlap.any():
                    print(f"Overlapping topological signal definitions: {topological_category_labels[i1]} and {topological_category_labels[i2]}")
                    row = all_df[condition1 & condition2].iloc[0]
                    print(f"Example: {row['true_num_gamma_pairconvert_in_FV']=}, {row['wc_truth_isCC']=}, {row['wc_truth_nuPdg']=}, {row['wc_truth_NprimPio']=}, {row['wc_truth_0e']=}, {row['wc_truth_0g']=}, {row['wc_truth_1g']=}, {row['wc_truth_2g']=}")
                    raise AssertionError
    all_df['topological_signal_category'] = np.select(topological_conditions, topological_category_labels, default="other")
    uncategorized_df = all_df[all_df['topological_signal_category'] == 'other']
    if len(uncategorized_df) > 0:
        print(f"Uncategorized topological signal categories!")
        row = uncategorized_df.iloc[0]
        print(f"Example: {row['true_num_gamma_pairconvert_in_FV']=}, {row['wc_truth_isCC']=}, {row['wc_truth_nuPdg']=}, {row['wc_truth_NprimPio']=}, {row['wc_truth_0e']=}, {row['wc_truth_0g']=}")
        raise AssertionError
    all_df["topological_signal_category"] = np.select(topological_conditions, topological_category_labels, default="other")
    print("\ntopological signal categories:")
    for topological_signal_category in all_df['topological_signal_category'].unique():
        curr_df = all_df[all_df['topological_signal_category'] == topological_signal_category]
        unweighted_num = curr_df.shape[0]
        weighted_num = curr_df['wc_net_weight'].sum()
        print(f"    {topological_signal_category}: {weighted_num:.2f} ({unweighted_num})")

    physics_conditions = [all_df.eval(query) for query in physics_category_queries]
    for i1, condition1 in enumerate(physics_conditions):
        for i2, condition2 in enumerate(physics_conditions):
            if i1 != i2:
                overlap = condition1 & condition2
                if overlap.any():
                    print(f"Overlapping physics signal definitions: {physics_category_labels[i1]} and {physics_category_labels[i2]}")
                    row = all_df[condition1 & condition2].iloc[0]
                    print(f"Example: {row['true_num_gamma_pairconvert_in_FV']=}, {row['wc_truth_isCC']=}, {row['wc_truth_nuPdg']=}, {row['wc_truth_NprimPio']=}, {row['wc_truth_0e']=}, {row['wc_truth_0g']=}, {row['wc_truth_1g']=}, {row['wc_truth_2g']=}")
                    raise AssertionError
    all_df["physics_signal_category"] = np.select(physics_conditions, physics_category_labels, default="other")
    uncategorized_df = all_df[all_df['physics_signal_category'] == 'other']
    if len(uncategorized_df) > 0:
        print(f"Uncategorized physics signal categories!")
        row = uncategorized_df.iloc[0]
        print(f"Example: {row['wc_truth_inFV']=}, {row['wc_truth_isCC']=}, {row['wc_truth_NCDelta']=}, {row['wc_truth_NprimPio']=}, {row['wc_truth_nueCC']=}")
        raise AssertionError
    print("\nphysics signal categories:")
    for physics_signal_category in all_df['physics_signal_category'].unique():
        curr_df = all_df[all_df['physics_signal_category'] == physics_signal_category]
        unweighted_num = curr_df.shape[0]
        weighted_num = curr_df['wc_net_weight'].sum()
        print(f"    {physics_signal_category}: {weighted_num:.2f} ({unweighted_num})")

    return all_df
