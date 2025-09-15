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
    has_photonuclear_absorption_flags = []
    max_true_prim_proton_energies = []
    sum_true_prim_proton_energies = []
    true_leading_shower_energies = []
    true_leading_shower_costhetas = []
    true_leading_pi0_energies = []
    true_leading_pi0_costhetas = []
    true_outgoing_lepton_energies = []
    true_nums_prim_protons = []
    true_nums_prim_protons_35 = []
    truth_ids = df["wc_truth_id"].to_numpy()
    truth_pdgs = df["wc_truth_pdg"].to_numpy()
    truth_mothers = df["wc_truth_mother"].to_numpy()
    truth_startMomentums = df["wc_truth_startMomentum"].to_numpy()
    for i in tqdm(range(df.shape[0])):
        max_true_prim_proton_energy = -1
        sum_true_prim_proton_energy = 0
        max_shower_energy = -1.
        max_shower_costheta = -2.
        max_pi0_energy = -1.
        max_pi0_costheta = -2.
        true_num_prim_protons = 0
        true_num_prim_protons_35 = 0
        true_outgoing_lepton_energy = -1.
        truth_id_list = truth_ids[i]
        truth_pdg_list = truth_pdgs[i]
        truth_mother_list = truth_mothers[i]
        truth_startMomentum_list = truth_startMomentums[i]
        if isinstance(truth_id_list, float) and np.isnan(truth_id_list):
            num_particles = 0
        else:
            num_particles = len(truth_id_list)
        has_photonuclear_absorption = False
        for j in range(num_particles):
            if truth_pdg_list[j] == 22: # photon
                truth_photon_parent_id = truth_id_list[j]
                for k in range(len(truth_pdg_list)):
                    if truth_id_list[k] == truth_photon_parent_id:
                        parent_pdg = truth_pdg_list[k]
                if parent_pdg == 111: # pi0 photon
                    # now, focusing on the pi0 daughter decay products
                    photon_daughter_pdgs = []
                    for k in range(num_particles):
                        if truth_mother_list[k] == truth_id_list[j]:
                            photon_daughter_pdgs.append(truth_pdg_list[k])

                    for pdg in photon_daughter_pdgs:
                        if pdg >= 1_000_000_000: # ten digit nuclear PDG code
                            has_photonuclear_absorption = True

            if truth_pdg_list[j] == 22 or abs(truth_pdg_list[j]) == 11: # photon, electron, or positron
                if truth_startMomentum_list[j][3] * 1000. > max_shower_energy:
                    max_shower_energy = truth_startMomentum_list[j][3] * 1000.
                    max_shower_costheta = truth_startMomentum_list[j][2] / truth_startMomentum_list[j][3] # should be basically z / (x**2 + y**2 + z**2)**0.5

            if truth_pdg_list[j] == 111: # pi0
                curr_pi0_energy = truth_startMomentum_list[j][3] * 1000. - 134.9768
                if curr_pi0_energy > max_pi0_energy:
                    max_pi0_energy = curr_pi0_energy
                    max_tot_momentum = np.sqrt(truth_startMomentum_list[j][0]**2 + truth_startMomentum_list[j][1]**2 + truth_startMomentum_list[j][2]**2)
                    max_z_momentum = truth_startMomentum_list[j][2]
                    max_pi0_costheta = max_z_momentum / max_tot_momentum

            if truth_mother_list[j] == 0 and truth_pdg_list[j] == 2212: # primary proton
                true_num_prim_protons += 1
                if truth_startMomentum_list[j][3] * 1000. - 938.272088 > 35.:
                    true_num_prim_protons_35 += 1
                max_true_prim_proton_energy = max(max_true_prim_proton_energy, truth_startMomentum_list[j][3] * 1000. - 938.272088)
                sum_true_prim_proton_energy += truth_startMomentum_list[j][3] * 1000. - 938.272088

            if truth_mother_list[j] == 0 and 11 <= abs(truth_pdg_list[j]) <= 16: # lepton
                true_outgoing_lepton_energy = truth_startMomentum_list[j][3] * 1000.

        max_true_prim_proton_energies.append(max_true_prim_proton_energy)
        sum_true_prim_proton_energies.append(sum_true_prim_proton_energy)
        true_outgoing_lepton_energies.append(true_outgoing_lepton_energy)
        true_nums_prim_protons.append(true_num_prim_protons)
        true_nums_prim_protons_35.append(true_num_prim_protons_35)
        true_leading_shower_energies.append(max_shower_energy)
        true_leading_shower_costhetas.append(max_shower_costheta)
        true_leading_pi0_energies.append(max_pi0_energy)
        true_leading_pi0_costhetas.append(max_pi0_costheta)
        has_photonuclear_absorption_flags.append(has_photonuclear_absorption)

    df["wc_true_max_prim_proton_energy"] = max_true_prim_proton_energies
    df["wc_true_sum_prim_proton_energy"] = sum_true_prim_proton_energies
    df["wc_true_outgoing_lepton_energy"] = true_outgoing_lepton_energies
    df["wc_true_num_prim_protons"] = true_nums_prim_protons
    df["wc_true_num_prim_protons_35"] = true_nums_prim_protons_35
    df["wc_true_leading_shower_energy"] = true_leading_shower_energies
    df["wc_true_leading_shower_costheta"] = true_leading_shower_costhetas
    df["wc_true_leading_pi0_energy"] = true_leading_pi0_energies
    df["wc_true_leading_pi0_costheta"] = true_leading_pi0_costhetas
    df["wc_true_has_photonuclear_absorption"] = has_photonuclear_absorption_flags

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

    # TODO: add multiplicities for different WC reco particles in the PF tree, primary and nonprimary, with different energy thresholds

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

    # accounting for the fact that rarely, wc_truth_NCDelta can be true for CC events, presumably due to some GENIE bug
    all_df["wc_truth_NCDeltaRad"] = (~truth_isCC_arr) & all_df["wc_truth_NCDelta"].to_numpy().astype(bool)

    topological_conditions = [all_df.eval(query) for query in topological_category_queries]
    for i1, condition1 in enumerate(topological_conditions):
        for i2, condition2 in enumerate(topological_conditions):
            if i1 != i2:
                overlap = condition1 & condition2
                if overlap.any():
                    print(f"Overlapping topological signal definitions: {topological_category_labels[i1]} and {topological_category_labels[i2]}")
                    row = all_df[condition1 & condition2].iloc[0]
                    print(f"Example: {row['filename']=}, {row['filetype']=}, {row['run']=}, {row['subrun']=}, {row['event']=}, {row['true_num_gamma_pairconvert_in_FV']=}, {row['wc_truth_isCC']=}, {row['wc_truth_nuPdg']=}, {row['wc_truth_NprimPio']=}, {row['wc_truth_0e']=}, {row['wc_truth_0g']=}, {row['wc_truth_1g']=}, {row['wc_truth_2g']=}")
                    raise AssertionError
    all_df['topological_signal_category'] = np.select(topological_conditions, topological_category_labels, default="other")
    uncategorized_df = all_df[all_df['topological_signal_category'] == 'other']
    if len(uncategorized_df) > 0:
        print(f"Uncategorized topological signal categories!")
        row = uncategorized_df.iloc[0]
        print(f"Example: {row['filename']=}, {row['filetype']=}, {row['run']=}, {row['subrun']=}, {row['event']=}, {row['true_num_gamma_pairconvert_in_FV']=}, {row['wc_truth_isCC']=}, {row['wc_truth_nuPdg']=}, {row['wc_truth_NprimPio']=}, {row['wc_truth_0e']=}, {row['wc_truth_0g']=}, {row['wc_truth_1g']=}, {row['wc_truth_2g']=}")
        raise AssertionError
    all_df["topological_signal_category"] = np.select(topological_conditions, topological_category_labels, default="other")


    print_categories = False
    if print_categories:
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
                    print(f"Example: {row['filename']=}, {row['filetype']=}, {row['run']=}, {row['subrun']=}, {row['event']=}, {row['true_num_gamma_pairconvert_in_FV']=}, {row['wc_truth_isCC']=}, {row['wc_truth_nuPdg']=}, {row['wc_truth_NprimPio']=}, {row['wc_truth_0e']=}, {row['wc_truth_0g']=}, {row['wc_truth_1g']=}, {row['wc_truth_2g']=}, {row['wc_truth_NCDelta']=}")
                    raise AssertionError
    all_df["physics_signal_category"] = np.select(physics_conditions, physics_category_labels, default="other")
    uncategorized_df = all_df[all_df['physics_signal_category'] == 'other']
    if len(uncategorized_df) > 0:
        print(f"Uncategorized physics signal categories!")
        row = uncategorized_df.iloc[0]
        print(f"Example: {row['filename']=}, {row['filetype']=}, {row['run']=}, {row['subrun']=}, {row['event']=}, {row['true_num_gamma_pairconvert_in_FV']=}, {row['wc_truth_isCC']=}, {row['wc_truth_nuPdg']=}, {row['wc_truth_NprimPio']=}, {row['wc_truth_0e']=}, {row['wc_truth_0g']=}, {row['wc_truth_1g']=}, {row['wc_truth_2g']=}, {row['wc_truth_NCDelta']=}")
        raise AssertionError
    
    if print_categories:
        print("\nphysics signal categories:")
        for physics_signal_category in all_df['physics_signal_category'].unique():
            curr_df = all_df[all_df['physics_signal_category'] == physics_signal_category]
            unweighted_num = curr_df.shape[0]
            weighted_num = curr_df['wc_net_weight'].sum()
            print(f"    {physics_signal_category}: {weighted_num:.2f} ({unweighted_num})")

    return all_df



def do_blip_postprocessing(df):

    # TODO: replace this blip analyzing code with a more detailed method or BDT according to Karan's studies

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

    closest_upstream_blip_distance = []
    closest_upstream_blip_angle = []
    closest_upstream_blip_impact_parameter = []
    closest_upstream_blip_energy = []
    closest_upstream_blip_dx = []
    closest_upstream_blip_dw = []
    for event_index in tqdm(range(len(df)), desc="Finding closest upstream blip"):

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
    
    return df

def do_glee_postprocessing(df):

    # TODO: add more gLEE variables here, make sure this max SSV score variable is the correct method
    # Seems to always give NaN for now?

    glee_ssv_score = df["glee_sss_candidate_veto_score"].to_numpy()
    glee_ssv_3d_score = df["glee_sss3d_shower_score"].to_numpy()
    max_ssv_scores = []
    max_ssv_3d_scores = []
    for event_i in tqdm(range(len(glee_ssv_score)), desc="Analyzing gLEE ssv scores"):
        if len(glee_ssv_score[event_i]) == 0:
            max_ssv_scores.append(np.nan)
            max_ssv_3d_scores.append(np.nan)
        else:
            print("nonzero length, max_ssv_scores: ", np.max(glee_ssv_score[event_i]))
            max_ssv_scores.append(np.max(glee_ssv_score[event_i]))
            max_ssv_3d_scores.append(np.max(glee_ssv_3d_score[event_i]))
    df["glee_max_ssv_score"] = max_ssv_scores
    df["glee_max_3d_shower_score"] = max_ssv_3d_scores
    return df

def do_lantern_postprocessing(df):

    # References of variables that the LANTERN team has found useful:
    # LANTERN 1e selection reco variables, from https://github.com/NuTufts/lantern_ana/blob/main/studies/nue_cc_inclusive/plot_mmr_nue_selection_vars_run4b.py
    """
    vars = [('eventweight_weight',200,0,10,'event weight',0), # don't include event weight as a training variable
            ('recoElectron_emax_econfidence', 40, 0, 20,'electron confidence score',0),
            ('recoElectron_emax_primary_score',40,0,1.0,'primary score',0),
            ('recoElectron_emax_fromneutral_score',40,0,1.0,'from neutral parent score',0),
            ('recoElectron_emax_fromcharged_score',40,0,1.0,'from charged parent score',0),
            ('recoElectron_emax_el_normedscore',40,0,1.0,'electron-like score (normalized)',0),
            ('recoMuonTrack_nMuTracks',20,0,20,'num of muon-like tracks',0),
            ('recoMuonTrack_max_muscore',50,-20,0.5,'max mu-like score',0),
            ('vertex_properties_score',60,0.4,1.0,'keypoint score',0),
            ('vertex_properties_cosmicfrac',50,0,1.01,'fraction of cosmic pixel near vertex',0),
            ('visible_energy',15,0,3000,'visible energy; MeV',0)
    """
    # LANTERN 1g selection reco variables, from https://github.com/NuTufts/lantern_ana/blob/main/studies/1gXp_nc/plot_1gX_vars.py
    """
    selection_cut = "vertex_properties_found==1"
    selection_cut += " && nphotons==1"
    selection_cut += " && nProtons==0"
    selection_cut += " && nPions==0"
    selection_cut += " && nElectrons==0"
    selection_cut += " && nMuons==0"
    selection_cut += " && TMath::IsNaN(vertex_properties_score)==0"
    """

    # see function here: https://github.com/NuTufts/lantern_ana/blob/94f31b83e8a170230ca7b948c701671ab3099bbd/lantern_ana/utils/get_primary_electron_candidates.py
    # this code is similar, but adds information about up to two electrons and two photons, and extra pi0-relevant information for the two photon case, and counts all particles

    max_shower_charges = []

    max_electron_shower_charges = []
    max_electron_shower_PhScores = []
    max_electron_shower_ElScores = []
    max_electron_shower_MuScores = []
    max_electron_shower_PiScores = []
    max_electron_shower_PrScores = []
    max_electron_shower_el_normedscores = []
    max_electron_shower_ph_normedscores = []
    max_electron_shower_electron_confidences = []
    max_electron_shower_PrimaryScores = []
    max_electron_shower_FromNeutralScores = []
    max_electron_shower_FromChargedScores = []
    max_electron_shower_CosThetas = []
    max_electron_shower_CosThetaYs = []
    max_electron_shower_DistToVtxs = []

    second_max_electron_shower_charges = []
    second_max_electron_shower_PhScores = []
    second_max_electron_shower_ElScores = []
    second_max_electron_shower_MuScores = []
    second_max_electron_shower_PiScores = []
    second_max_electron_shower_PrScores = []
    second_max_electron_shower_el_normedscores = []
    second_max_electron_shower_ph_normedscores = []
    second_max_electron_shower_electron_confidences = []
    second_max_electron_shower_PrimaryScores = []
    second_max_electron_shower_FromNeutralScores = []
    second_max_electron_shower_FromChargedScores = []
    second_max_electron_shower_CosThetas = []
    second_max_electron_shower_CosThetaYs = []
    second_max_electron_shower_DistToVtxs = []
    
    max_photon_shower_charges = []
    max_photon_shower_PhScores = []
    max_photon_shower_ElScores = []
    max_photon_shower_MuScores = []
    max_photon_shower_PiScores = []
    max_photon_shower_PrScores = []
    max_photon_shower_el_normedscores = []
    max_photon_shower_ph_normedscores = []
    max_photon_shower_electron_confidences = []
    max_photon_shower_PrimaryScores = []
    max_photon_shower_FromNeutralScores = []
    max_photon_shower_FromChargedScores = []
    max_photon_shower_CosThetas = []
    max_photon_shower_CosThetaYs = []
    max_photon_shower_DistToVtxs = []

    second_max_photon_shower_charges = []
    second_max_photon_shower_PhScores = []
    second_max_photon_shower_ElScores = []
    second_max_photon_shower_MuScores = []
    second_max_photon_shower_PiScores = []
    second_max_photon_shower_PrScores = []
    second_max_photon_shower_el_normedscores = []
    second_max_photon_shower_ph_normedscores = []
    second_max_photon_shower_electron_confidences = []
    second_max_photon_shower_PrimaryScores = []
    second_max_photon_shower_FromNeutralScores = []
    second_max_photon_shower_FromChargedScores = []
    second_max_photon_shower_CosThetas = []
    second_max_photon_shower_CosThetaYs = []
    second_max_photon_shower_DistToVtxs = []

    diphoton_opening_angles = []
    diphoton_energies = []
    diphoton_costhetas = []
    diphoton_masses = []

    prim_shower_photon_nums = []
    prim_shower_electron_nums = []
    prim_shower_muon_nums = []
    prim_shower_charged_pion_nums = []
    prim_shower_proton_nums = []

    nShowers = df["lantern_nShowers"].to_numpy()
    showerIsSecondary = df["lantern_showerIsSecondary"].to_numpy()
    showerPID = df["lantern_showerPID"].to_numpy()
    showerPhScore = df["lantern_showerPhScore"].to_numpy()
    showerElScore = df["lantern_showerElScore"].to_numpy()
    showerMuScore = df["lantern_showerMuScore"].to_numpy()
    showerPiScore = df["lantern_showerPiScore"].to_numpy()
    showerPrScore = df["lantern_showerPrScore"].to_numpy()
    showerCharge = df["lantern_showerCharge"].to_numpy()
    showerPurity = df["lantern_showerPurity"].to_numpy()
    showerComp = df["lantern_showerComp"].to_numpy()
    showerPrimaryScore = df["lantern_showerPrimaryScore"].to_numpy()
    showerFromNeutralScore = df["lantern_showerFromNeutralScore"].to_numpy()
    showerFromChargedScore = df["lantern_showerFromChargedScore"].to_numpy()
    showerCosTheta = df["lantern_showerCosTheta"].to_numpy()
    showerCosThetaY = df["lantern_showerCosThetaY"].to_numpy()
    showerDistToVtx = df["lantern_showerDistToVtx"].to_numpy()
    showerStartDirX = df["lantern_showerStartDirX"].to_numpy()
    showerStartDirY = df["lantern_showerStartDirY"].to_numpy()
    showerStartDirZ = df["lantern_showerStartDirZ"].to_numpy()
    for event_i in tqdm(range(len(df)), desc="Analyzing LANTERN showers"):
        curr_nShowers = nShowers[event_i]
        curr_showerIsSecondary = showerIsSecondary[event_i]
        curr_showerPID = showerPID[event_i]
        curr_showerPhScore = showerPhScore[event_i]
        curr_showerElScore = showerElScore[event_i]
        curr_showerMuScore = showerMuScore[event_i]
        curr_showerPiScore = showerPiScore[event_i]
        curr_showerPrScore = showerPrScore[event_i]
        curr_showerCharge = showerCharge[event_i]
        curr_showerPurity = showerPurity[event_i]
        curr_showerComp = showerComp[event_i]
        curr_showerPrimaryScore = showerPrimaryScore[event_i]
        curr_showerFromNeutralScore = showerFromNeutralScore[event_i]
        curr_showerFromChargedScore = showerFromChargedScore[event_i]
        curr_showerCosTheta = showerCosTheta[event_i]
        curr_showerCosThetaY = showerCosThetaY[event_i]
        curr_showerDistToVtx = showerDistToVtx[event_i]
        curr_showerStartDirX = showerStartDirX[event_i]
        curr_showerStartDirY = showerStartDirY[event_i]
        curr_showerStartDirZ = showerStartDirZ[event_i]

        max_shower_charge = 0

        max_photon_shower_charge = 0
        max_photon_shower_PhScore = np.nan
        max_photon_shower_ElScore = np.nan
        max_photon_shower_MuScore = np.nan
        max_photon_shower_PiScore = np.nan
        max_photon_shower_PrScore = np.nan
        max_photon_shower_electron_confidence = np.nan
        max_photon_shower_ph_normedscore = np.nan
        max_photon_shower_el_normedscore = np.nan
        max_photon_shower_PrimaryScore = np.nan
        max_photon_shower_FromNeutralScore = np.nan
        max_photon_shower_FromChargedScore = np.nan
        max_photon_shower_CosTheta = np.nan
        max_photon_shower_CosThetaY = np.nan
        max_photon_shower_DistToVtx = np.nan
        max_photon_shower_StartDirX = np.nan
        max_photon_shower_StartDirY = np.nan
        max_photon_shower_StartDirZ = np.nan
        second_max_photon_shower_charge = -1e-6
        second_max_photon_shower_PhScore = np.nan
        second_max_photon_shower_ElScore = np.nan
        second_max_photon_shower_MuScore = np.nan
        second_max_photon_shower_PiScore = np.nan
        second_max_photon_shower_PrScore = np.nan
        second_max_photon_shower_electron_confidence = np.nan
        second_max_photon_shower_ph_normedscore = np.nan
        second_max_photon_shower_el_normedscore = np.nan
        second_max_photon_shower_PrimaryScore = np.nan
        second_max_photon_shower_FromNeutralScore = np.nan
        second_max_photon_shower_FromChargedScore = np.nan
        second_max_photon_shower_CosTheta = np.nan
        second_max_photon_shower_CosThetaY = np.nan
        second_max_photon_shower_DistToVtx = np.nan
        second_max_photon_shower_StartDirX = np.nan
        second_max_photon_shower_StartDirY = np.nan
        second_max_photon_shower_StartDirZ = np.nan

        max_electron_shower_charge = 0
        max_electron_shower_PhScore = np.nan
        max_electron_shower_ElScore = np.nan
        max_electron_shower_MuScore = np.nan
        max_electron_shower_PiScore = np.nan
        max_electron_shower_PrScore = np.nan
        max_electron_shower_electron_confidence = np.nan
        max_electron_shower_ph_normedscore = np.nan
        max_electron_shower_el_normedscore = np.nan
        max_electron_shower_PrimaryScore = np.nan
        max_electron_shower_FromNeutralScore = np.nan
        max_electron_shower_FromChargedScore = np.nan
        max_electron_shower_CosTheta = np.nan
        max_electron_shower_CosThetaY = np.nan
        max_electron_shower_DistToVtx = np.nan
        second_max_electron_shower_charge = -1e-6
        second_max_electron_shower_PhScore = np.nan
        second_max_electron_shower_ElScore = np.nan
        second_max_electron_shower_MuScore = np.nan
        second_max_electron_shower_PiScore = np.nan
        second_max_electron_shower_PrScore = np.nan
        second_max_electron_shower_electron_confidence = np.nan
        second_max_electron_shower_ph_normedscore = np.nan
        second_max_electron_shower_el_normedscore = np.nan
        second_max_electron_shower_PrimaryScore = np.nan
        second_max_electron_shower_FromNeutralScore = np.nan
        second_max_electron_shower_FromChargedScore = np.nan
        second_max_electron_shower_CosTheta = np.nan
        second_max_electron_shower_CosThetaY = np.nan
        second_max_electron_shower_DistToVtx = np.nan

        diphoton_opening_angle = np.nan
        diphoton_energy = np.nan
        diphoton_costheta = np.nan
        diphoton_mass = np.nan

        curr_num_photons = 0
        curr_num_electrons = 0
        curr_num_muons = 0
        curr_num_charged_pions = 0
        curr_num_protons = 0

        for shower_i in range(curr_nShowers):
            if curr_showerIsSecondary[shower_i] != 0: # only considering primary showers
                continue

            min_charge, min_completeness, min_purity = 0, 0, 0
            if curr_showerCharge[shower_i] < min_charge or curr_showerComp[shower_i] < min_completeness or curr_showerPurity[shower_i] < min_purity:
                continue

            if curr_showerPID[shower_i] == 22:
                curr_num_photons += 1
            elif curr_showerPID[shower_i] == 11:
                curr_num_electrons += 1
            elif curr_showerPID[shower_i] == 13:
                curr_num_muons += 1
            elif curr_showerPID[shower_i] == 211:
                curr_num_charged_pions += 1
            elif curr_showerPID[shower_i] == 2212:
                curr_num_protons += 1
            else:
                assert False, "Invalid shower PID"

            is_electron = curr_showerElScore[shower_i] > curr_showerPhScore[shower_i]
            electron_confidence = curr_showerElScore[shower_i] - (curr_showerPhScore[shower_i] + curr_showerPiScore[shower_i]) / 2
            el_normedscore = np.exp(curr_showerElScore[shower_i]) / (np.exp(curr_showerPhScore[shower_i])
                                                                   + np.exp(curr_showerElScore[shower_i])
                                                                   + np.exp(curr_showerMuScore[shower_i])
                                                                   + np.exp(curr_showerPiScore[shower_i])
                                                                   + np.exp(curr_showerPrScore[shower_i]))
            ph_normedscore = np.exp(curr_showerPhScore[shower_i]) / (np.exp(curr_showerElScore[shower_i])
                                                                   + np.exp(curr_showerPhScore[shower_i])
                                                                   + np.exp(curr_showerMuScore[shower_i])
                                                                   + np.exp(curr_showerPiScore[shower_i])
                                                                   + np.exp(curr_showerPrScore[shower_i]))
            
            if curr_showerCharge[shower_i] > max_shower_charge:
                max_shower_charge = curr_showerCharge[shower_i]

            if is_electron:
                if curr_showerCharge[shower_i] > max_electron_shower_charge: # new largest electron shower
                    max_electron_shower_charge = curr_showerCharge[shower_i]
                    second_max_electron_shower_charge = max_shower_charge
                    second_max_electron_shower_charge = max_electron_shower_charge
                    second_max_electron_shower_PhScore = max_electron_shower_PhScore
                    second_max_electron_shower_ElScore = max_electron_shower_ElScore
                    second_max_electron_shower_MuScore = max_electron_shower_MuScore
                    second_max_electron_shower_PiScore = max_electron_shower_PiScore
                    second_max_electron_shower_PrScore = max_electron_shower_PrScore
                    second_max_electron_shower_ph_normedscore = max_electron_shower_ph_normedscore
                    second_max_electron_shower_el_normedscore = max_electron_shower_el_normedscore
                    second_max_electron_shower_electron_confidence = max_electron_shower_electron_confidence
                    second_max_electron_shower_PrimaryScore = max_electron_shower_PrimaryScore
                    second_max_electron_shower_FromNeutralScore = max_electron_shower_FromNeutralScore
                    second_max_electron_shower_FromChargedScore = max_electron_shower_FromChargedScore
                    second_max_electron_shower_CosTheta = max_electron_shower_CosTheta
                    second_max_electron_shower_CosThetaY = max_electron_shower_CosThetaY
                    second_max_electron_shower_DistToVtx = max_electron_shower_DistToVtx

                    max_electron_shower_charge = curr_showerCharge[shower_i]
                    max_electron_shower_PhScore = curr_showerPhScore[shower_i]
                    max_electron_shower_ElScore = curr_showerElScore[shower_i]
                    max_electron_shower_MuScore = curr_showerMuScore[shower_i]
                    max_electron_shower_PiScore = curr_showerPiScore[shower_i]
                    max_electron_shower_PrScore = curr_showerPrScore[shower_i]
                    max_electron_shower_ph_normedscore = ph_normedscore
                    max_electron_shower_el_normedscore = el_normedscore
                    max_electron_shower_electron_confidence = electron_confidence
                    max_electron_shower_PrimaryScore = curr_showerPrimaryScore[shower_i]
                    max_electron_shower_FromNeutralScore = curr_showerFromNeutralScore[shower_i]
                    max_electron_shower_FromChargedScore = curr_showerFromChargedScore[shower_i]
                    max_electron_shower_CosTheta = curr_showerCosTheta[shower_i]
                    max_electron_shower_CosThetaY = curr_showerCosThetaY[shower_i]
                    max_electron_shower_DistToVtx = curr_showerDistToVtx[shower_i]
                elif curr_showerCharge[shower_i] > second_max_electron_shower_charge: # new second largest electron shower
                    second_max_electron_shower_charge = curr_showerCharge[shower_i]
                    second_max_electron_shower_PhScore = curr_showerPhScore[shower_i]
                    second_max_electron_shower_ElScore = curr_showerElScore[shower_i]
                    second_max_electron_shower_MuScore = curr_showerMuScore[shower_i]
                    second_max_electron_shower_PiScore = curr_showerPiScore[shower_i]
                    second_max_electron_shower_PrScore = curr_showerPrScore[shower_i]
                    second_max_electron_shower_ph_normedscore = ph_normedscore
                    second_max_electron_shower_el_normedscore = el_normedscore
                    second_max_electron_shower_electron_confidence = electron_confidence
                    second_max_electron_shower_PrimaryScore = curr_showerPrimaryScore[shower_i]
                    second_max_electron_shower_FromNeutralScore = curr_showerFromNeutralScore[shower_i]
                    second_max_electron_shower_FromChargedScore = curr_showerFromChargedScore[shower_i]
                    second_max_electron_shower_CosTheta = curr_showerCosTheta[shower_i]
                    second_max_electron_shower_CosThetaY = curr_showerCosThetaY[shower_i]
                    second_max_electron_shower_DistToVtx = curr_showerDistToVtx[shower_i]

            else: # photon shower
                if curr_showerCharge[shower_i] > max_photon_shower_charge: # new largest photon shower
                    second_max_photon_shower_charge = max_photon_shower_charge
                    second_max_photon_shower_PhScore = max_photon_shower_PhScore
                    second_max_photon_shower_ElScore = max_photon_shower_ElScore
                    second_max_photon_shower_MuScore = max_photon_shower_MuScore
                    second_max_photon_shower_PiScore = max_photon_shower_PiScore
                    second_max_photon_shower_PrScore = max_photon_shower_PrScore
                    second_max_photon_shower_ph_normedscore = max_photon_shower_ph_normedscore
                    second_max_photon_shower_el_normedscore = max_photon_shower_el_normedscore
                    second_max_photon_shower_electron_confidence = max_photon_shower_electron_confidence
                    second_max_photon_shower_PrimaryScore = max_photon_shower_PrimaryScore
                    second_max_photon_shower_FromNeutralScore = max_photon_shower_FromNeutralScore
                    second_max_photon_shower_FromChargedScore = max_photon_shower_FromChargedScore
                    second_max_photon_shower_CosTheta = max_photon_shower_CosTheta
                    second_max_photon_shower_CosThetaY = max_photon_shower_CosThetaY
                    second_max_photon_shower_DistToVtx = max_photon_shower_DistToVtx
                    second_max_photon_shower_StartDirX = max_photon_shower_StartDirX
                    second_max_photon_shower_StartDirY = max_photon_shower_StartDirY
                    second_max_photon_shower_StartDirZ = max_photon_shower_StartDirZ

                    max_photon_shower_charge = curr_showerCharge[shower_i]
                    max_photon_shower_PhScore = curr_showerPhScore[shower_i]
                    max_photon_shower_ElScore = curr_showerElScore[shower_i]
                    max_photon_shower_MuScore = curr_showerMuScore[shower_i]
                    max_photon_shower_PiScore = curr_showerPiScore[shower_i]
                    max_photon_shower_PrScore = curr_showerPrScore[shower_i]
                    max_photon_shower_ph_normedscore = ph_normedscore
                    max_photon_shower_el_normedscore = el_normedscore
                    max_photon_shower_electron_confidence = electron_confidence
                    max_photon_shower_PrimaryScore = curr_showerPrimaryScore[shower_i]
                    max_photon_shower_FromNeutralScore = curr_showerFromNeutralScore[shower_i]
                    max_photon_shower_FromChargedScore = curr_showerFromChargedScore[shower_i]
                    max_photon_shower_CosTheta = curr_showerCosTheta[shower_i]
                    max_photon_shower_CosThetaY = curr_showerCosThetaY[shower_i]
                    max_photon_shower_DistToVtx = curr_showerDistToVtx[shower_i]
                    max_photon_shower_StartDirX = curr_showerStartDirX[shower_i]
                    max_photon_shower_StartDirY = curr_showerStartDirY[shower_i]
                    max_photon_shower_StartDirZ = curr_showerStartDirZ[shower_i]

                elif curr_showerCharge[shower_i] > second_max_photon_shower_charge: # new second largest photon shower
                    second_max_photon_shower_charge = curr_showerCharge[shower_i]
                    second_max_photon_shower_PhScore = curr_showerPhScore[shower_i]
                    second_max_photon_shower_ElScore = curr_showerElScore[shower_i]
                    second_max_photon_shower_MuScore = curr_showerMuScore[shower_i]
                    second_max_photon_shower_PiScore = curr_showerPiScore[shower_i]
                    second_max_photon_shower_PrScore = curr_showerPrScore[shower_i]
                    second_max_photon_shower_ph_normedscore = ph_normedscore
                    second_max_photon_shower_el_normedscore = el_normedscore
                    second_max_photon_shower_electron_confidence = electron_confidence
                    second_max_photon_shower_PrimaryScore = curr_showerPrimaryScore[shower_i]
                    second_max_photon_shower_FromNeutralScore = curr_showerFromNeutralScore[shower_i]
                    second_max_photon_shower_FromChargedScore = curr_showerFromChargedScore[shower_i]
                    second_max_photon_shower_CosTheta = curr_showerCosTheta[shower_i]
                    second_max_photon_shower_CosThetaY = curr_showerCosThetaY[shower_i]
                    second_max_photon_shower_DistToVtx = curr_showerDistToVtx[shower_i]
                    second_max_photon_shower_StartDirX = curr_showerStartDirX[shower_i]
                    second_max_photon_shower_StartDirY = curr_showerStartDirY[shower_i]
                    second_max_photon_shower_StartDirZ = curr_showerStartDirZ[shower_i]

        if max_photon_shower_charge > 0 and second_max_photon_shower_charge > 0: # LANTERN sees two photons

            photon_1_energy = max_photon_shower_charge
            photon_2_energy = second_max_photon_shower_charge

            photon_1_dir = np.array([max_photon_shower_StartDirX, max_photon_shower_StartDirY, max_photon_shower_StartDirZ])
            photon_2_dir = np.array([second_max_photon_shower_StartDirX, second_max_photon_shower_StartDirY, second_max_photon_shower_StartDirZ])
            photon_1_dir = photon_1_dir / np.linalg.norm(photon_1_dir)
            photon_2_dir = photon_2_dir / np.linalg.norm(photon_2_dir)

            diphoton_energy = photon_1_energy + photon_2_energy
            diphoton_momentum = photon_1_energy * photon_1_dir + photon_2_energy * photon_2_dir

            diphoton_opening_angle = np.arccos(np.dot(photon_1_dir, photon_2_dir)) * 180 / np.pi
            diphoton_costheta = diphoton_momentum[2] / diphoton_energy
            diphoton_mass_squared = diphoton_energy**2 - diphoton_momentum[0]**2 - diphoton_momentum[1]**2 - diphoton_momentum[2]**2
            diphoton_mass_squared = np.clip(diphoton_mass_squared, 0.0, None) # accounting for floating point errors near 0
            diphoton_mass = np.sqrt(diphoton_mass_squared)

        max_shower_charges.append(max_shower_charge)
        max_electron_shower_charges.append(max_electron_shower_charge)
        max_electron_shower_PhScores.append(max_electron_shower_PhScore)
        max_electron_shower_ElScores.append(max_electron_shower_ElScore)
        max_electron_shower_MuScores.append(max_electron_shower_MuScore)
        max_electron_shower_PiScores.append(max_electron_shower_PiScore)
        max_electron_shower_PrScores.append(max_electron_shower_PrScore)
        max_electron_shower_electron_confidences.append(max_electron_shower_electron_confidence)
        max_electron_shower_ph_normedscores.append(max_electron_shower_ph_normedscore)
        max_electron_shower_el_normedscores.append(max_electron_shower_el_normedscore)
        max_electron_shower_PrimaryScores.append(max_electron_shower_PrimaryScore)
        max_electron_shower_FromNeutralScores.append(max_electron_shower_FromNeutralScore)
        max_electron_shower_FromChargedScores.append(max_electron_shower_FromChargedScore)
        max_electron_shower_CosThetas.append(max_electron_shower_CosTheta)
        max_electron_shower_CosThetaYs.append(max_electron_shower_CosThetaY)
        max_electron_shower_DistToVtxs.append(max_electron_shower_DistToVtx)
        second_max_electron_shower_charges.append(second_max_electron_shower_charge)
        second_max_electron_shower_PhScores.append(second_max_electron_shower_PhScore)
        second_max_electron_shower_ElScores.append(second_max_electron_shower_ElScore)
        second_max_electron_shower_MuScores.append(second_max_electron_shower_MuScore)
        second_max_electron_shower_PiScores.append(second_max_electron_shower_PiScore)
        second_max_electron_shower_PrScores.append(second_max_electron_shower_PrScore)
        second_max_electron_shower_electron_confidences.append(second_max_electron_shower_electron_confidence)
        second_max_electron_shower_ph_normedscores.append(second_max_electron_shower_ph_normedscore)
        second_max_electron_shower_el_normedscores.append(second_max_electron_shower_el_normedscore)
        second_max_electron_shower_PrimaryScores.append(second_max_electron_shower_PrimaryScore)
        second_max_electron_shower_FromNeutralScores.append(second_max_electron_shower_FromNeutralScore)
        second_max_electron_shower_FromChargedScores.append(second_max_electron_shower_FromChargedScore)
        second_max_electron_shower_CosThetas.append(second_max_electron_shower_CosTheta)
        second_max_electron_shower_CosThetaYs.append(second_max_electron_shower_CosThetaY)
        second_max_electron_shower_DistToVtxs.append(second_max_electron_shower_DistToVtx)
        max_photon_shower_charges.append(max_photon_shower_charge)
        max_photon_shower_PhScores.append(max_photon_shower_PhScore)
        max_photon_shower_ElScores.append(max_photon_shower_ElScore)
        max_photon_shower_MuScores.append(max_photon_shower_MuScore)
        max_photon_shower_PiScores.append(max_photon_shower_PiScore)
        max_photon_shower_PrScores.append(max_photon_shower_PrScore)
        max_photon_shower_electron_confidences.append(max_photon_shower_electron_confidence)
        max_photon_shower_ph_normedscores.append(max_photon_shower_ph_normedscore)
        max_photon_shower_el_normedscores.append(max_photon_shower_el_normedscore)
        max_photon_shower_PrimaryScores.append(max_photon_shower_PrimaryScore)
        max_photon_shower_FromNeutralScores.append(max_photon_shower_FromNeutralScore)
        max_photon_shower_FromChargedScores.append(max_photon_shower_FromChargedScore)
        max_photon_shower_CosThetas.append(max_photon_shower_CosTheta)
        max_photon_shower_CosThetaYs.append(max_photon_shower_CosThetaY)
        max_photon_shower_DistToVtxs.append(max_photon_shower_DistToVtx)
        second_max_photon_shower_charges.append(second_max_photon_shower_charge)
        second_max_photon_shower_PhScores.append(second_max_photon_shower_PhScore)
        second_max_photon_shower_ElScores.append(second_max_photon_shower_ElScore)
        second_max_photon_shower_MuScores.append(second_max_photon_shower_MuScore)
        second_max_photon_shower_PiScores.append(second_max_photon_shower_PiScore)
        second_max_photon_shower_PrScores.append(second_max_photon_shower_PrScore)
        second_max_photon_shower_electron_confidences.append(second_max_photon_shower_electron_confidence)
        second_max_photon_shower_ph_normedscores.append(second_max_photon_shower_ph_normedscore)
        second_max_photon_shower_el_normedscores.append(second_max_photon_shower_el_normedscore)
        second_max_photon_shower_PrimaryScores.append(second_max_photon_shower_PrimaryScore)
        second_max_photon_shower_FromNeutralScores.append(second_max_photon_shower_FromNeutralScore)
        second_max_photon_shower_FromChargedScores.append(second_max_photon_shower_FromChargedScore)
        second_max_photon_shower_CosThetas.append(second_max_photon_shower_CosTheta)
        second_max_photon_shower_CosThetaYs.append(second_max_photon_shower_CosThetaY)
        second_max_photon_shower_DistToVtxs.append(second_max_photon_shower_DistToVtx)
        diphoton_opening_angles.append(diphoton_opening_angle)
        diphoton_energies.append(diphoton_energy)
        diphoton_costhetas.append(diphoton_costheta)
        diphoton_masses.append(diphoton_mass)
        prim_shower_photon_nums.append(curr_num_photons)
        prim_shower_electron_nums.append(curr_num_electrons)
        prim_shower_muon_nums.append(curr_num_muons)
        prim_shower_charged_pion_nums.append(curr_num_charged_pions)
        prim_shower_proton_nums.append(curr_num_protons)

    new_lantern_cols_dic = {}

    new_lantern_cols_dic.update({
        "lantern_max_shower_charge": max_shower_charges,

        "lantern_max_electron_shower_charge": max_electron_shower_charges,
        "lantern_max_electron_shower_ElScore": max_electron_shower_ElScores,
        "lantern_max_electron_shower_PhScore": max_electron_shower_PhScores,
        "lantern_max_electron_shower_MuScore": max_electron_shower_MuScores,
        "lantern_max_electron_shower_PiScore": max_electron_shower_PiScores,
        "lantern_max_electron_shower_PrScore": max_electron_shower_PrScores,
        "lantern_max_electron_shower_electron_confidence": max_electron_shower_electron_confidences,
        "lantern_max_electron_shower_ph_normedscore": max_electron_shower_ph_normedscores,
        "lantern_max_electron_shower_el_normedscore": max_electron_shower_el_normedscores,
        "lantern_max_electron_shower_PrimaryScore": max_electron_shower_PrimaryScores,
        "lantern_max_electron_shower_FromNeutralScore": max_electron_shower_FromNeutralScores,
        "lantern_max_electron_shower_FromChargedScore": max_electron_shower_FromChargedScore,
        "lantern_max_electron_shower_CosTheta": max_electron_shower_CosThetas,
        "lantern_max_electron_shower_CosThetaY": max_electron_shower_CosThetaYs,
        "lantern_max_electron_shower_DistToVtx": max_electron_shower_DistToVtxs,
        "lantern_second_max_electron_shower_charge": second_max_electron_shower_charges,
        "lantern_second_max_electron_shower_PhScore": second_max_electron_shower_PhScores,
        "lantern_second_max_electron_shower_ElScore": second_max_electron_shower_ElScores,
        "lantern_second_max_electron_shower_MuScore": second_max_electron_shower_MuScores,
        "lantern_second_max_electron_shower_PiScore": second_max_electron_shower_PiScores,
        "lantern_second_max_electron_shower_PrScore": second_max_electron_shower_PrScores,
        "lantern_second_max_electron_shower_electron_confidence": second_max_electron_shower_electron_confidences,
        "lantern_second_max_electron_shower_ph_normedscore": second_max_electron_shower_ph_normedscores,
        "lantern_second_max_electron_shower_el_normedscore": second_max_electron_shower_el_normedscores,
        "lantern_second_max_electron_shower_PrimaryScore": second_max_electron_shower_PrimaryScores,
        "lantern_second_max_electron_shower_FromNeutralScore": second_max_electron_shower_FromNeutralScores,
        "lantern_second_max_electron_shower_FromChargedScore": second_max_electron_shower_FromChargedScore,
        "lantern_second_max_electron_shower_CosTheta": second_max_electron_shower_CosThetas,
        "lantern_second_max_electron_shower_CosThetaY": second_max_electron_shower_CosThetaYs,
        "lantern_second_max_electron_shower_DistToVtx": second_max_electron_shower_DistToVtxs,

        "lantern_max_photon_shower_charge": max_photon_shower_charges,
        "lantern_max_photon_shower_PhScore": max_photon_shower_PhScores,
        "lantern_max_photon_shower_ElScore": max_photon_shower_ElScores,
        "lantern_max_photon_shower_MuScore": max_photon_shower_MuScores,
        "lantern_max_photon_shower_PiScore": max_photon_shower_PiScores,
        "lantern_max_photon_shower_PrScore": max_photon_shower_PrScores,
        "lantern_max_photon_shower_electron_confidence": max_photon_shower_electron_confidences,
        "lantern_max_photon_shower_ph_normedscore": max_photon_shower_ph_normedscores,
        "lantern_max_photon_shower_el_normedscore": max_photon_shower_el_normedscores,
        "lantern_max_photon_shower_PrimaryScore": max_photon_shower_PrimaryScores,
        "lantern_max_photon_shower_FromNeutralScore": max_photon_shower_FromNeutralScores,
        "lantern_max_photon_shower_FromChargedScore": max_photon_shower_FromChargedScores,
        "lantern_max_photon_shower_CosTheta": max_photon_shower_CosThetas,
        "lantern_max_photon_shower_CosThetaY": max_photon_shower_CosThetaYs,
        "lantern_max_photon_shower_DistToVtx": max_photon_shower_DistToVtxs,
        "lantern_second_max_photon_shower_charge": second_max_photon_shower_charges,
        "lantern_second_max_photon_shower_PhScore": second_max_photon_shower_PhScores,
        "lantern_second_max_photon_shower_ElScore": second_max_photon_shower_ElScores,
        "lantern_second_max_photon_shower_MuScore": second_max_photon_shower_MuScores,
        "lantern_second_max_photon_shower_PiScore": second_max_photon_shower_PiScores,
        "lantern_second_max_photon_shower_PrScore": second_max_photon_shower_PrScores,
        "lantern_second_max_photon_shower_electron_confidence": second_max_photon_shower_electron_confidences,
        "lantern_second_max_photon_shower_ph_normedscore": second_max_photon_shower_ph_normedscores,
        "lantern_second_max_photon_shower_el_normedscore": second_max_photon_shower_el_normedscores,
        "lantern_second_max_photon_shower_PrimaryScore": second_max_photon_shower_PrimaryScores,
        "lantern_second_max_photon_shower_FromNeutralScore": second_max_photon_shower_FromNeutralScores,
        "lantern_second_max_photon_shower_FromChargedScore": second_max_photon_shower_FromChargedScores,
        "lantern_second_max_photon_shower_CosTheta": second_max_photon_shower_CosThetas,
        "lantern_second_max_photon_shower_CosThetaY": second_max_photon_shower_CosThetaYs,
        "lantern_second_max_photon_shower_DistToVtx": second_max_photon_shower_DistToVtxs,

        "lantern_diphoton_opening_angle": diphoton_opening_angles,
        "lantern_diphoton_energy": diphoton_energies,
        "lantern_diphoton_costheta": diphoton_costhetas,
        "lantern_diphoton_mass": diphoton_masses,

        "lantern_prim_shower_photon_num": prim_shower_photon_nums,
        "lantern_prim_shower_electron_num": prim_shower_electron_nums,
        "lantern_prim_shower_muon_num": prim_shower_muon_nums,
        "lantern_prim_shower_charged_pion_num": prim_shower_charged_pion_nums,
        "lantern_prim_shower_proton_num": prim_shower_proton_nums,

    })


    # see function here: https://github.com/NuTufts/lantern_ana/blob/94f31b83e8a170230ca7b948c701671ab3099bbd/lantern_ana/utils/get_primary_muon_candidates.py
    # this code is similar, but considers all types of tracks, and counts all types of particles

    prim_muon_track_max_muscores = []
    prim_proton_track_max_prscores = []
    prim_charged_pion_track_max_piscores = []

    prim_track_photon_nums = []
    prim_track_electron_nums = []
    prim_track_muon_nums = []
    prim_track_charged_pion_nums = []
    prim_track_proton_nums = []
    
    nTracks = df["lantern_nTracks"].to_numpy()
    trackIsSecondary = df["lantern_trackIsSecondary"].to_numpy()
    trackClassified = df["lantern_trackClassified"].to_numpy()
    trackCharge = df["lantern_trackCharge"].to_numpy()
    #trackCosTheta = df["lantern_trackCosTheta"].to_numpy()
    #trackCosThetaY = df["lantern_trackCosThetaY"].to_numpy()
    #trackDistToVtx = df["lantern_trackDistToVtx"].to_numpy()
    trackComp = df["lantern_trackComp"].to_numpy()
    trackPurity = df["lantern_trackPurity"].to_numpy()
    #trackPrimaryScore = df["lantern_trackPrimaryScore"].to_numpy()
    #trackFromNeutralScore = df["lantern_trackFromNeutralScore"].to_numpy()
    #trackFromChargedScore = df["lantern_trackFromChargedScore"].to_numpy()
    trackPID = df["lantern_trackPID"].to_numpy()
    #trackElScore = df["lantern_trackElScore"].to_numpy()
    #trackPhScore = df["lantern_trackPhScore"].to_numpy()
    trackMuScore = df["lantern_trackMuScore"].to_numpy()
    trackPiScore = df["lantern_trackPiScore"].to_numpy()
    trackPrScore = df["lantern_trackPrScore"].to_numpy()

    for event_i in tqdm(range(len(df)), desc="Analyzing LANTERN tracks"):
        curr_nTracks = nTracks[event_i]
        curr_trackIsSecondary = trackIsSecondary[event_i]
        curr_trackClassified = trackClassified[event_i]
        curr_trackCharge = trackCharge[event_i]
        curr_trackComp = trackComp[event_i]
        curr_trackPurity = trackPurity[event_i]
        curr_trackPID = trackPID[event_i]
        curr_trackMuScore = trackMuScore[event_i]
        curr_trackPiScore = trackPiScore[event_i]
        curr_trackPrScore = trackPrScore[event_i]

        curr_num_photons = 0
        curr_num_electrons = 0
        curr_num_muons = 0
        curr_num_protons = 0
        curr_num_charged_pions = 0

        curr_max_muscore = np.nan   
        curr_max_prscore = np.nan
        curr_max_piscore = np.nan
        
        for track_i in range(curr_nTracks):
            if curr_trackIsSecondary[track_i] != 0: # only considering primary tracks
                continue

            if curr_trackClassified[track_i] != 0: # only considering classified tracks
                continue

            min_charge, min_completeness, min_purity = 0, 0, 0
            if curr_trackCharge[track_i] < min_charge or curr_trackComp[track_i] < min_completeness or curr_trackPurity[track_i] < min_purity:
                continue

            if curr_trackPID[track_i] == 13: # muon
                curr_num_muons += 1
                if curr_trackMuScore[track_i] > curr_max_muscore or curr_max_muscore is np.nan:
                    curr_max_muscore = curr_trackMuScore[track_i]
            elif curr_trackPID[track_i] == 2212: # proton
                curr_num_protons += 1
                if curr_trackPrScore[track_i] > curr_max_prscore or curr_max_prscore is np.nan:
                    curr_max_prscore = curr_trackPrScore[track_i]
            elif curr_trackPID[track_i] == 211: # charged pion
                curr_num_charged_pions += 1
                if curr_trackPiScore[track_i] > curr_max_piscore or curr_max_piscore is np.nan:
                    curr_max_piscore = curr_trackPiScore[track_i]
            elif curr_trackPID[track_i] == 22: # photon
                curr_num_photons += 1
            elif curr_trackPID[track_i] == 11: # electron
                curr_num_electrons += 1
            else:
                assert False, "Invalid track PID"

        prim_track_photon_nums.append(curr_num_photons)
        prim_track_electron_nums.append(curr_num_electrons)
        prim_track_muon_nums.append(curr_num_muons)
        prim_track_charged_pion_nums.append(curr_num_charged_pions)
        prim_track_proton_nums.append(curr_num_protons)

        prim_muon_track_max_muscores.append(curr_max_muscore)
        prim_proton_track_max_prscores.append(curr_max_prscore)
        prim_charged_pion_track_max_piscores.append(curr_max_piscore)

    new_lantern_cols_dic.update({
        "lantern_prim_track_photon_num": prim_track_photon_nums,
        "lantern_prim_track_electron_num": prim_track_electron_nums,
        "lantern_prim_track_muon_num": prim_track_muon_nums,
        "lantern_prim_track_charged_pion_num": prim_track_charged_pion_nums,
        "lantern_prim_track_proton_num": prim_track_proton_nums,

        "lantern_prim_photon_num": np.array(prim_shower_photon_nums) + np.array(prim_track_photon_nums),
        "lantern_prim_electron_num": np.array(prim_shower_electron_nums) + np.array(prim_track_electron_nums),
        "lantern_prim_muon_num": np.array(prim_shower_muon_nums) + np.array(prim_track_muon_nums),
        "lantern_prim_charged_pion_num": np.array(prim_shower_charged_pion_nums) + np.array(prim_track_charged_pion_nums),
        "lantern_prim_proton_num": np.array(prim_shower_proton_nums) + np.array(prim_track_proton_nums),

        "lantern_prim_muon_track_max_muscore": prim_muon_track_max_muscores,
        "lantern_prim_proton_track_max_prscore": prim_proton_track_max_prscores,
        "lantern_prim_charged_pion_track_max_piscore": prim_charged_pion_track_max_piscores,
    })

    df = pd.concat([df, pd.DataFrame(new_lantern_cols_dic, index=df.index)], axis=1)

    return df

def do_combined_postprocessing(df):

    # checked that using the Pandora SCE vertex is further from the WC and Lantern vertices

    wc_vtx_x = df["wc_reco_nuvtxX"].to_numpy()
    wc_vtx_y = df["wc_reco_nuvtxY"].to_numpy()
    wc_vtx_z = df["wc_reco_nuvtxZ"].to_numpy()
    pandora_vtx_x = df["pelee_reco_nu_vtx_x"].to_numpy()
    pandora_vtx_y = df["pelee_reco_nu_vtx_y"].to_numpy()
    pandora_vtx_z = df["pelee_reco_nu_vtx_z"].to_numpy()
    #pandora_vtx_sce_x = df["pelee_reco_nu_vtx_sce_x"].to_numpy()
    #pandora_vtx_sce_y = df["pelee_reco_nu_vtx_sce_y"].to_numpy()
    #pandora_vtx_sce_z = df["pelee_reco_nu_vtx_sce_z"].to_numpy()
    lantern_vtx_x = df["lantern_vtxX"].to_numpy()
    lantern_vtx_y = df["lantern_vtxY"].to_numpy()
    lantern_vtx_z = df["lantern_vtxZ"].to_numpy()

    # replacing default values with nan
    wc_vtx_x[wc_vtx_x == -1] = np.nan
    wc_vtx_y[wc_vtx_y == -1] = np.nan
    wc_vtx_z[wc_vtx_z == -1] = np.nan
    lantern_vtx_x[lantern_vtx_x == -999] = np.nan
    lantern_vtx_y[lantern_vtx_y == -999] = np.nan
    lantern_vtx_z[lantern_vtx_z == -999] = np.nan

    wc_pandora_dist = np.sqrt((wc_vtx_x - pandora_vtx_x)**2 + (wc_vtx_y - pandora_vtx_y)**2 + (wc_vtx_z - pandora_vtx_z)**2)
    #wc_pandora_sce_dist = np.sqrt((wc_vtx_x - pandora_vtx_sce_x)**2 + (wc_vtx_y - pandora_vtx_sce_y)**2 + (wc_vtx_z - pandora_vtx_sce_z)**2)

    wc_lantern_dist = np.sqrt((wc_vtx_x - lantern_vtx_x)**2 + (wc_vtx_y - lantern_vtx_y)**2 + (wc_vtx_z - lantern_vtx_z)**2)

    lantern_pandora_dist = np.sqrt((lantern_vtx_x - pandora_vtx_x)**2 + (lantern_vtx_y - pandora_vtx_y)**2 + (lantern_vtx_z - pandora_vtx_z)**2)
    #lantern_pandora_sce_dist = np.sqrt((lantern_vtx_x - pandora_vtx_sce_x)**2 + (lantern_vtx_y - pandora_vtx_sce_y)**2 + (lantern_vtx_z - pandora_vtx_sce_z)**2)

    df["wc_pandora_dist"] = wc_pandora_dist
    #df["wc_pandora_sce_dist"] = wc_pandora_sce_dist
    df["wc_lantern_dist"] = wc_lantern_dist
    df["lantern_pandora_dist"] = lantern_pandora_dist
    #df["lantern_pandora_sce_dist"] = lantern_pandora_sce_dist

    return df
