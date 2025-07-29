import numpy as np
import polars as pl
from tqdm import tqdm
from signal_categories import topological_category_queries, topological_category_labels
from signal_categories import physics_category_queries, physics_category_labels


def do_orthogonalization_and_POT_weighting(df, pot_dic, normalizing_POT=1.11e21):

    original_length = df.height

    summed_POT_nc_1pi0 = pot_dic['nc_pi0_overlay'] + pot_dic['nu_overlay']

    # Get masks for different event types using polars expressions
    nc_pi0_overlay_true_nc_1pi0_mask = (
        (pl.col("filetype") == 'nc_pi0_overlay') & 
        (pl.col("wc_truth_isCC") == 0) & (pl.col("wc_truth_NprimPio") == 1) & (pl.col("wc_truth_vtxInside") == 1)
    )
    nu_overlay_true_nc_1pi0_mask = (
        (pl.col("filetype") == 'nu_overlay') & 
        (pl.col("wc_truth_isCC") == 0) & (pl.col("wc_truth_NprimPio") == 1) & (pl.col("wc_truth_vtxInside") == 1)
    )
    nu_overlay_other_mask = (
        (pl.col("filetype") == 'nu_overlay') & 
        ~((pl.col("wc_truth_isCC") == 0) & (pl.col("wc_truth_NprimPio") == 1) & (pl.col("wc_truth_vtxInside") == 1))
    )
    dirt_mask = pl.col("filetype") == 'dirt_overlay'
    ext_mask = pl.col("filetype") == 'ext'


    # TODO HERE
    # setting the POTs in order to combine the NC Pi0 overlay and nu overlay files without throwing away MC statistics
    df.loc[nc_pi0_overlay_true_nc_1pi0_mask, "wc_file_POT"] = summed_POT_nc_1pi0
    df.loc[nu_overlay_true_nc_1pi0_mask, "wc_file_POT"] = summed_POT_nc_1pi0


    # Filter out unwanted events by keeping only the events we want
    combined_mask = nc_pi0_overlay_true_nc_1pi0_mask | nu_overlay_true_nc_1pi0_mask | nu_overlay_other_mask | dirt_mask | ext_mask
    
    df = df.filter(combined_mask)

    # Calculate net weights using vectorized operations
    df = df.with_columns([
        # Calculate base weight
        (pl.col("wc_weight_cv") * pl.col("wc_weight_spline")).alias("genie_weight"),
    ]).with_columns([
        # Clean up bad weights and apply POT normalization
        pl.when(
            (pl.col("genie_weight") <= 0.0) | 
            (pl.col("genie_weight") > 30.0) | 
            pl.col("genie_weight").is_null() | 
            pl.col("genie_weight").is_infinite()
        )
        .then(pl.lit(1.0))
        .otherwise(pl.col("genie_weight"))
        .alias("cleaned_genie_weight")
    ]).with_columns([
        # Apply POT weighting
        (pl.col("cleaned_genie_weight") * normalizing_POT / pl.col("wc_file_POT")).alias("wc_net_weight")
    ]).drop(["genie_weight", "cleaned_genie_weight"])

    final_length = df.height

    print(f"When combining different file types, went from {original_length} to {final_length} events")

    return df


def do_wc_postprocessing(df):

    # Helper function for proton and other track counting
    def count_reco_particles(energy_list, pdg_list):
        if isinstance(energy_list, float) and np.isnan(energy_list):
            return (np.nan, np.nan)
        
        if energy_list is None or pdg_list is None:
            return (np.nan, np.nan)
            
        proton_num = 0
        other_track_num = 0
        
        for i in range(len(energy_list)):
            if abs(pdg_list[i]) == 2212:
                if energy_list[i] > 35:  # 35 MeV reco proton kinetic energy threshold
                    proton_num += 1
            if abs(pdg_list[i]) == 13 or abs(pdg_list[i]) == 211:  # see N_tracks at https://github.com/BNLIF/wcp-uboone-bdt/blob/main/inc/WCPLEEANA/cuts.h
                if energy_list[i] > 10.:
                    other_track_num += 1
        return (proton_num, other_track_num)

    # Helper function for truth primary proton energy
    def get_max_prim_proton_energy(truth_pdg_list, truth_mother_list, truth_startMomentum_list):
        if isinstance(truth_pdg_list, float) and np.isnan(truth_pdg_list):
            return -1
        
        if truth_pdg_list is None or truth_mother_list is None or truth_startMomentum_list is None:
            return -1
            
        max_prim_proton_energy = 0
        for j in range(len(truth_pdg_list)):
            if truth_mother_list[j] == 0 and truth_pdg_list[j] == 2212:  # primary proton
                max_prim_proton_energy = max(truth_startMomentum_list[j][3] * 1000. - 938.272089, max_prim_proton_energy)
        return max_prim_proton_energy

    # Helper function for shower calculations
    def calculate_shower_variables(reco_shower_momentum, reco_nu_vtx_x, reco_nu_vtx_y, reco_nu_vtx_z):
        if isinstance(reco_shower_momentum, float) and np.isnan(reco_shower_momentum):
            return (np.nan, np.nan, np.nan, np.nan)
        
        if reco_shower_momentum is None:
            return (np.nan, np.nan, np.nan, np.nan)

        reco_shower_momentum_0 = reco_shower_momentum[0]
        reco_shower_momentum_1 = reco_shower_momentum[1]
        reco_shower_momentum_2 = reco_shower_momentum[2]
        reco_shower_momentum_3 = reco_shower_momentum[3]
        
        if reco_shower_momentum_3 <= 0:
            theta = np.nan
            phi = np.nan
            backwards_projected_dist = np.nan
        else:
            reco_shower_momentum_perp = np.sqrt(reco_shower_momentum_0 * reco_shower_momentum_0 + 
                                                reco_shower_momentum_1 * reco_shower_momentum_1)
            theta = np.arctan2(reco_shower_momentum_perp, reco_shower_momentum_2) * 180. / np.pi
            phi = np.arctan2(reco_shower_momentum_0, reco_shower_momentum_1) * 180. / np.pi

            shower_momentum_total_3d = np.sqrt(reco_shower_momentum_0 * reco_shower_momentum_0 + 
                                              reco_shower_momentum_1 * reco_shower_momentum_1 + 
                                              reco_shower_momentum_2 * reco_shower_momentum_2)
            shower_unit_vector_3d = [reco_shower_momentum_0 / shower_momentum_total_3d, 
                                    reco_shower_momentum_1 / shower_momentum_total_3d, 
                                    reco_shower_momentum_2 / shower_momentum_total_3d]

            min_backwards_projected_dist = 1e9
                    
            # projecting to x walls
            if shower_unit_vector_3d[0] > 0:
                min_backwards_projected_dist = min(min_backwards_projected_dist, (reco_nu_vtx_x - (-1)) / shower_unit_vector_3d[0])
            elif shower_unit_vector_3d[0] < 0:
                min_backwards_projected_dist = min(min_backwards_projected_dist, (reco_nu_vtx_x - (254.3)) / shower_unit_vector_3d[0])
                
            # projecting to y walls
            if shower_unit_vector_3d[1] > 0:
                min_backwards_projected_dist = min(min_backwards_projected_dist, (reco_nu_vtx_y - (-115.)) / shower_unit_vector_3d[1])
            elif shower_unit_vector_3d[1] < 0:
                min_backwards_projected_dist = min(min_backwards_projected_dist, (reco_nu_vtx_y - (117.)) / shower_unit_vector_3d[1])
                
            # projecting to z walls
            if shower_unit_vector_3d[2] > 0:
                min_backwards_projected_dist = min(min_backwards_projected_dist, (reco_nu_vtx_z - (0.6)) / shower_unit_vector_3d[2])
            elif shower_unit_vector_3d[2] < 0:
                min_backwards_projected_dist = min(min_backwards_projected_dist, (reco_nu_vtx_z - (1036.4)) / shower_unit_vector_3d[2])

            backwards_projected_dist = min_backwards_projected_dist
            
        distance_to_boundary = np.min([
            abs(reco_nu_vtx_x - (-1.)),
            abs(reco_nu_vtx_x - (254.3)),
            abs(reco_nu_vtx_y - (-115.)),
            abs(reco_nu_vtx_y - (117.)),
            abs(reco_nu_vtx_z - (0.6)),
            abs(reco_nu_vtx_z - (1036.4))
        ])
        
        return (theta, phi, distance_to_boundary, backwards_projected_dist)

    print("Adding WC reco particle multiplicity...")
    particle_counts = df.select([
        pl.col("wc_kine_energy_particle"), 
        pl.col("wc_kine_particle_type")
    ]).map_rows(lambda x: count_reco_particles(x[0], x[1]))
    
    df = df.with_columns([
        particle_counts.map_elements(lambda x: x[0], return_dtype=pl.Int32).alias("wc_reco_num_protons"),
        particle_counts.map_elements(lambda x: x[1], return_dtype=pl.Int32).alias("wc_reco_num_other_tracks")
    ])

    print("Adding WC truth primary proton energy...")
    max_proton_energies = df.select([
        pl.col("wc_truth_pdg"), 
        pl.col("wc_truth_mother"), 
        pl.col("wc_truth_startMomentum")
    ]).map_rows(lambda x: get_max_prim_proton_energy(x[0], x[1], x[2]))
    
    df = df.with_columns([
        max_proton_energies.alias("wc_true_max_prim_proton_energy")
    ])

    print("Adding WC shower position and angle variables...")
    shower_vars = df.select([
        pl.col("wc_reco_showerMomentum"),
        pl.col("wc_reco_showervtxX"),
        pl.col("wc_reco_showervtxY"),
        pl.col("wc_reco_showervtxZ")
    ]).map_rows(lambda x: calculate_shower_variables(x[0], x[1], x[2], x[3]))
    
    df = df.with_columns([
        shower_vars.map_elements(lambda x: x[0], return_dtype=pl.Float64).alias("wc_reco_shower_theta"),
        shower_vars.map_elements(lambda x: x[1], return_dtype=pl.Float64).alias("wc_reco_shower_phi"),
        shower_vars.map_elements(lambda x: x[2], return_dtype=pl.Float64).alias("wc_reco_distance_to_boundary"),
        shower_vars.map_elements(lambda x: x[3], return_dtype=pl.Float64).alias("wc_reco_backwards_projected_dist")
    ])

    return df


def do_blip_postprocessing(df):

    # Helper function to separate blips by particle type
    def separate_blips_by_pdg(blip_pdgs, blip_x, blip_y, blip_z, blip_energy):
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

        if isinstance(blip_pdgs, float) and np.isnan(blip_pdgs):
            return (electron_blips_x, electron_blips_y, electron_blips_z, electron_blips_energy,
                    proton_blips_x, proton_blips_y, proton_blips_z, proton_blips_energy,
                    other_blips_x, other_blips_y, other_blips_z, other_blips_energy)
        
        if blip_pdgs is None:
            return (electron_blips_x, electron_blips_y, electron_blips_z, electron_blips_energy,
                    proton_blips_x, proton_blips_y, proton_blips_z, proton_blips_energy,
                    other_blips_x, other_blips_y, other_blips_z, other_blips_energy)

        for blip_index in range(len(blip_pdgs)):
            if blip_pdgs[blip_index] == 11:
                electron_blips_x.append(blip_x[blip_index])
                electron_blips_y.append(blip_y[blip_index])
                electron_blips_z.append(blip_z[blip_index])
                electron_blips_energy.append(blip_energy[blip_index])
            elif blip_pdgs[blip_index] == 2212:
                proton_blips_x.append(blip_x[blip_index])
                proton_blips_y.append(blip_y[blip_index])
                proton_blips_z.append(blip_z[blip_index])
                proton_blips_energy.append(blip_energy[blip_index])
            else:
                other_blips_x.append(blip_x[blip_index])
                other_blips_y.append(blip_y[blip_index])
                other_blips_z.append(blip_z[blip_index])
                other_blips_energy.append(blip_energy[blip_index])

        return (electron_blips_x, electron_blips_y, electron_blips_z, electron_blips_energy,
                proton_blips_x, proton_blips_y, proton_blips_z, proton_blips_energy,
                other_blips_x, other_blips_y, other_blips_z, other_blips_energy)

    print("Adding separate e/p/other blips...")
    # Apply blip separation
    blip_results = df.select([
        pl.col("blip_true_pdg"),
        pl.col("blip_x"),
        pl.col("blip_y"),
        pl.col("blip_z"),
        pl.col("blip_energy")
    ]).map_rows(lambda x: separate_blips_by_pdg(x[0], x[1], x[2], x[3], x[4]))
    
    blip_labels = [
        "electron_blip_x", "electron_blip_y", "electron_blip_z", "electron_blip_energy",
        "proton_blip_x", "proton_blip_y", "proton_blip_z", "proton_blip_energy",
        "other_blip_x", "other_blip_y", "other_blip_z", "other_blip_energy"
    ]

    df = df.with_columns([
        blip_results.map_elements(lambda x, i=i: x[i], return_dtype=pl.List(pl.Float64)).alias(name)
        for i, name in enumerate(blip_labels)
    ])

    return df


def add_extra_true_photon_variables(df):
    num_infinite_loops_broken = 0

    # Helper function to process photon variables for a single event
    def process_photon_variables(truth_id, truth_pdg, truth_mother, truth_startMomentum, truth_startXYZT):
        if isinstance(truth_id, float) and np.isnan(truth_id):
            return (np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)
        
        if truth_id is None:
            return (np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)

        num_particles = len(truth_id)
        
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
            if truth_pdg[i] == 111:
                pi0_ids.append(truth_id[i])

        primary_or_pi0_gamma_ids = []
        for i in range(num_particles):
            if truth_mother[i] in pi0_ids or truth_mother[i] == 0:  # this is a daughter of a pi0 or a primary particle
                if truth_pdg[i] == 22:  # this is a photon from a pi0 or a primary photon (most likely from an eta or Delta radiative)
                    curr_true_num_gamma += 1
                    curr_true_gamma_energies.append(truth_startMomentum[i][3])
                    primary_or_pi0_gamma_ids.append(truth_id[i])

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
            if truth_id[i] in primary_or_pi0_gamma_ids:  # pi0/primary -> gamma, this won't include the manually deleted photon

                original_gamma_energy = truth_startMomentum[i][3]
                cumulative_deposited_energy = 0

                visited_ids = set()
                iteration_count = 0
                
                while True:
                    curr_id = truth_id[i]
                    
                    if curr_id in visited_ids:
                        num_infinite_loops_broken += 1
                        break

                    visited_ids.add(curr_id)
                    iteration_count += 1
                    
                    descendants_ids = []
                    descendants_indices = []
                    descendants_pdgs = []
                    for j in range(num_particles):
                        if truth_mother[j] == curr_id:  # pi0/primary -> gamma -> this particle
                            descendants_ids.append(truth_id[j])
                            descendants_indices.append(j)
                            descendants_pdgs.append(truth_pdg[j])

                    for descendant_i in range(len(descendants_indices)):
                        if abs(descendants_pdgs[descendant_i]) == 11:  # electron/positron daughter
                            cumulative_deposited_energy += truth_startMomentum[descendants_indices[descendant_i]][3]

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
                    photon_id = truth_id[i]
                    position = photon_id_to_position[photon_id]
                    
                    curr_true_gamma_pairconversion_xs[position] = truth_startXYZT[descendants_indices[0]][0]
                    curr_true_gamma_pairconversion_ys[position] = truth_startXYZT[descendants_indices[0]][1]
                    curr_true_gamma_pairconversion_zs[position] = truth_startXYZT[descendants_indices[0]][2]
                    curr_true_num_gamma_pairconvert += 1

                    if -1 < curr_true_gamma_pairconversion_xs[position] <= 254.3 and -115.0 < curr_true_gamma_pairconversion_ys[position] <= 117.0 and 0.6 < curr_true_gamma_pairconversion_zs[position] <= 1036.4:
                        curr_true_num_gamma_pairconvert_in_FV += 1

                        if original_gamma_energy > 0.02:
                            curr_true_num_gamma_pairconvert_in_FV_20_MeV += 1

        return (curr_true_num_gamma, curr_true_gamma_energies, 
                curr_true_gamma_pairconversion_xs, curr_true_gamma_pairconversion_ys, 
                curr_true_gamma_pairconversion_zs, curr_true_num_gamma_pairconvert,
                curr_true_num_gamma_pairconvert_in_FV, curr_true_num_gamma_pairconvert_in_FV_20_MeV)

    print("Adding true photon variables...")
    # Apply photon variable processing
    photon_results = df.select([
        pl.col("wc_truth_id"),
        pl.col("wc_truth_pdg"),
        pl.col("wc_truth_mother"),
        pl.col("wc_truth_startMomentum"),
        pl.col("wc_truth_startXYZT")
    ]).map_rows(lambda x: process_photon_variables(x[0], x[1], x[2], x[3], x[4]))

    if num_infinite_loops_broken > 0:
        print(f"Broke infinite loops in the true gamma daughter search {num_infinite_loops_broken} / {df.shape[0]} times")

    true_gamma_vars = [
        "true_num_gamma",
        "true_gamma_energies",
        "true_gamma_pairconversion_xs",
        "true_gamma_pairconversion_ys",
        "true_gamma_pairconversion_zs",
        "true_num_gamma_pairconvert",
        "true_num_gamma_pairconvert_in_FV",
        "true_num_gamma_pairconvert_in_FV_20_MeV"
    ]

    df = df.with_columns([
        photon_results.map_elements(lambda x, i=i: x[i], return_dtype=pl.List(pl.Float64)).alias(name)
        for i, name in enumerate(true_gamma_vars)
    ])

    return df


def add_signal_categories(all_df):

    # Add derived boolean columns that are used in signal category definitions
    all_df = all_df.with_columns([
        # Normal overlay (not dirt or ext)
        (~(pl.col("filetype") == "dirt_overlay") & ~(pl.col("filetype") == "ext")).alias("normal_overlay"),
        
        # Truth in FV
        (pl.col("wc_truth_vtxInside") == 1).alias("wc_truth_inFV"),
        
        # Gamma counts in FV 
        (pl.col("true_num_gamma_pairconvert_in_FV") == 0).alias("wc_truth_0g"),
        (pl.col("true_num_gamma_pairconvert_in_FV") == 1).alias("wc_truth_1g"),
        (pl.col("true_num_gamma_pairconvert_in_FV") == 2).alias("wc_truth_2g"),
        (pl.col("true_num_gamma_pairconvert_in_FV") >= 3).alias("wc_truth_3plusg"),
        
        # Proton counts
        (pl.col("wc_true_max_prim_proton_energy") >= 35).alias("wc_truth_Np"),
        (pl.col("wc_true_max_prim_proton_energy") < 35).alias("wc_truth_0p"),
        
        # Muon counts
        ((pl.col("wc_truth_isCC").cast(pl.Boolean)) & 
         (pl.col("wc_truth_nuPdg").abs() == 14)).alias("wc_truth_1mu"),
        (~((pl.col("wc_truth_isCC").cast(pl.Boolean)) & 
           (pl.col("wc_truth_nuPdg").abs() == 14))).alias("wc_truth_0mu"),
        
        # Pi0 counts  
        (pl.col("wc_truth_NprimPio") == 0).alias("wc_truth_0pi0"),
        (pl.col("wc_truth_NprimPio") == 1).alias("wc_truth_1pi0"),
        (pl.col("wc_truth_NprimPio") > 1).alias("wc_truth_multi_pi0"),
        
        # Add one pairconvert in FV 20 MeV
        (pl.col("true_num_gamma_pairconvert_in_FV_20_MeV") == 1).alias("true_one_pairconvert_in_FV_20_MeV")
    ])

    # assigning topological signal categories
    topological_category_conditions = [all_df.eval(query) for query in topological_category_queries]
    for i1, condition1 in enumerate(topological_category_conditions):
        for i2, condition2 in enumerate(topological_category_conditions):
            if i1 != i2:
                overlap = all_df.filter(condition1 & condition2)
                if overlap.height > 0:
                    print(f"Overlapping inFV topological signal definitions: {topological_category_labels[i1]} and {topological_category_labels[i2]}")
                    print(overlap.select(["filetype", "wc_truth_inFV", "wc_truth_1g", "wc_truth_Np", "wc_truth_0mu"]).head(2))
                    raise AssertionError(f"Overlapping inFV topological signal definitions: {topological_category_labels[i1]} and {topological_category_labels[i2]}")

    # TODO: Construct conditions from signal_categories.py

    # Apply topological signal categories
    topological_when_chain = pl.when(topological_conditions[0]).then(pl.lit(topological_category_labels[0]))
    for i in range(1, len(topological_conditions)):
        topological_when_chain = topological_when_chain.when(topological_conditions[i]).then(pl.lit(topological_category_labels[i]))
    
    all_df = all_df.with_columns([
        topological_when_chain.otherwise(pl.lit("other")).alias("topological_signal_category")
    ])

    # Check for uncategorized events
    uncategorized_df = all_df.filter(pl.col("topological_signal_category") == "other")
    assert uncategorized_df.height == 0, "Uncategorized topological signal categories!"

    print("\ntopological signal categories:")
    for category in topological_category_labels:
        curr_df = all_df.filter(pl.col("topological_signal_category") == category)
        unweighted_num = curr_df.height
        weighted_num = curr_df.get_column("wc_net_weight").sum()
        print(f"{category}: {weighted_num:.2f} ({unweighted_num})")

    
    # TODO: Same for physics categories

    return all_df
