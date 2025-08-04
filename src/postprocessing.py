import numpy as np
import polars as pl
from tqdm import tqdm
from signal_categories import topological_category_queries, topological_category_labels
from signal_categories import physics_category_queries, physics_category_labels
import time

def do_orthogonalization_and_POT_weighting(df, pot_dic, normalizing_POT=1.11e21):

    original_length = df.height

    summed_POT_nc_1pi0 = pot_dic['nc_pi0_overlay'] + pot_dic['nu_overlay']

    print("getting masks...")

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

    print("assigning summed_POT_nc_1pi0 value...")

    # setting nc_pi0_overlay_true_nc_1pi0 and nu_overlay_true_nc_1pi0_mask to summed_POT_nc_1pi0
    # (considering the fact that we use both the NC Pi0 overlay and nu overlay files)
    df = df.with_columns([
        pl.when(nc_pi0_overlay_true_nc_1pi0_mask | nu_overlay_true_nc_1pi0_mask)
        .then(pl.lit(summed_POT_nc_1pi0))
        .otherwise(pl.col("wc_file_POT"))
        .alias("wc_file_POT")
    ])

    # Filter out unwanted events by keeping only the events we want
    combined_mask = nc_pi0_overlay_true_nc_1pi0_mask | nu_overlay_true_nc_1pi0_mask | nu_overlay_other_mask | dirt_mask | ext_mask

    start_time = time.time()
    mask_computed = df.select(combined_mask.alias("mask"))["mask"]
    end_time = time.time()

    print("filtering out unwanted events...", end="", flush=True)
    start_time = time.time()
    temp_df = df.filter(mask_computed)
    end_time = time.time()
    print(f"done ({end_time - start_time:.2f} seconds)")
    

    print("assigning net weights...")

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
    
    print("Adding WC reco particle multiplicity...")

    energy = pl.col("wc_kine_energy_particle").cast(pl.List(pl.Float64), strict=False)
    pdg_abs = pl.col("wc_kine_particle_type").abs()
    sufficient_energy_protons = (energy > 35) & (pdg_abs == 2212)
    sufficient_energy_muons_pions = (energy > 10) & ((pdg_abs == 13) | (pdg_abs == 211))
    df = df.with_columns([
        sufficient_energy_protons.list.sum().alias("wc_reco_num_protons"),
        sufficient_energy_muons_pions.list.sum().alias("wc_reco_num_other_tracks")
    ])


    print("Adding WC truth primary proton number, max energy, and summed energy...")

    pdg = pl.col("wc_truth_pdg")
    mother = pl.col("wc_truth_mother")
    start_p = pl.col("wc_truth_startMomentum")
    index = pl.arr.index()
    is_primary_proton = (
        (pdg.arr.get(index) == 2212) &
        (mother.arr.get(index) == 0)
    )
    energy = start_p.arr.get(index).arr.get(3)
    df = df.with_columns([
        start_p.arr.eval(
            is_primary_proton,
            parallel=True
        ).arr.sum().alias("wc_truth_num_prim_protons"),

        start_p.arr.eval(
            is_primary_proton.then(energy).otherwise(0.0),
            parallel=True
        ).arr.sum().alias("wc_truth_sum_prim_proton_energy"),

        start_p.arr.eval(
            is_primary_proton.then(energy).otherwise(None),
            parallel=True
        ).arr.max().alias("wc_truth_max_prim_proton_energy"),
    ])

    print("Adding WC shower position and angle variables...")

    vtx_x = pl.col("wc_reco_showervtxX")
    vtx_y = pl.col("wc_reco_showervtxY")
    vtx_z = pl.col("wc_reco_showervtxZ")
    shower_momentum = pl.col("wc_reco_showerMomentum")
    px = shower_momentum.arr.get(0)
    py = shower_momentum.arr.get(1)
    pz = shower_momentum.arr.get(2)
    E  = shower_momentum.arr.get(3)

    # Magnitudes
    p_total = (px**2 + py**2 + pz**2).sqrt()
    p_perp = (px**2 + py**2).sqrt()

    # Angular variables
    theta = (p_perp / pz).arctan() * (180 / np.pi)
    phi = pl.atan2(px, py) * (180 / np.pi)

    # Unit vector
    ux = px / p_total
    uy = py / p_total
    uz = pz / p_total

    # Backward projection distances
    x_proj = pl.when(ux > 0).then((vtx_x - (-1.0)) / ux).when(ux < 0).then((vtx_x - 254.3) / ux).otherwise(1e9)
    y_proj = pl.when(uy > 0).then((vtx_y - (-115.0)) / uy).when(uy < 0).then((vtx_y - 117.0) / uy).otherwise(1e9)
    z_proj = pl.when(uz > 0).then((vtx_z - 0.6) / uz).when(uz < 0).then((vtx_z - 1036.4) / uz).otherwise(1e9)
    backproj_dist = pl.min_horizontal([x_proj, y_proj, z_proj])

    dist_to_boundary = pl.min_horizontal([
        (vtx_x - (-1.0)).abs(),
        (vtx_x - 254.3).abs(),
        (vtx_y - (-115.0)).abs(),
        (vtx_y - 117.0).abs(),
        (vtx_z - 0.6).abs(),
        (vtx_z - 1036.4).abs()
    ])

    df = df.with_columns([
        theta.alias("wc_shower_theta"),
        phi.alias("wc_shower_phi"),
        dist_to_boundary.alias("wc_reco_distance_to_boundary"),
        backproj_dist.alias("wc_reco_backwards_projected_dist")
    ])

    return df

def do_blip_postprocessing(df):

    print("Separating blips by PDG code...")

    pdg = pl.col("blip_true_pdg")
    x = pl.col("blip_x")
    y = pl.col("blip_y")
    z = pl.col("blip_z")
    energy = pl.col("blip_energy")
    i = pl.arr.index()

    is_electron = pdg.arr.get(i) == 11
    is_proton = pdg.arr.get(i) == 2212
    is_other = ~(is_electron | is_proton)

    return df.with_columns([

        pl.col("blip_x").arr.eval(is_electron.then(pl.element()).otherwise(None), parallel=True).drop_nulls().alias("electron_blip_x"),
        pl.col("blip_y").arr.eval(is_electron.then(pl.element()).otherwise(None), parallel=True).drop_nulls().alias("electron_blip_y"),
        pl.col("blip_z").arr.eval(is_electron.then(pl.element()).otherwise(None), parallel=True).drop_nulls().alias("electron_blip_z"),
        pl.col("blip_energy").arr.eval(is_electron.then(pl.element()).otherwise(None), parallel=True).drop_nulls().alias("electron_blip_energy"),
        
        pl.col("blip_x").arr.eval(is_proton.then(pl.element()).otherwise(None), parallel=True).drop_nulls().alias("proton_blip_x"),
        pl.col("blip_y").arr.eval(is_proton.then(pl.element()).otherwise(None), parallel=True).drop_nulls().alias("proton_blip_y"),
        pl.col("blip_z").arr.eval(is_proton.then(pl.element()).otherwise(None), parallel=True).drop_nulls().alias("proton_blip_z"),
        pl.col("blip_energy").arr.eval(is_proton.then(pl.element()).otherwise(None), parallel=True).drop_nulls().alias("proton_blip_energy"),

        pl.col("blip_x").arr.eval(is_other.then(pl.element()).otherwise(None), parallel=True).drop_nulls().alias("other_blip_x"),
        pl.col("blip_y").arr.eval(is_other.then(pl.element()).otherwise(None), parallel=True).drop_nulls().alias("other_blip_y"),
        pl.col("blip_z").arr.eval(is_other.then(pl.element()).otherwise(None), parallel=True).drop_nulls().alias("other_blip_z"),
        pl.col("blip_energy").arr.eval(is_other.then(pl.element()).otherwise(None), parallel=True).drop_nulls().alias("other_blip_energy"),

    ])


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
