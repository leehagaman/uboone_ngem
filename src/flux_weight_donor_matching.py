"""Borrow flux multisim weights for events that have none, by true neutrino energy.

Some samples in the prediction carry no flux systematic universes: the
NC_coherent_1g_reweighted rows are isotropic photon-gun events reweighted to the
NC coherent 1g shape, so their ROOT files have no `weights` map.  The BNB flux
uncertainty is, to a very good approximation, only a function of neutrino flavor and
true neutrino energy (the flux varies negligibly across the detector), so each such
"recipient" event can take its flux universe vector from a "donor" nu_overlay event of
the same flavor and (nearly) the same true neutrino energy.

Design notes
------------
* Donors are drawn from the K nearest neighbours in true E_nu rather than the single
  nearest one, so the ~16k coherent recipients do not all reuse the same handful of
  donor vectors (which would make the per-bin covariance noisy).  The K-neighbour
  window is only a few MeV wide for the bulk of the BNB spectrum, so the flux weight
  is effectively evaluated at the recipient's own energy.
* The donor pool is built from the light columns only (keys + E_nu); the wide
  List[Float32] universe columns are read only for the donors actually chosen, joined
  per weight part, so the full weights dataframe is never materialised.
* Written generically (recipient df + donor filetype/pdg + list of weight columns) so
  it can be reused for a future numu -> nue "fullosc" sample that starts from numuCC
  flux weights.
"""

import numpy as np
import polars as pl

JOIN_KEYS = ["filename", "run", "subrun", "event"]


def is_flux_weight_column(name):
    """True for every column that holds BNB flux universes/variations:

    * the combined multisims in presel_weights_df ("flux_all", "ppfx_all"), and
    * the per-knob columns in spline_weights_df, named <knob>_<EventWeight type> from
      microboone_eventweight_flux: horncurrent/expskin/pion*/nucleon* (FluxUnisim),
      piplus/piminus (PrimaryHadronSWCentralSplineVariation), kplus
      (PrimaryHadronFeynmanScaling), kzero (PrimaryHadronSanfordWang), kminus
      (PrimaryHadronNormalization).
    """
    return (name in ("flux_all", "ppfx_all")
            or name.endswith("_FluxUnisim")
            or "_PrimaryHadron" in name)


def flux_weight_columns(column_names):
    return [c for c in column_names if is_flux_weight_column(c)]


def build_donor_pool(weight_parts, presel_df_path, donor_filetype="nu_overlay", donor_nu_pdg=14,
                     nu_energy_col="wc_truth_nuEnergy"):
    """Collect (keys, true E_nu) for every preselected event of `donor_filetype` with
    |wc_truth_nuPdg| == donor_nu_pdg, across the per-file weight parquet parts.

    Returns an eager DataFrame with JOIN_KEYS + nu_energy_col + "weight_part" (the path
    of the part holding that donor's universe columns), sorted by nu_energy_col."""
    presel_lf = (
        pl.scan_parquet(presel_df_path)
        .filter((pl.col("filetype") == donor_filetype) & (pl.col("wc_truth_nuPdg").abs() == donor_nu_pdg))
        .select(JOIN_KEYS + [nu_energy_col])
    )
    pools = []
    for wp in weight_parts:
        part_lf = pl.scan_parquet(wp)
        # cheap filetype check on one row (each part is a single ROOT file)
        ft = part_lf.select("filetype").head(1).collect()
        if ft.height == 0 or ft["filetype"][0] != donor_filetype:
            continue
        keys_lf = part_lf.select(JOIN_KEYS)
        pools.append(
            keys_lf.join(presel_lf, on=JOIN_KEYS, how="inner")
            .with_columns(pl.lit(wp).alias("weight_part"))
            .collect()
        )
    if not pools:
        raise ValueError(f"No {donor_filetype} weight parts found to donate flux weights from!")
    pool = pl.concat(pools, how="vertical")
    # fully deterministic order (ties broken by keys) so that the weight-part and
    # spline-part passes choose identical donors for identical recipients
    pool = (pool.filter(pl.col(nu_energy_col).is_finite() & (pl.col(nu_energy_col) > 0))
            .sort([nu_energy_col] + JOIN_KEYS))
    return pool


def choose_donors(recipient_nu_energy, pool_nu_energy_sorted, k_nearest=20, seed=42):
    """For each recipient energy, pick one donor index at random from the k_nearest
    donors closest in energy (pool_nu_energy_sorted must be ascending).

    Returns (donor_indices, donor_energy_offsets) as numpy arrays."""
    pool = np.asarray(pool_nu_energy_sorted, dtype=np.float64)
    rec = np.asarray(recipient_nu_energy, dtype=np.float64)
    n = len(pool)
    if n < k_nearest:
        raise ValueError(f"donor pool ({n}) smaller than k_nearest ({k_nearest})")
    rng = np.random.default_rng(seed)

    # window of k_nearest donors centred on the insertion point, clipped to the pool
    ins = np.searchsorted(pool, rec)
    start = np.clip(ins - k_nearest // 2, 0, n - k_nearest)
    donor_idx = start + rng.integers(0, k_nearest, size=len(rec))
    return donor_idx, pool[donor_idx] - rec


def borrow_flux_weights_by_true_nu_energy(recipient_df, weight_parts, presel_df_path,
                                          donor_filetype="nu_overlay", donor_nu_pdg=14,
                                          weight_cols=None,
                                          nu_energy_col="wc_truth_nuEnergy",
                                          k_nearest=20, seed=42, keep_diagnostics=True, verbose=True):
    """Return `recipient_df` with `weight_cols` (List[Float32] universe columns) copied
    from energy-matched donors.  With keep_diagnostics, also adds
    `flux_donor_nuEnergy` (donor true E_nu, MeV) and `flux_donor_filename`/`_run`/
    `_subrun`/`_event` (donor keys, so a donor can be traced back); switch it off for
    outputs whose every column becomes a ROOT branch (the spline parquet).

    weight_cols defaults to every flux column found in the first weight part (see
    is_flux_weight_column), so the same call works for presel_weights_df parts
    (flux_all/ppfx_all) and spline_weights_df parts (the 13 per-knob columns).

    The donor choice is deterministic given (recipient energies, donor pool, seed), and
    the weight and spline parts hold exactly the same preselected events, so calling
    this once per parquet family with the same seed pairs each recipient with the SAME
    donor in both -- the flux_all universes and the per-knob variations stay consistent.

    recipient_df must be eager and carry JOIN_KEYS + nu_energy_col.  Recipients with a
    non-finite or non-positive energy raise, since they cannot be matched."""
    if weight_cols is None:
        weight_cols = flux_weight_columns(pl.scan_parquet(weight_parts[0]).collect_schema().names())
        if not weight_cols:
            raise ValueError(f"no flux weight columns found in {weight_parts[0]}")
    rec_e = recipient_df[nu_energy_col].to_numpy().astype(np.float64)
    if not np.all(np.isfinite(rec_e) & (rec_e > 0)):
        bad = int(np.sum(~(np.isfinite(rec_e) & (rec_e > 0))))
        raise ValueError(f"{bad} recipient events have non-finite or non-positive {nu_energy_col}; cannot match flux donors")

    pool = build_donor_pool(weight_parts, presel_df_path, donor_filetype, donor_nu_pdg, nu_energy_col)
    if verbose:
        print(f"  flux donor pool: {pool.height:,} {donor_filetype} events with |nuPdg|=={donor_nu_pdg} "
              f"from {pool['weight_part'].n_unique()} weight parts, "
              f"E_nu range [{pool[nu_energy_col].min():.0f}, {pool[nu_energy_col].max():.0f}] MeV")

    donor_idx, offsets = choose_donors(rec_e, pool[nu_energy_col].to_numpy(), k_nearest=k_nearest, seed=seed)
    if verbose:
        print(f"  matched {len(rec_e):,} recipients to donors (k_nearest={k_nearest}); "
              f"|E_nu offset| median {np.median(np.abs(offsets)):.1f} MeV, "
              f"95% {np.percentile(np.abs(offsets), 95):.1f} MeV, max {np.max(np.abs(offsets)):.1f} MeV; "
              f"{len(np.unique(donor_idx)):,} distinct donors used")

    donors = pool[donor_idx.tolist()].select(JOIN_KEYS + [nu_energy_col, "weight_part"])
    donors = donors.with_columns(pl.arange(0, donors.height, eager=True).alias("_recipient_row"))

    # fetch the universe columns for the chosen donors, one weight part at a time
    fetched = []
    for wp in donors["weight_part"].unique().to_list():
        d_wp = donors.filter(pl.col("weight_part") == wp).select(JOIN_KEYS + ["_recipient_row"])
        got = (
            pl.scan_parquet(wp).select(JOIN_KEYS + list(weight_cols))
            .join(d_wp.lazy(), on=JOIN_KEYS, how="inner")
            .collect()
        )
        fetched.append(got)
    fetched = pl.concat(fetched, how="vertical").sort("_recipient_row")
    if fetched.height != donors.height:
        raise ValueError(f"fetched {fetched.height} donor weight rows for {donors.height} recipients "
                         "(duplicate or missing donor keys in the weight parts?)")

    out = recipient_df.with_columns(pl.arange(0, recipient_df.height, eager=True).alias("_recipient_row"))
    diag = donors.select([
        "_recipient_row",
        pl.col(nu_energy_col).cast(pl.Float32).alias("flux_donor_nuEnergy"),
        pl.col("filename").alias("flux_donor_filename"),
        pl.col("run").alias("flux_donor_run"),
        pl.col("subrun").alias("flux_donor_subrun"),
        pl.col("event").alias("flux_donor_event"),
    ])
    out = out.join(fetched.select(["_recipient_row"] + list(weight_cols)), on="_recipient_row", how="left")
    if keep_diagnostics:
        out = out.join(diag, on="_recipient_row", how="left")
    out = out.drop("_recipient_row")
    for c in weight_cols:
        if out[c].null_count() != 0:
            raise ValueError(f"{out[c].null_count()} recipients ended up without a {c} donor vector")
    return out
