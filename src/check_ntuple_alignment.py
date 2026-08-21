"""Check that the per-event trees inside a checkout ntuple are entry-aligned.

The pipeline reads one entry per event from several trees in the same file
(wcpselection/T_eval, nuselection/NeutrinoSelectionFilter, singlephotonana/*,
lantern/EventTree, spline_weights, ...) and assumes entry i of every tree is the
same event.  That assumption silently broke for
checkout_MCC9.10_Run123_v10_04_07_20_BNB_nu_overlay_surprise_reco2_hist_2.root
(found 2026-08-17): its nuselection tree has the same entry count as T_eval, but
its run/sub/evt agree with T_eval on only 18 of 727,013 entries and the last
576,821 entries all carry the single key run 11723 / subrun 216 / event 10850.

This module compares run/subrun/event across every id-carrying tree against
wcpselection/T_eval, over the FULL tree (reading three int branches is well
under a second per million entries), and reports the evidence.

Standalone use (only needs uproot + numpy, works on a bare downloaded file):

    python src/check_ntuple_alignment.py /path/to/checkout_..._hist_2.root [more.root ...]

Exit code is 1 if any file has a misaligned tree.  Inside the pipeline,
``assert_ntuple_trees_aligned(f, filename)`` raises RuntimeError with the same
evidence, so a bad file is caught when it is first opened rather than showing up
as a puzzling join later on.
"""

import sys
import numpy as np
import uproot

REFERENCE_TREE = "wcpselection/T_eval"

# (tree path, (run branch, subrun branch, event branch)) for every per-event
# tree that carries event ids; trees absent from a file are simply skipped.
ID_TREES = [
    ("wcpselection/T_PFeval", ("run", "subrun", "event")),
    ("wcpselection/T_BDTvars", ("run", "subrun", "event")),
    ("nuselection/NeutrinoSelectionFilter", ("run", "sub", "evt")),
    ("singlephotonana/vertex_tree", ("run_number", "subrun_number", "event_number")),
    ("singlephotonana/eventweight_tree", ("run", "subrun", "event")),
    ("lantern/EventTree", ("run", "subrun", "event")),
    ("spline_weights", ("run", "subrun", "event")),
]

# In samples without GENIE truth (beam on/off data, nuwro overlays, the
# delete/isotropic one-gamma samples), gLEE's eventweight_tree is written but
# never filled: its run/subrun/event hold uninitialized garbage and every
# payload branch is at its sentinel.  That is not misalignment -- there is no
# content to misattach -- so when the ids disagree with T_eval we check the
# payload branch below, and if it is the sentinel on every entry the tree is
# reported as unfilled and skipped rather than failing the file.
UNFILLED_SENTINELS = {
    "singlephotonana/eventweight_tree": ("GTruth_gQ2", -9999.0),
}


def _packed_keys(tree, branches, entry_stop=None):
    """run/subrun/event as one opaque 24-byte value per entry (a structured void
    view of an (n, 3) int64 array), so numpy can compare and count them without
    any assumption about the range of the ids -- garbage values stay comparable."""
    arrs = tree.arrays(list(branches), library="np", entry_stop=entry_stop)
    stacked = np.ascontiguousarray(np.stack([arrs[b].astype(np.int64) for b in branches], axis=1))
    return stacked.view(np.dtype((np.void, 24))).reshape(-1)


def _unpack(key):
    run, sub, evt = np.frombuffer(bytes(key), dtype=np.int64)
    return int(run), int(sub), int(evt)


def check_ntuple_alignment(f, entry_stop=None):
    """Compare every id-carrying tree in the open uproot file ``f`` against T_eval.

    Returns a list of dicts, one per tree found (T_eval itself excluded), with:
        tree, n_entries, n_ref, aligned (bool), frac_agree, first_bad_entry,
        n_unique, top_key (run, subrun, event), top_key_count
    ``entry_stop`` limits the comparison to the first N entries (None = whole tree).
    """
    ref_tree = f[REFERENCE_TREE]
    ref = _packed_keys(ref_tree, ("run", "subrun", "event"), entry_stop)
    n_ref = ref_tree.num_entries

    results = []
    for path, branches in ID_TREES:
        if path not in f:
            continue
        tree = f[path]
        res = {"tree": path, "n_entries": tree.num_entries, "n_ref": n_ref}
        if tree.num_entries != n_ref:
            res.update(aligned=False, frac_agree=0.0, first_bad_entry=None,
                       n_unique=None, top_key=None, top_key_count=None)
            results.append(res)
            continue
        keys = _packed_keys(tree, branches, entry_stop)
        agree = keys == ref
        n_agree = int(agree.sum())
        uniq, counts = np.unique(keys, return_counts=True)
        top = int(np.argmax(counts))
        res.update(
            aligned=bool(n_agree == len(keys)),
            frac_agree=n_agree / max(len(keys), 1),
            first_bad_entry=None if n_agree == len(keys) else int(np.argmin(agree)),
            n_unique=int(len(uniq)),
            top_key=_unpack(uniq[top]),
            top_key_count=int(counts[top]),
            unfilled=False,
        )
        if not res["aligned"] and path in UNFILLED_SENTINELS:
            branch, sentinel = UNFILLED_SENTINELS[path]
            vals = tree.arrays([branch], library="np", entry_stop=entry_stop)[branch]
            res["unfilled"] = bool(np.all(vals == sentinel))
        results.append(res)
    return results


def format_report(filename, results):
    lines = [f"{filename}"]
    n_ref = results[0]["n_ref"] if results else "?"
    lines.append(f"  reference {REFERENCE_TREE}: {n_ref} entries")
    for r in results:
        if r["aligned"]:
            lines.append(f"  OK   {r['tree']:<40s} {r['n_entries']:>9d} entries, run/subrun/event match T_eval on every entry")
            continue
        if r.get("unfilled"):
            branch, sentinel = UNFILLED_SENTINELS[r["tree"]]
            lines.append(
                f"  SKIP {r['tree']:<40s} {r['n_entries']:>9d} entries, ids never filled "
                f"(garbage run/subrun/event, {branch} == {sentinel} on every entry -- no GENIE truth in this sample)"
            )
            continue
        if r["n_entries"] != r["n_ref"]:
            lines.append(f"  BAD  {r['tree']:<40s} {r['n_entries']:>9d} entries != {r['n_ref']} in T_eval")
            continue
        run, sub, evt = r["top_key"]
        lines.append(
            f"  BAD  {r['tree']:<40s} {r['n_entries']:>9d} entries, only {r['frac_agree']:.4%} match T_eval "
            f"(first mismatch at entry {r['first_bad_entry']}); {r['n_unique']} unique keys, "
            f"most common run {run} / subrun {sub} / event {evt} appears {r['top_key_count']} times"
        )
    return "\n".join(lines)


def assert_ntuple_trees_aligned(f, filename, entry_stop=None):
    """Raise RuntimeError (with the evidence) if any per-event tree is misaligned."""
    results = check_ntuple_alignment(f, entry_stop)
    bad = [r for r in results if not r["aligned"] and not r.get("unfilled")]
    if bad:
        raise RuntimeError(
            "ntuple trees are not entry-aligned -- refusing to load this file, since every "
            "branch read from the misaligned tree(s) would be attached to the wrong event:\n"
            + format_report(filename, results)
            + "\n(run src/check_ntuple_alignment.py on the file for the same report)"
        )
    return results


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("files", nargs="+", help="ROOT file paths (or bare filenames under data_files_location)")
    parser.add_argument("--head", type=int, default=None, help="only compare the first N entries (default: whole tree)")
    args = parser.parse_args(argv)

    any_bad = False
    for path in args.files:
        try:
            f = uproot.open(path)
        except FileNotFoundError:
            # convenience: bare filename relative to the project's data location
            from file_locations import data_files_location
            f = uproot.open(f"{data_files_location}/{path}")
        results = check_ntuple_alignment(f, args.head)
        print(format_report(path, results), flush=True)
        if any(not r["aligned"] and not r.get("unfilled") for r in results):
            any_bad = True
            print("  ==> MISALIGNED TREES FOUND", flush=True)
        f.close()
    return 1 if any_bad else 0


if __name__ == "__main__":
    sys.exit(main())
