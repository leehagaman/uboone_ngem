"""Area-proportional circle Venn diagrams and overlap matrices for weighted event selections.

Everything works from a dict of *exclusive* region weights, keyed by a membership tuple of bools (one per
set): e.g. for sets (A, B, C), regions[(True, False, True)] is the summed weight of events in A and C but
not B.  `region_weights_from_membership` builds that dict from per-event boolean membership columns.

  * draw_venn2_exact      two circles, exactly area-proportional (closed-form lens area + bisection)
  * draw_venn_fitted      2-4 circles fitted so each region's area matches its weight (eulerr-style least
                          squares on the region areas); circles cannot represent every configuration of 3+
                          sets exactly, so the fit error is returned as diagError; overlaps below a cutoff
                          (or not drawable with circles) are left out, with a note under the diagram
  * draw_overlap_matrix   pairwise |A and B| / |A| heatmap (row-normalized fractions)

Imports nothing from the project, so it can be used from any notebook.
"""

import itertools

import numpy as np
from matplotlib.patches import Circle
from scipy import ndimage, optimize

DEFAULT_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]


# ----------------------------------------------------------------------------------------------------------
# region bookkeeping
# ----------------------------------------------------------------------------------------------------------

def all_patterns(n):
    """Every non-empty membership pattern for n sets."""
    return [p for p in itertools.product([False, True], repeat=n) if any(p)]


def region_weights_from_membership(membership, weights):
    """membership: (N, n) bool array, weights: (N,) array -> {pattern: summed weight} for every non-empty pattern."""
    membership = np.asarray(membership, dtype=bool)
    weights = np.asarray(weights, dtype=float)
    n = membership.shape[1]
    codes = membership.astype(np.int64) @ (1 << np.arange(n))
    sums = np.bincount(codes, weights=weights, minlength=1 << n)
    return {p: float(sums[sum(1 << i for i, b in enumerate(p) if b)]) for p in all_patterns(n)}


def set_totals(regions, n):
    return [sum(w for p, w in regions.items() if p[i]) for i in range(n)]


def fmt_weight(w):
    if w >= 100:
        return f"{w:,.0f}"
    if w >= 10:
        return f"{w:.1f}"
    return f"{w:.2f}"


# ----------------------------------------------------------------------------------------------------------
# circle geometry
# ----------------------------------------------------------------------------------------------------------

def lens_area(r1, r2, d):
    """Intersection area of two circles with radii r1, r2 and center distance d."""
    if d >= r1 + r2:
        return 0.0
    if d <= abs(r1 - r2):
        return np.pi * min(r1, r2) ** 2
    a1 = r1 ** 2 * np.arccos(np.clip((d ** 2 + r1 ** 2 - r2 ** 2) / (2 * d * r1), -1, 1))
    a2 = r2 ** 2 * np.arccos(np.clip((d ** 2 + r2 ** 2 - r1 ** 2) / (2 * d * r2), -1, 1))
    a3 = 0.5 * np.sqrt(max((-d + r1 + r2) * (d + r1 - r2) * (d - r1 + r2) * (d + r1 + r2), 0.0))
    return a1 + a2 - a3


def circle_distance_for_overlap(r1, r2, overlap):
    """Center distance of two circles giving the requested intersection area (bisection; area decreases with d)."""
    if overlap <= 0:
        return r1 + r2
    if overlap >= np.pi * min(r1, r2) ** 2:
        return abs(r1 - r2)
    lo, hi = abs(r1 - r2), r1 + r2
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if lens_area(r1, r2, mid) > overlap:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _signed_distance(circles, gx, gy):
    """Signed distance (positive inside) of every grid point to every circle [cx, cy, r]: (n, *grid.shape)."""
    return np.array([r - np.hypot(gx - cx, gy - cy) for cx, cy, r in circles])


def _region_areas(inside, patterns, cell_area):
    """inside: (n, ...) soft or hard membership in [0, 1] -> area of every exclusive region pattern."""
    areas = []
    for p in patterns:
        prod = np.ones(inside.shape[1:])
        for i, b in enumerate(p):
            prod = prod * (inside[i] if b else 1 - inside[i])
        areas.append(prod.sum() * cell_area)
    return np.array(areas)


def _grid(bounds, npts):
    xmin, xmax, ymin, ymax = bounds
    xs = np.linspace(xmin, xmax, npts)
    ys = np.linspace(ymin, ymax, npts)
    gx, gy = np.meshgrid(xs, ys)
    return gx, gy, (xs[1] - xs[0]) * (ys[1] - ys[0])


def _bounds(circles, margin=0.05):
    """Square bounding box around the circles."""
    xmin, xmax = (circles[:, 0] - circles[:, 2]).min(), (circles[:, 0] + circles[:, 2]).max()
    ymin, ymax = (circles[:, 1] - circles[:, 2]).min(), (circles[:, 1] + circles[:, 2]).max()
    cx, cy = 0.5 * (xmin + xmax), 0.5 * (ymin + ymax)
    half = 0.5 * max(xmax - xmin, ymax - ymin) * (1 + margin)
    return (cx - half, cx + half, cy - half, cy + half)


def _hard_region_fracs(circles, patterns, npts=500):
    """Region areas with a hard membership on a fine grid, as fractions of the total area."""
    gx, gy, cell = _grid(_bounds(circles), npts)
    areas = _region_areas((_signed_distance(circles, gx, gy) > 0).astype(float), patterns, cell)
    if areas.sum() <= 0:
        return np.zeros(len(patterns))
    return areas / areas.sum()


# ----------------------------------------------------------------------------------------------------------
# fitting
# ----------------------------------------------------------------------------------------------------------

def _initial_circles(regions, n):
    """Radii from the set totals, centers from the exact pairwise-overlap distances placed with classical MDS."""
    radii = np.sqrt(np.maximum(set_totals(regions, n), 1e-12) / np.pi)
    dist = np.zeros((n, n))
    for i, j in itertools.combinations(range(n), 2):
        overlap = sum(w for p, w in regions.items() if p[i] and p[j])
        d = circle_distance_for_overlap(radii[i], radii[j], overlap)
        if overlap <= 0:
            d *= 1.05  # keep disjoint sets visibly apart
        dist[i, j] = dist[j, i] = d
    j_mat = np.eye(n) - np.ones((n, n)) / n
    evals, evecs = np.linalg.eigh(-0.5 * j_mat @ (dist ** 2) @ j_mat)
    order = np.argsort(evals)[::-1][:2]
    centers = evecs[:, order] * np.sqrt(np.maximum(evals[order], 0))
    return centers, radii


def fit_circles(regions, n, n_restarts=8, grid_points=140, seed=0):
    """Fit n circles so the area of every exclusive region matches its weight (least squares on the region
    areas normalized to unit total, like the R package eulerr, plus the same on the square-root areas so small
    non-empty regions keep visible area).  The area integrals use a soft (tanh)
    membership on a grid so the loss is smooth, annealed towards a hard membership; the best of several
    jittered starts is kept.

    Returns a dict with the (n, 3) circle array [cx, cy, r] in units where area = weight, the patterns, the
    target and fitted region weights, and the eulerr diagError (max |fitted - target| region fraction)."""
    patterns = all_patterns(n)
    target = np.array([max(regions[p], 0.0) for p in patterns])
    total = target.sum()
    if total <= 0:
        raise ValueError("all regions are empty")
    target_frac = target / total

    centers0, radii0 = _initial_circles({p: w / total for p, w in regions.items()}, n)
    to_circles = lambda x: np.column_stack([x.reshape(n, 3)[:, :2], np.exp(x.reshape(n, 3)[:, 2])])
    pack = lambda centers: np.column_stack([centers, np.log(radii0)]).ravel()

    bounds = _bounds(np.column_stack([centers0, radii0]), margin=0.8)
    gx, gy, cell = _grid(bounds, grid_points)
    spacing = (bounds[1] - bounds[0]) / grid_points

    def loss(x, width):
        inside = 0.5 * (1 + np.tanh(_signed_distance(to_circles(x), gx, gy) / width))
        areas = _region_areas(inside, patterns, cell)
        # absolute term (eulerr) plus a square-root term, so small but non-empty regions are not dropped
        return np.sum((areas - target_frac) ** 2) + np.sum((np.sqrt(areas + 1e-6) - np.sqrt(target_frac + 1e-6)) ** 2)

    rng = np.random.default_rng(seed)
    # MDS start plus jittered starts, half small and half large jitter
    starts = [pack(centers0)] + [pack(centers0 + rng.normal(0, 0.15 if k % 2 else 0.6, centers0.shape) * radii0.mean())
                                 for k in range(n_restarts - 1)]
    best = None
    for x in starts:
        for width in [4 * spacing, 1.5 * spacing, 0.6 * spacing]:
            x = optimize.minimize(loss, x, args=(width,), method="L-BFGS-B", options={"maxiter": 300}).x
        hard = _hard_region_fracs(to_circles(x), patterns)
        # keep the start with the lowest hard-membership loss (same terms as the fitted loss)
        hard_loss = np.sum((hard - target_frac) ** 2) + np.sum((np.sqrt(hard) - np.sqrt(target_frac)) ** 2)
        if best is None or hard_loss < best[1]:
            best = (x, hard_loss, hard)

    x, _, hard = best
    diag_error = np.max(np.abs(hard - target_frac))
    x, diag_error, hard = best
    circles = to_circles(x) * np.sqrt(total)
    return {"circles": circles, "patterns": patterns, "target": target, "fitted": hard * total,
            "diag_error": diag_error}


# ----------------------------------------------------------------------------------------------------------
# drawing
# ----------------------------------------------------------------------------------------------------------

def _draw_circles_and_labels(ax, circles, regions, labels, colors, alpha=0.3, npts=500, totals=None):
    n = len(circles)
    for i, (cx, cy, r) in enumerate(circles):
        ax.add_patch(Circle((cx, cy), r, facecolor=colors[i], alpha=alpha, edgecolor="none"))
        ax.add_patch(Circle((cx, cy), r, facecolor="none", edgecolor=colors[i], lw=1.8))

    bounds = _bounds(circles, margin=0.02)
    gx, gy, _ = _grid(bounds, npts)
    inside = _signed_distance(circles, gx, gy) > 0
    spacing = (bounds[1] - bounds[0]) / (npts - 1)

    # region labels at each region's most interior point (largest distance to the region's boundary)
    for p in all_patterns(n):
        w = regions[p]
        if w <= 0:
            continue
        mask = np.ones(gx.shape, dtype=bool)
        for i, b in enumerate(p):
            mask &= inside[i] if b else ~inside[i]
        if not mask.any():
            continue  # region with weight but no area in the fit (fit error); listed in the printed table
        dt = ndimage.distance_transform_edt(mask)
        iy, ix = np.unravel_index(np.argmax(dt), dt.shape)
        roomy = dt[iy, ix] * spacing / (bounds[1] - bounds[0]) > 0.04
        ax.text(gx[iy, ix], gy[iy, ix], fmt_weight(w), ha="center", va="center",
                fontsize=10 if roomy else 7, fontweight="bold" if roomy else "normal")

    # set labels just outside each circle, at the boundary angle farthest from the other circles and from the
    # labels already placed (largest circles first, so the small ones adapt around them)
    totals = set_totals(regions, n) if totals is None else totals
    span = bounds[1] - bounds[0]
    placed = []
    angles = np.radians(np.arange(0, 360, 10))
    for i in np.argsort(-circles[:, 2]):
        cx, cy, r = circles[i]
        best_score, direction = -np.inf, None
        for a in angles:
            d = np.array([np.cos(a), np.sin(a)])
            pos = np.array([cx, cy]) + d * (r + 0.12 * span)  # rough label center
            clearance = min([np.hypot(*(pos - c[:2])) - c[2] for k, c in enumerate(circles) if k != i] + [span])
            label_gap = min([np.hypot(*(pos - q)) for q in placed] + [span])
            score = min(clearance, 0.5 * label_gap) + 0.01 * span * d[1]  # mild preference for above
            if score > best_score:
                best_score, direction = score, d
        placed.append(np.array([cx, cy]) + direction * (r + 0.12 * span))
        pos = np.array([cx, cy]) + direction * r * 1.04
        ax.text(pos[0], pos[1], f"{labels[i]}\n(total {fmt_weight(totals[i])})", color=colors[i],
                ha="center" if abs(direction[0]) < 0.3 else ("left" if direction[0] > 0 else "right"),
                va="center" if abs(direction[1]) < 0.3 else ("bottom" if direction[1] > 0 else "top"),
                fontsize=10, fontweight="bold")

    span = bounds[1] - bounds[0]
    ax.set_xlim(bounds[0] - 0.3 * span, bounds[1] + 0.3 * span)
    ax.set_ylim(bounds[2] - 0.3 * span, bounds[3] + 0.3 * span)
    ax.set_aspect("equal")
    ax.axis("off")


def draw_venn2_exact(ax, regions, labels, colors=None, title=None):
    """Exactly area-proportional two-circle Venn diagram."""
    colors = colors or DEFAULT_COLORS
    tot_a, tot_b = set_totals(regions, 2)
    r1, r2 = np.sqrt(tot_a / np.pi), np.sqrt(tot_b / np.pi)
    d = circle_distance_for_overlap(r1, r2, regions[(True, True)])
    _draw_circles_and_labels(ax, np.array([[0.0, 0.0, r1], [d, 0.0, r2]]), regions, labels, colors)
    if title:
        ax.set_title(title)


def draw_venn_fitted(ax, regions, labels, colors=None, title=None, min_shown=0.0, **fit_kwargs):
    """Fitted area-proportional circle diagram (2-4 sets).

    Overlap regions with weight below min_shown are left out of the diagram (set to zero before fitting).  If
    a larger region still cannot be drawn with circles (no area in the fit), the cutoff is raised to the next
    integer above it and the fit redone, so "overlaps with less than <cutoff> events are not shown" is exactly
    true; that note is written under the diagram whenever something is left out.  Region labels are the true
    weights, and the set labels give the true totals (including hidden overlaps).

    Returns the fit (see fit_circles) with "min_shown" (the cutoff actually used) and "hidden" (the hidden
    region patterns) added."""
    colors = colors or DEFAULT_COLORS
    n = len(labels)
    cutoff = min_shown
    for _ in range(10):
        shown = {p: (w if w >= cutoff else 0.0) for p, w in regions.items()}
        fit = fit_circles(shown, n, **fit_kwargs)
        undrawn = [shown[p] for k, p in enumerate(fit["patterns"]) if shown[p] > 0 and fit["fitted"][k] == 0]
        if not undrawn:
            break
        cutoff = np.floor(max(undrawn)) + 1
    fit["min_shown"] = cutoff
    fit["hidden"] = [p for p, w in regions.items() if 0 < w < cutoff]

    _draw_circles_and_labels(ax, fit["circles"], shown, labels, colors, totals=set_totals(regions, n))
    if fit["hidden"]:
        ax.text(0.5, 0.02, f"Overlaps with less than {cutoff:g} events are not shown", transform=ax.transAxes,
                ha="center", va="bottom", fontsize=11)
    if title:
        ax.set_title(title)
    return fit


def overlap_matrix(membership, weights):
    """M[i, j] = weight(i and j) / weight(i), plus the intersection weights and set totals."""
    membership = np.asarray(membership, dtype=float)
    inter = (membership * np.asarray(weights, dtype=float)[:, None]).T @ membership
    totals = np.diag(inter).copy()
    with np.errstate(invalid="ignore", divide="ignore"):
        frac = inter / totals[:, None]
    return frac, inter, totals


def draw_overlap_matrix(ax, membership, weights, labels, title=None):
    """Heatmap of the fraction of each row selection's weighted events that also pass each column selection."""
    frac, inter, totals = overlap_matrix(membership, weights)
    n = len(labels)
    im = ax.imshow(frac, cmap="viridis", vmin=0, vmax=1)
    for i in range(n):
        for j in range(n):
            if np.isfinite(frac[i, j]) and inter[i, j] > 0:
                ax.text(j, i, f"{100 * frac[i, j]:.0f}%\n{fmt_weight(inter[i, j])}", ha="center", va="center",
                        fontsize=6.5, color="white" if frac[i, j] < 0.6 else "black")
    ax.set_xticks(range(n))
    ax.set_xticklabels(labels, rotation=60, ha="right")
    ax.set_yticks(range(n))
    ax.set_yticklabels([f"{lab} ({fmt_weight(t)})" for lab, t in zip(labels, totals)])
    ax.set_xlabel("also passing this selection")
    ax.set_ylabel("events in this selection (weighted total)")
    if title:
        ax.set_title(title)
    return im
