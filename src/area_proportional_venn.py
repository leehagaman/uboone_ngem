"""Area-proportional Venn diagrams (circles, ellipses, mosaics) and overlap matrices for weighted event selections.

Everything works from a dict of *exclusive* region weights, keyed by a membership tuple of bools (one per
set): e.g. for sets (A, B, C), regions[(True, False, True)] is the summed weight of events in A and C but
not B.  `region_weights_from_membership` builds that dict from per-event boolean membership columns.

  * draw_venn2_exact      two circles, exactly area-proportional (closed-form lens area + bisection)
  * draw_venn_fitted      2-4 circles (or ellipses, shape="ellipse") fitted so each region's area matches its
                          weight (eulerr-style least squares on the region areas); circles cannot represent
                          every configuration of 3+ sets exactly (ellipses can represent more), so the fit
                          error is returned as diagError; overlaps below a cutoff (or not drawable) are left
                          out, with a note under the diagram
  * draw_mosaic           two partitions of the events (e.g. WC Np/0p vs BAGELS Np/0p) as a mosaic: every
                          overlap is a rectangle with exactly proportional area
  * draw_overlap_matrix   pairwise |A and B| / |A| heatmap (row-normalized fractions)

Imports nothing from the project, so it can be used from any notebook.
"""

import itertools

import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.patches import Ellipse, Rectangle
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


def _as_ellipses(shapes):
    """(n, 3) circles [cx, cy, r] or (n, 5) ellipses [cx, cy, a, b, theta] -> (n, 5) ellipses."""
    shapes = np.asarray(shapes, dtype=float)
    if shapes.shape[1] == 5:
        return shapes
    return np.column_stack([shapes[:, :3], shapes[:, 2], np.zeros(len(shapes))])


def _rho(shapes, gx, gy):
    """Normalized elliptical radius of every point for every shape (< 1 inside): (n, *grid.shape)."""
    out = []
    for cx, cy, a, b, t in _as_ellipses(shapes):
        dx, dy = gx - cx, gy - cy
        u = dx * np.cos(t) + dy * np.sin(t)
        v = -dx * np.sin(t) + dy * np.cos(t)
        out.append(np.sqrt((u / a) ** 2 + (v / b) ** 2))
    return np.array(out)


def _signed_distance(shapes, gx, gy):
    """Signed distance (positive inside) of every point to every shape: (n, *grid.shape).  Exact for circles;
    for ellipses (1 - rho) * sqrt(ab), which has the right sign and is close to the distance near the boundary."""
    e = _as_ellipses(shapes)
    scale = np.sqrt(e[:, 2] * e[:, 3]).reshape((-1,) + (1,) * np.ndim(gx))
    return (1 - _rho(e, gx, gy)) * scale


def _shape_areas(shapes):
    e = _as_ellipses(shapes)
    return np.pi * e[:, 2] * e[:, 3]


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


def _bounds(shapes, margin=0.05):
    """Square bounding box around the shapes."""
    cx, cy, a, b, t = _as_ellipses(shapes).T
    wx = np.sqrt((a * np.cos(t)) ** 2 + (b * np.sin(t)) ** 2)
    wy = np.sqrt((a * np.sin(t)) ** 2 + (b * np.cos(t)) ** 2)
    xmin, xmax = (cx - wx).min(), (cx + wx).max()
    ymin, ymax = (cy - wy).min(), (cy + wy).max()
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


def fit_circles(regions, n, n_restarts=8, grid_points=140, seed=0, empty_penalty=10.0):
    """Fit n circles so the area of every exclusive region matches its weight (least squares on the region
    areas normalized to unit total, like the R package eulerr, plus the same on the square-root areas so small
    non-empty regions keep visible area).  The area integrals use a soft (tanh)
    membership on a grid so the loss is smooth, annealed towards a hard membership; the best of several
    jittered starts is kept.  Area in regions that should be empty (e.g. an overlap of two mutually exclusive
    selections) costs empty_penalty times more, so when circles can't match everything the error goes into the
    sizes of real regions rather than into drawing impossible overlaps.

    Returns a dict with the (n, 3) circle array [cx, cy, r] in units where area = weight, the patterns, the
    target and fitted region weights, and the eulerr diagError (max |fitted - target| region fraction)."""
    patterns = all_patterns(n)
    target = np.array([max(regions[p], 0.0) for p in patterns])
    total = target.sum()
    if total <= 0:
        raise ValueError("all regions are empty")
    target_frac = target / total
    empty = target_frac == 0

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
        return (np.sum((areas - target_frac) ** 2) + np.sum((np.sqrt(areas + 1e-6) - np.sqrt(target_frac + 1e-6)) ** 2)
                + empty_penalty * np.sum(areas[empty]))

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
        hard_loss = (np.sum((hard - target_frac) ** 2) + np.sum((np.sqrt(hard) - np.sqrt(target_frac)) ** 2)
                     + empty_penalty * np.sum(hard[empty]))
        if best is None or hard_loss < best[1]:
            best = (x, hard_loss, hard)

    x, _, hard = best
    diag_error = np.max(np.abs(hard - target_frac))
    circles = to_circles(x) * np.sqrt(total)
    return {"circles": circles, "patterns": patterns, "target": target, "fitted": hard * total,
            "diag_error": diag_error}


def fit_ellipses(regions, n, grid_points=140, empty_penalty=10.0, **circle_kwargs):
    """Like fit_circles, but each set is an ellipse [cx, cy, a, b, theta] (two more degrees of freedom per set,
    so more configurations can be drawn exactly, as in eulerr).  Starts from the circle fit.  Returns the same
    dict, with "circles" holding the (n, 5) ellipses."""
    start = fit_circles(regions, n, grid_points=grid_points, empty_penalty=empty_penalty, **circle_kwargs)
    patterns, target = start["patterns"], start["target"]
    total = target.sum()
    target_frac = target / total
    empty = target_frac == 0

    c0 = start["circles"] / np.sqrt(total)
    to_shapes = lambda x: np.column_stack([x.reshape(n, 5)[:, :2], np.exp(x.reshape(n, 5)[:, 2:4]),
                                           x.reshape(n, 5)[:, 4]])
    bounds = _bounds(c0, margin=0.8)
    gx, gy, cell = _grid(bounds, grid_points)
    spacing = (bounds[1] - bounds[0]) / grid_points

    def loss(x, width):
        inside = 0.5 * (1 + np.tanh(_signed_distance(to_shapes(x), gx, gy) / width))
        areas = _region_areas(inside, patterns, cell)
        return (np.sum((areas - target_frac) ** 2) + np.sum((np.sqrt(areas + 1e-6) - np.sqrt(target_frac + 1e-6)) ** 2)
                + empty_penalty * np.sum(areas[empty]))

    def hard_loss(x):
        hard = _hard_region_fracs(to_shapes(x), patterns)
        return (np.sum((hard - target_frac) ** 2) + np.sum((np.sqrt(hard) - np.sqrt(target_frac)) ** 2)
                + empty_penalty * np.sum(hard[empty])), hard

    # circle start, plus the same with every set slightly elongated along a few orientations (a circle is a
    # saddle point for the orientation, so the optimizer needs a nudge)
    base = np.column_stack([c0[:, :2], np.log(c0[:, 2]), np.log(c0[:, 2]), np.zeros(n)])
    starts = [base.ravel()]
    for k, t in enumerate(np.linspace(0, np.pi, 4, endpoint=False)):
        x = base.copy()
        x[:, 2] += 0.15
        x[:, 3] -= 0.15
        x[:, 4] = t + np.arange(n) * np.pi / n * (k % 2)
        starts.append(x.ravel())
    best = None
    for x in starts:
        for width in [4 * spacing, 1.5 * spacing, 0.6 * spacing]:
            x = optimize.minimize(loss, x, args=(width,), method="L-BFGS-B", options={"maxiter": 400}).x
        hl, hard = hard_loss(x)
        if best is None or hl < best[1]:
            best = (x, hl, hard)
    x, _, hard = best
    shapes = to_shapes(x)
    shapes[:, :4] *= np.sqrt(total)
    return {"circles": shapes, "patterns": patterns, "target": target, "fitted": hard * total,
            "diag_error": np.max(np.abs(hard - target_frac))}


# ----------------------------------------------------------------------------------------------------------
# drawing
# ----------------------------------------------------------------------------------------------------------

def _darken(color, amount):
    """Mix a color toward black (amount 0 = unchanged, 1 = black), e.g. for legible label text."""
    return tuple((1 - amount) * c for c in to_rgb(color))


def _draw_circles_and_labels(ax, circles, regions, labels, colors, alpha=0.3, npts=500, totals=None,
                             linestyles=None):
    """alpha and linestyles may be a single value or one per circle."""
    circles = _as_ellipses(circles)
    n = len(circles)
    alphas = np.broadcast_to(np.asarray(alpha, dtype=float), (n,))
    linestyles = [linestyles or "solid"] * n if linestyles is None or isinstance(linestyles, str) else linestyles
    for i, (cx, cy, a, b, t) in enumerate(circles):
        ax.add_patch(Ellipse((cx, cy), 2 * a, 2 * b, angle=np.degrees(t), facecolor=colors[i], alpha=alphas[i],
                             edgecolor="none"))
        ax.add_patch(Ellipse((cx, cy), 2 * a, 2 * b, angle=np.degrees(t), facecolor="none", edgecolor=colors[i],
                             lw=2.0, linestyle=linestyles[i]))

    bounds = _bounds(circles, margin=0.02)
    span = bounds[1] - bounds[0]
    pad = 0.3 * span
    ax.set_xlim(bounds[0] - pad, bounds[1] + pad)
    ax.set_ylim(bounds[2] - pad, bounds[3] + pad)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.apply_aspect()
    limits = np.array([bounds[0] - pad, bounds[1] + pad, bounds[2] - pad, bounds[3] + pad])
    renderer = ax.figure.canvas.get_renderer()
    to_data = ax.transData.inverted()

    def text_size(text):
        """(width, height) of a drawn text in data units."""
        bb = text.get_window_extent(renderer).transformed(to_data)
        return bb.width, bb.height

    boxes = []  # (xmin, xmax, ymin, ymax) of every placed label, so later labels avoid them

    def box_gap(box):
        """Separation from the nearest placed label box (negative if overlapping)."""
        if not boxes:
            return span
        b = np.array(boxes)
        dx = np.maximum(b[:, 0] - box[1], box[0] - b[:, 1])
        dy = np.maximum(b[:, 2] - box[3], box[2] - b[:, 3])
        return np.min(np.where((dx > 0) & (dy > 0), np.hypot(dx, dy), np.maximum(dx, dy)))

    # labeling grid covers the whole axes, so numbers that have no room inside their region can go outside
    gx, gy, _ = _grid(limits, npts)
    spacing = (limits[1] - limits[0]) / (npts - 1)
    inside = _signed_distance(circles, gx, gy) > 0
    outside_all = ~inside.any(axis=0)
    outside_depth = ndimage.distance_transform_edt(outside_all) * spacing

    def free(w, h, allowed):
        """Grid points in allowed where a w x h box centered there overlaps no placed label."""
        ok = allowed.copy()
        for x0, x1, y0, y1 in boxes:
            ok &= (np.abs(gx - 0.5 * (x0 + x1)) >= 0.5 * (w + x1 - x0) + 0.005 * span) | \
                  (np.abs(gy - 0.5 * (y0 + y1)) >= 0.5 * (h + y1 - y0) + 0.005 * span)
        return ok

    # region numbers at the most interior uncrowded point of each region, largest regions first; a region
    # with no room left gets its number just outside the circles with a leader line
    for p in sorted(all_patterns(n), key=lambda p: -regions[p]):
        w = regions[p]
        if w <= 0:
            continue
        mask = np.ones(gx.shape, dtype=bool)
        for i, b in enumerate(p):
            mask &= inside[i] if b else ~inside[i]
        if not mask.any():
            continue  # region with weight but no area in the fit (fit error); listed in the printed table
        dt = ndimage.distance_transform_edt(mask) * spacing
        iy, ix = np.unravel_index(np.argmax(dt), dt.shape)
        anchor = np.array([gx[iy, ix], gy[iy, ix]])
        roomy = dt[iy, ix] / span > 0.04
        t = ax.text(*anchor, fmt_weight(w), ha="center", va="center",
                    fontsize=10 if roomy else 7, fontweight="bold" if roomy else "normal")
        tw, th = text_size(t)
        ok = free(tw, th, mask & (dt >= 0.5 * th))
        if ok.any():
            iy, ix = np.unravel_index(np.argmax(np.where(ok, dt, -1)), dt.shape)
            pos = np.array([gx[iy, ix], gy[iy, ix]])
        else:
            ok = free(tw, th, outside_all & (outside_depth >= 0.5 * np.hypot(tw, th) + 0.01 * span))
            if ok.any():
                dist = np.where(ok, np.hypot(gx - anchor[0], gy - anchor[1]), np.inf)
                iy, ix = np.unravel_index(np.argmin(dist), dist.shape)
                pos = np.array([gx[iy, ix], gy[iy, ix]])
                ax.annotate("", xy=anchor, xytext=pos, arrowprops=dict(arrowstyle="-", lw=0.7, color="0.3",
                                                                    shrinkA=4, shrinkB=0))
            else:
                pos = anchor
        t.set_position(pos)
        boxes.append((pos[0] - tw / 2, pos[0] + tw / 2, pos[1] - th / 2, pos[1] + th / 2))

    # set labels outside each circle, at the angle and distance with the most clearance from the circles and
    # the labels already placed (largest circles first); a label pushed away from its circle gets a leader line
    totals = set_totals(regions, n) if totals is None else totals
    angles = np.radians(np.arange(0, 360, 10))
    for i in np.argsort(-_shape_areas(circles)):
        cx, cy = circles[i, :2]
        t = ax.text(cx, cy, f"{labels[i]}\n(total {fmt_weight(totals[i])})", color=_darken(colors[i], 0.25),
                    fontsize=10, fontweight="bold")
        tw, th = text_size(t)
        best = None
        for a in angles:
            d = np.array([np.cos(a), np.sin(a)])
            r = 1 / _rho(circles[i:i + 1], cx + d[0], cy + d[1])[0]  # center-to-boundary distance along d
            ha = "center" if abs(d[0]) < 0.3 else ("left" if d[0] > 0 else "right")
            va = "center" if abs(d[1]) < 0.3 else ("bottom" if d[1] > 0 else "top")
            for ext in [0.02, 0.08, 0.15, 0.25]:
                pos = np.array([cx, cy]) + d * (r + ext * span)
                x0 = pos[0] - {"left": 0, "center": tw / 2, "right": tw}[ha]
                y0 = pos[1] - {"bottom": 0, "center": th / 2, "top": th}[va]
                box = (x0, x0 + tw, y0, y0 + th)
                if box[0] < limits[0] or box[1] > limits[1] or box[2] < limits[2] or box[3] > limits[3]:
                    continue
                # distance from the shapes to a grid of points over the box (negative: box reaches inside)
                bx, by = np.meshgrid(np.linspace(box[0], box[1], 7), np.linspace(box[2], box[3], 3))
                clearance = -np.max(_signed_distance(circles, bx, by))
                # enough clearance (capped) beats more of it, so labels stay next to their circle when they can
                score = min(clearance, box_gap(box), 0.02 * span) - 0.2 * ext * span + 0.002 * span * d[1]
                if best is None or score > best[0]:
                    best = (score, pos, ha, va, box, d, ext)
        _, pos, ha, va, box, d, ext = best
        t.set_position(pos)
        t.set_ha(ha)
        t.set_va(va)
        boxes.append(box)
        if ext > 0.02:
            r = 1 / _rho(circles[i:i + 1], cx + d[0], cy + d[1])[0]
            ax.annotate("", xy=np.array([cx, cy]) + d * r, xytext=pos,
                        arrowprops=dict(arrowstyle="-", lw=0.8, color=_darken(colors[i], 0.25), shrinkA=2, shrinkB=0))


def draw_venn2_exact(ax, regions, labels, colors=None, title=None):
    """Exactly area-proportional two-circle Venn diagram."""
    colors = colors or DEFAULT_COLORS
    tot_a, tot_b = set_totals(regions, 2)
    r1, r2 = np.sqrt(tot_a / np.pi), np.sqrt(tot_b / np.pi)
    d = circle_distance_for_overlap(r1, r2, regions[(True, True)])
    _draw_circles_and_labels(ax, np.array([[0.0, 0.0, r1], [d, 0.0, r2]]), regions, labels, colors)
    if title:
        ax.set_title(title)


def draw_venn_fitted(ax, regions, labels, colors=None, title=None, min_shown=0.0, alpha=0.3, linestyles=None,
                     shape="circle", **fit_kwargs):
    """Fitted area-proportional circle diagram (2-4 sets).

    Overlap regions with weight below min_shown are left out of the diagram (set to zero before fitting).  If
    a larger region still cannot be drawn with circles (no area in the fit), the cutoff is raised to the next
    integer above it and the fit redone, so "overlaps with less than <cutoff> events are not shown" is exactly
    true; that note is written under the diagram whenever something is left out.  Region labels are the true
    weights, and the set labels give the true totals (including hidden overlaps).

    Returns the fit (see fit_circles) with "min_shown" (the cutoff actually used) and "hidden" (the hidden
    region patterns) added.  alpha (fill) and linestyles (edge) may be one value or one per set."""
    colors = colors or DEFAULT_COLORS
    n = len(labels)
    cutoff = min_shown
    for _ in range(10):
        shown = {p: (w if w >= cutoff else 0.0) for p, w in regions.items()}
        fit = (fit_ellipses if shape == "ellipse" else fit_circles)(shown, n, **fit_kwargs)
        undrawn = [shown[p] for k, p in enumerate(fit["patterns"]) if shown[p] > 0 and fit["fitted"][k] == 0]
        if not undrawn:
            break
        cutoff = np.floor(max(undrawn)) + 1
    fit["min_shown"] = cutoff
    fit["hidden"] = [p for p, w in regions.items() if 0 < w < cutoff]

    _draw_circles_and_labels(ax, fit["circles"], shown, labels, colors, totals=set_totals(regions, n),
                             alpha=alpha, linestyles=linestyles)
    if fit["hidden"]:
        ax.text(0.5, 0.02, f"Overlaps with less than {cutoff:g} events are not shown", transform=ax.transAxes,
                ha="center", va="bottom", fontsize=11)
    if title:
        ax.set_title(title)
    return fit


def draw_mosaic(ax, col_masks, row_masks, weights, col_labels, row_labels, col_colors, row_colors,
                col_none_label="neither", row_none_label="neither", min_label_frac=0.004):
    """Exactly area-proportional mosaic for two partitions (e.g. WC Np / 0p vs BAGELS Np / 0p).

    col_masks / row_masks: lists of mutually exclusive boolean event masks.  Only events in the union of all
    the selections are drawn.  Columns are the col_masks groups plus "neither" (in no column selection), with
    widths proportional to their weight; each column is split vertically into the row groups plus "neither",
    so every cell's area is its exact weight.  Cells are filled with the row color (grey for "neither"); the
    column headers are drawn in the column colors."""
    col_masks = [np.asarray(m, dtype=bool) for m in col_masks]
    row_masks = [np.asarray(m, dtype=bool) for m in row_masks]
    weights = np.asarray(weights, dtype=float)
    for group in (col_masks, row_masks):
        if np.any(np.sum(group, axis=0) > 1):
            raise ValueError("mosaic groups must be mutually exclusive")
    cols = col_masks + [~np.any(col_masks, axis=0)]
    rows = row_masks + [~np.any(row_masks, axis=0)]
    w = np.array([[weights[c & r].sum() for r in rows] for c in cols])
    w[-1, -1] = 0.0  # in no selection at all: not part of the diagram
    total = w.sum()
    col_tot = w.sum(axis=1)
    keep = [j for j in range(len(cols)) if col_tot[j] > 0]
    row_colors = list(row_colors) + ["0.85"]
    row_labels = list(row_labels) + [row_none_label]
    col_labels = list(col_labels) + [col_none_label]
    col_colors = list(col_colors) + ["0.6"]

    gap = 0.012
    width_avail = 1 - gap * (len(keep) - 1)
    x = 0.0
    renderer = ax.figure.canvas.get_renderer()
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.25)
    ax.axis("off")
    to_data = ax.transData.inverted()
    header_levels = []  # right edge of the last header placed on each level
    for j in keep:
        cw = width_avail * col_tot[j] / total
        ax.add_patch(Rectangle((x, 1.01), cw, 0.025, facecolor=col_colors[j], edgecolor="none"))
        # header centered on its column, raised to the first level where it doesn't hit an earlier header
        t = ax.text(x + cw / 2, 1.045, f"{col_labels[j]}\n(total {fmt_weight(col_tot[j])})", ha="center",
                    va="bottom", fontsize=10, fontweight="bold", color=_darken(col_colors[j], 0.25))
        bb = t.get_window_extent(renderer).transformed(to_data)
        tx = max(x + cw / 2, -0.02 + bb.width / 2)
        level = next((k for k, right in enumerate(header_levels) if tx - bb.width / 2 > right + 0.01),
                     len(header_levels))
        if level == len(header_levels):
            header_levels.append(-np.inf)
        header_levels[level] = tx + bb.width / 2
        t.set_position((tx, 1.045 + level * 1.05 * bb.height))
        if level > 0 or tx != x + cw / 2:
            ax.plot([x + cw / 2, tx], [1.037, 1.045 + level * 1.05 * bb.height], color="0.5", lw=0.7)
        heights = w[j] / col_tot[j]
        n_rows_shown = np.count_nonzero(heights)
        height_avail = 1 - gap * max(n_rows_shown - 1, 0)
        y = 1.0
        for i in range(len(rows)):
            if w[j, i] <= 0:
                continue
            h = height_avail * heights[i]
            y -= h
            ax.add_patch(Rectangle((x, y), cw, h, facecolor=row_colors[i], alpha=0.55 if i < len(rows) - 1 else 1,
                                   edgecolor=_darken(row_colors[i], 0.2) if i < len(rows) - 1 else "0.6", lw=1.2))
            if w[j, i] / total >= min_label_frac:
                t = ax.text(x + cw / 2, y + h / 2, fmt_weight(w[j, i]), ha="center", va="center", fontsize=10,
                            fontweight="bold")
                bb = t.get_window_extent(renderer).transformed(ax.transData.inverted())
                if bb.height > h or bb.width > cw:  # does not fit: smaller, or rotated in a narrow column
                    t.set_fontsize(7)
                    t.set_fontweight("normal")
                    if bb.width > cw and h > cw:
                        t.set_rotation(90)
            y -= gap
        x += cw + gap
    handles = [Rectangle((0, 0), 1, 1, facecolor=c, alpha=0.55 if k < len(row_colors) - 1 else 1,
                         edgecolor=_darken(c, 0.2) if k < len(row_colors) - 1 else "0.6")
               for k, c in enumerate(row_colors)]
    ax.legend(handles, row_labels, loc="upper center", bbox_to_anchor=(0.5, -0.01), ncol=len(row_labels),
              frameon=False, fontsize=10)
    return w


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
