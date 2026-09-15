"""
Data-derived [Mn/Fe] and [Mn/Mg] vs [Fe/Mg] reference lines, binned in [Mg/H].

This is the canonical (single-copy) version of a computation that used to be
copy-pasted into many notebooks under examples/metallicity_dependance/: read
astroNN_2proc_crossmatch.fits, take the ASPCAP-adjusted abundances, bin stars
by [Mg/H], and fit a line to [Mn/Fe] vs [Fe/Mg] (and separately [Mn/Mg] vs
[Fe/Mg]) within each bin. Those per-bin (slope, intercept) pairs are what
notebooks overlay as a reference line on model output.

Typical usage
-------------
    from chemevo.mn_fe_lines import load_mn_fe_lines, mn_fe_model_line

    lines_df = load_mn_fe_lines()  # reads the saved CSV; regenerate with
                                    # `python -m chemevo.mn_fe_lines` if the
                                    # source FITS file or binning changes
    y = mn_fe_model_line(lines_df, mg_h_bin=-0.2, fe_mg=my_fe_mg_array)
"""
import os

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.table import Table
from scipy.stats import linregress

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_FITS_PATH = os.path.join(REPO_ROOT, "examples", "data", "astroNN_2proc_crossmatch.fits")
DEFAULT_OUTPUT_PATH = os.path.join(REPO_ROOT, "examples", "data", "mn_fe_lines.csv")

DEFAULT_MG_H_BIN_EDGES = np.arange(-0.7, 0.50, 0.20)
DEFAULT_GALZ_RANGE = (-2, 2)
DEFAULT_GALR_RANGE = (3, 15)


def _decode_bytes(df):
    """FITS string columns come back as byte strings (b'...'); decode them to normal strings."""
    byte_columns = df.select_dtypes([object])
    if byte_columns.empty:
        return df

    stacked = byte_columns.stack()          # one long series of every byte-string value
    decoded = stacked.str.decode("utf-8")   # decode each one to a normal Python string
    decoded = decoded.unstack()             # reshape back into the original columns

    for col in decoded:
        df[col] = decoded[col]

    return df


def _fitsrec_to_pandas(fits_rec):
    """Convert a FITS_rec (from astropy.io.fits) into a pandas DataFrame."""
    table = Table(fits_rec)

    # pandas can't hold multi-dimensional columns, so drop those.
    column_names = []
    for name in table.colnames:
        if len(table[name].shape) <= 1:
            column_names.append(name)

    df = table[column_names].to_pandas()
    return _decode_bytes(df)


def load_apogee_data(fits_path=DEFAULT_FITS_PATH,
                      galz_range=DEFAULT_GALZ_RANGE, galr_range=DEFAULT_GALR_RANGE):
    """
    Read astroNN_2proc_crossmatch.fits and return a DataFrame of ASPCAP-adjusted
    abundance ratios (MG_H, FE_H, MN_H, MN_FE, FE_MG, MN_MG, MG_FE, ...),
    quality-cut to DFLAG == 0 and a solar-neighborhood spatial selection.
    """
    with fits.open(fits_path) as hdul:
        astronn_data = hdul[1].data
        twoprocess_data = hdul[2].data

    two_process_df = _fitsrec_to_pandas(twoprocess_data)
    astronn_df = _fitsrec_to_pandas(astronn_data)

    # twoprocess_df has the ASPCAP_ADJ abundance columns we want; astronn_df
    # has the galr/galz spatial columns we need. Combine the two.
    data = pd.concat(
        [
            two_process_df.reset_index(drop=True),
            astronn_df[["galr", "galz"]].reset_index(drop=True),
        ],
        axis=1,
    )

    data = data[data["DFLAG"] == 0]

    data["MG_H"] = data["MG_H_ASPCAP_ADJ"]
    data["FE_H"] = data["FE_H_ASPCAP_ADJ"]
    data["MN_H"] = data["MN_H_ASPCAP_ADJ"]
    data["MN_FE"] = data["MN_H_ASPCAP_ADJ"] - data["FE_H_ASPCAP_ADJ"]
    data["MN_MG"] = data["MN_H_ASPCAP_ADJ"] - data["MG_H_ASPCAP_ADJ"]
    data["FE_MG"] = data["FE_H_ASPCAP_ADJ"] - data["MG_H_ASPCAP_ADJ"]
    data["MG_FE"] = data["MG_H_ASPCAP_ADJ"] - data["FE_H_ASPCAP_ADJ"]

    galz_lo, galz_hi = galz_range
    galr_lo, galr_hi = galr_range
    in_galz_range = (data["galz"] > galz_lo) & (data["galz"] < galz_hi)
    in_galr_range = (data["galr"] > galr_lo) & (data["galr"] < galr_hi)
    data = data[in_galz_range & in_galr_range]

    return data.reset_index(drop=True)


def _fit_line(x, y, min_points):
    """
    Fit a straight line y = slope*x + intercept, using only the points
    where both x and y are finite (not NaN/inf).

    Returns (slope, intercept), or (nan, nan) if there are fewer than
    `min_points` finite points to fit.
    """
    both_finite = np.isfinite(x) & np.isfinite(y)
    n_finite = both_finite.sum()

    if n_finite < min_points:
        return np.nan, np.nan

    fit = linregress(x[both_finite], y[both_finite])
    return fit.slope, fit.intercept


def compute_mn_fe_lines(data=None, mg_h_bin_edges=DEFAULT_MG_H_BIN_EDGES, min_points_per_bin=3, **load_kwargs):
    """
    Bin `data` by [Mg/H] and fit [Mn/Fe] vs [Fe/Mg] and [Mn/Mg] vs [Fe/Mg]
    (independently, via scipy.stats.linregress) within each bin.

    Parameters
    ----------
    data : pandas.DataFrame, optional
        Must have MG_H, FE_MG, MN_FE, MN_MG columns. If not given, calls
        `load_apogee_data(**load_kwargs)`.
    mg_h_bin_edges : np.ndarray
        Bin edges in [Mg/H]. Default: 0.2 dex bins from -0.7 to 0.5.
    min_points_per_bin : int
        A bin with fewer finite (x, y) pairs than this gets NaN slope/intercept.

    Returns
    -------
    pandas.DataFrame with columns: mg_h_bin_center, slope_mn_fe,
    intercept_mn_fe, slope_mn_mg, intercept_mn_mg.
    """
    if data is None:
        data = load_apogee_data(**load_kwargs)

    bin_centers = 0.5 * (mg_h_bin_edges[:-1] + mg_h_bin_edges[1:])
    n_bins = len(bin_centers)

    # Which bin (by index) each star's [Mg/H] falls into.
    mg_h_bin_index = pd.cut(data["MG_H"], bins=mg_h_bin_edges, right=False, labels=False)

    rows = []
    for i in range(n_bins):
        subset = data[mg_h_bin_index == i]
        fe_mg = subset["FE_MG"]

        slope_mn_fe, intercept_mn_fe = _fit_line(fe_mg, subset["MN_FE"], min_points_per_bin)
        slope_mn_mg, intercept_mn_mg = _fit_line(fe_mg, subset["MN_MG"], min_points_per_bin)

        rows.append({
            "mg_h_bin_center": round(float(bin_centers[i]), 10),
            "slope_mn_fe": slope_mn_fe,
            "intercept_mn_fe": intercept_mn_fe,
            "slope_mn_mg": slope_mn_mg,
            "intercept_mn_mg": intercept_mn_mg,
        })

    return pd.DataFrame(rows)


def save_mn_fe_lines(lines_df, path=DEFAULT_OUTPUT_PATH):
    lines_df.to_csv(path, index=False)


def load_mn_fe_lines(path=DEFAULT_OUTPUT_PATH):
    return pd.read_csv(path)


def mn_fe_model_line(lines_df, mg_h_bin, fe_mg, kind="mn_fe"):
    """
    Evaluate the fitted reference line at arbitrary [Fe/Mg] value(s).

    Parameters
    ----------
    lines_df : pandas.DataFrame
        As returned by `load_mn_fe_lines()` / `compute_mn_fe_lines()`.
    mg_h_bin : float
        The [Mg/H] bin center to use (matched via np.isclose).
    fe_mg : float or array-like
        [Fe/Mg] value(s) to evaluate the line at.
    kind : {"mn_fe", "mn_mg"}
        Which fitted line to evaluate.

    Returns
    -------
    Same shape as `fe_mg`.
    """
    is_this_bin = np.isclose(lines_df["mg_h_bin_center"], mg_h_bin)
    matching_rows = lines_df[is_this_bin]
    if matching_rows.empty:
        raise ValueError(f"No line fit found for mg_h_bin={mg_h_bin}")

    row = matching_rows.iloc[0]
    slope = row[f"slope_{kind}"]
    intercept = row[f"intercept_{kind}"]
    return slope * np.asarray(fe_mg) + intercept


def main():
    lines_df = compute_mn_fe_lines()
    save_mn_fe_lines(lines_df)
    print(f"Wrote {len(lines_df)} [Mg/H]-binned line fits to {DEFAULT_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
