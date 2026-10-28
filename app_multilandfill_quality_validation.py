import os
import tempfile

import matplotlib.pyplot as plt
import netCDF4
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Landfill Satellite Data Framework", page_icon="🛰️", layout="wide")
st.title("🛰️ Landfill Satellite Data Framework")



def find_var(group, names):
    for name in names:
        if name in group.variables:
            return group.variables[name]
    for child in group.groups.values():
        found = find_var(child, names)
        if found is not None:
            return found
    return None


def read_arr(var):
    var.set_auto_maskandscale(True)
    arr = np.ma.asarray(var[:], dtype=float).filled(np.nan)
    arr[~np.isfinite(arr)] = np.nan
    arr[np.abs(arr) > 1e20] = np.nan
    return np.squeeze(arr)


def haversine_km(lat1, lon1, lat2, lon2):
    earth_radius_km = 6371.0088
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlon / 2) ** 2
    return 2 * earth_radius_km * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def get_timestamps(time_var, n_scanlines, ds):
    """Return one timestamp per scanline, using time_utc or metadata fallback."""
    if time_var is not None:
        try:
            raw = np.asarray(time_var[:]).squeeze()
            if raw.ndim == 0:
                raw = raw.reshape(1)
            if raw.ndim == 1 and len(raw) == n_scanlines:
                out = []
                for value in raw:
                    if isinstance(value, bytes):
                        value = value.decode("utf-8", errors="replace")
                    out.append(str(value).strip())
                return out
        except Exception:
            pass
    fallback = str(getattr(ds, "time_coverage_start", ""))
    return [fallback] * n_scanlines


def process_file(uploaded_file, site_name, target_lat, target_lon, radius_km, qa_min):
    """Extract spatially relevant, quality-screened observations from one L2 file for one site."""
    temp_path = None
    try:
        fd, temp_path = tempfile.mkstemp(suffix=".nc")
        os.close(fd)
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        with netCDF4.Dataset(temp_path, "r") as ds:
            product = ds.groups.get("PRODUCT")
            if product is None:
                return [], {
                    "landfill_name": site_name, "file": uploaded_file.name,
                    "status": "Skipped: no PRODUCT group", "pixels_in_radius": 0,
                    "pixels_with_methane": 0, "qa_pass": 0, "usable": 0
                }

            ch4_var = find_var(product, ["methane_mixing_ratio", "methane_mixing_ratio_bias_corrected"])
            lat_var = find_var(product, ["latitude"])
            lon_var = find_var(product, ["longitude"])
            qa_var = find_var(product, ["qa_value"])
            time_var = find_var(product, ["time_utc"])

            if any(v is None for v in [ch4_var, lat_var, lon_var]):
                return [], {
                    "landfill_name": site_name, "file": uploaded_file.name,
                    "status": "Skipped: required variables missing", "pixels_in_radius": 0,
                    "pixels_with_methane": 0, "qa_pass": 0, "usable": 0
                }

            ch4, lat, lon = read_arr(ch4_var), read_arr(lat_var), read_arr(lon_var)
            qa = read_arr(qa_var) if qa_var is not None else None

            if ch4.ndim != 2 or lat.shape != ch4.shape or lon.shape != ch4.shape:
                return [], {
                    "landfill_name": site_name, "file": uploaded_file.name,
                    "status": f"Skipped: unsupported shapes CH4={ch4.shape}, lat={lat.shape}, lon={lon.shape}",
                    "pixels_in_radius": 0, "pixels_with_methane": 0, "qa_pass": 0, "usable": 0
                }
            if qa is not None and qa.shape != ch4.shape:
                qa = None

            timestamps = get_timestamps(time_var, ch4.shape[0], ds)
            geo_ok = np.isfinite(lat) & np.isfinite(lon) & (np.abs(lat) <= 90) & (np.abs(lon) <= 180)
            distance = np.full(ch4.shape, np.nan, dtype=float)
            distance[geo_ok] = haversine_km(target_lat, target_lon, lat[geo_ok], lon[geo_ok])
            in_radius = geo_ok & (distance <= radius_km)
            data_ok = np.isfinite(ch4)

            if qa is not None:
                quality_ok = np.isfinite(qa) & (qa >= qa_min)
                qa_status = "Applied"
            else:
                quality_ok = np.ones(ch4.shape, dtype=bool)
                qa_status = "Unavailable; not applied"

            usable = in_radius & data_ok & quality_ok
            scan_idx, pixel_idx = np.where(usable)
            rows = []
            for i, j in zip(scan_idx, pixel_idx):
                rows.append({
                    "landfill_name": site_name,
                    "target_latitude": float(target_lat),
                    "target_longitude": float(target_lon),
                    "source_file": uploaded_file.name,
                    "observation_time_utc": timestamps[i],
                    "pixel_latitude": float(lat[i, j]),
                    "pixel_longitude": float(lon[i, j]),
                    "distance_from_landfill_km": float(distance[i, j]),
                    "methane_mixing_ratio_ppb": float(ch4[i, j]),
                    "qa_value": float(qa[i, j]) if qa is not None and np.isfinite(qa[i, j]) else np.nan,
                    "scanline_index": int(i),
                    "ground_pixel_index": int(j),
                })

            summary = {
                "landfill_name": site_name,
                "file": uploaded_file.name,
                "status": "Processed",
                "time_start": str(getattr(ds, "time_coverage_start", "")),
                "time_end": str(getattr(ds, "time_coverage_end", "")),
                "pixels_in_radius": int(in_radius.sum()),
                "pixels_with_methane": int((in_radius & data_ok).sum()),
                "qa_pass": int((in_radius & data_ok & quality_ok).sum()),
                "usable": int(usable.sum()),
                "qa_filter": qa_status,
                "variable": ch4_var.name,
                "units_attribute": str(getattr(ch4_var, "units", "not specified")),
            }
            return rows, summary
    except Exception as e:
        return [], {
            "landfill_name": site_name, "file": uploaded_file.name,
            "status": f"Error: {e}", "pixels_in_radius": 0,
            "pixels_with_methane": 0, "qa_pass": 0, "usable": 0
        }
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass


def load_sites_from_csv(uploaded_csv):
    """Validate the cleaned landfill CSV and return valid sites plus a coordinate review table."""
    sites_df = pd.read_csv(uploaded_csv)
    required = {"landfill_name", "latitude_decimal", "longitude_decimal"}
    if not required.issubset(sites_df.columns):
        missing = ", ".join(sorted(required - set(sites_df.columns)))
        raise ValueError(f"CSV is missing required column(s): {missing}")

    sites_df["latitude_decimal"] = pd.to_numeric(sites_df["latitude_decimal"], errors="coerce")
    sites_df["longitude_decimal"] = pd.to_numeric(sites_df["longitude_decimal"], errors="coerce")
    coord_ok = (
        sites_df["latitude_decimal"].notna()
        & sites_df["longitude_decimal"].notna()
        & sites_df["latitude_decimal"].between(-90, 90)
        & sites_df["longitude_decimal"].between(-180, 180)
    )
    sites_df["processing_status"] = np.where(coord_ok, "Coordinates available; verify location", "Skipped: missing/invalid coordinates")
    valid = sites_df.loc[coord_ok].copy()
    sites = [
        (str(row["landfill_name"]), float(row["latitude_decimal"]), float(row["longitude_decimal"]))
        for _, row in valid.iterrows()
    ]
    return sites, sites_df


def classify_job_quality(row):
    """Create a simple, transparent diagnostic label from per-file/site counts."""
    status = str(row.get("status", ""))
    if not status.startswith("Processed"):
        return "Processing issue"
    near = int(row.get("pixels_in_radius", 0) or 0)
    with_ch4 = int(row.get("pixels_with_methane", 0) or 0)
    qa_pass = int(row.get("qa_pass", 0) or 0)
    usable = int(row.get("usable", 0) or 0)
    if near == 0:
        return "No pixels in radius"
    if with_ch4 == 0:
        return "Nearby pixels have no valid methane values"
    if qa_pass == 0:
        return "No pixels passed QA"
    if usable == 0:
        return "No usable observations"
    if usable < with_ch4:
        return "Usable observations; some excluded by QA/missing data"
    return "Usable observations"


with st.sidebar:
    st.header("Landfill locations")
    mode = st.radio("Choose processing mode", ["Single landfill", "Upload landfill CSV"], index=0)
    if mode == "Single landfill":
        site_name = st.text_input("Landfill name", "Pirana Landfill, Ahmedabad")
        target_lat = st.number_input("Latitude", value=22.98, min_value=-90.0, max_value=90.0, format="%.6f")
        target_lon = st.number_input("Longitude", value=72.56, min_value=-180.0, max_value=180.0, format="%.6f")
        sites = [(site_name, float(target_lat), float(target_lon))]
        site_review_df = None
    else:
        site_csv = st.file_uploader("Upload landfill locations CSV", type=["csv"])
        sites = []
        site_review_df = None
        if site_csv is not None:
            try:
                sites, site_review_df = load_sites_from_csv(site_csv)
                st.caption(f"{len(sites)} site(s) have usable numeric coordinates.")
                if site_review_df is not None:
                    st.caption("Coordinates still need independent verification before scientific interpretation.")
            except Exception as exc:
                st.error(f"Could not read landfill CSV: {exc}")

    radius_km = st.slider("Landfill search radius (km)", min_value=5, max_value=100, value=25, step=5)
    qa_min = st.slider(
        "Minimum QA value", min_value=0.0, max_value=1.0, value=0.5, step=0.1,
        help="Only observations at or above this threshold are retained when QA is available."
    )
    st.subheader("Background comparison (exploratory)")
    background_outer_km = st.slider("Background outer radius (km)", min_value=30, max_value=150, value=50, step=5)
    background_enabled = st.checkbox(
        "Include background ring", value=True,
        help="Uses pixels between the landfill search radius and the outer radius. Exploratory only; not validated plume attribution."
    )

st.header("1. Upload TROPOMI data")
uploaded_files = st.file_uploader(
    "Select one or more TROPOMI L2 CH₄ NetCDF files",
    type=["nc"],
    accept_multiple_files=True,
)

if not uploaded_files:
    st.stop()

if not sites:
    if mode == "Upload landfill CSV":
        st.warning("Upload a valid landfill CSV in the sidebar. It must include landfill_name, latitude_decimal, and longitude_decimal columns.")
    else:
        st.warning("Enter a valid landfill location.")
    st.stop()

if site_review_df is not None:
    st.header("2. Landfill coordinate review")
    st.caption("Rows with missing or invalid coordinates are not processed. All coordinates should be verified against reliable site locations.")
    st.dataframe(site_review_df, width="stretch", hide_index=True)
    st.download_button(
        "Download coordinate review (CSV)",
        site_review_df.to_csv(index=False).encode("utf-8"),
        file_name="landfill_coordinate_review.csv",
        mime="text/csv"
    )

all_rows = []
file_summaries = []
total_jobs = len(uploaded_files) * len(sites)
progress = st.progress(0, text="Preparing landfill/file combinations...")
job = 0
for site_name, target_lat, target_lon in sites:
    for uploaded_file in uploaded_files:
        progress.progress(job / max(1, total_jobs), text=f"Processing {site_name} — {uploaded_file.name}...")
        rows, summary = process_file(uploaded_file, site_name, target_lat, target_lon, radius_km, qa_min)
        all_rows.extend(rows)
        file_summaries.append(summary)
        job += 1
progress.progress(1.0, text="Processing complete")
progress.empty()

results = pd.DataFrame(all_rows)
summary_df = pd.DataFrame(file_summaries)

st.header("3. File coverage and quality report")
if not summary_df.empty:
    # Derived diagnostic ratios are app-calculated; the pixel counts originate from reading each NetCDF swath.
    for numerator, denominator, col_name in [
        ("pixels_with_methane", "pixels_in_radius", "valid_methane_fraction_pct"),
        ("qa_pass", "pixels_with_methane", "qa_pass_fraction_pct"),
        ("usable", "pixels_in_radius", "usable_fraction_of_nearby_pct"),
    ]:
        denom = pd.to_numeric(summary_df[denominator], errors="coerce").replace(0, np.nan)
        numer = pd.to_numeric(summary_df[numerator], errors="coerce")
        summary_df[col_name] = (100 * numer / denom).round(1)
    summary_df["quality_diagnostic"] = summary_df.apply(classify_job_quality, axis=1)

    st.dataframe(summary_df, width="stretch", hide_index=True)
    total_usable = int(summary_df.get("usable", pd.Series(dtype=int)).fillna(0).sum())
    total_jobs_processed = int((summary_df["status"] == "Processed").sum()) if "status" in summary_df else 0
    total_jobs = len(summary_df)
    no_coverage_jobs = int((summary_df["quality_diagnostic"] == "No pixels in radius").sum())
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Landfill sites selected", len(sites))
    m2.metric("Files uploaded", len(uploaded_files))
    m3.metric("Processed site/file pairs", f"{total_jobs_processed}/{total_jobs}")
    m4.metric("Usable observations", f"{total_usable:,}")
    st.caption(f"Site/file pairs with no pixels inside the selected radius: {no_coverage_jobs}. This indicates no matching pixels in that swath/radius, not absence of methane.")
    st.download_button(
        "Download file-processing report (CSV)",
        summary_df.to_csv(index=False).encode("utf-8"),
        file_name="tropomi_file_processing_report.csv",
        mime="text/csv"
    )

st.header("4. Quality validation checks")
st.caption("These are diagnostic checks on coverage and filtering. They do not independently validate the satellite retrieval or prove landfill-source methane.")
if not summary_df.empty:
    site_qc = summary_df.groupby("landfill_name", dropna=False).agg(
        site_file_pairs=("file", "size"),
        processed_pairs=("status", lambda x: int((x == "Processed").sum())),
        total_pixels_in_radius=("pixels_in_radius", "sum"),
        pixels_with_methane=("pixels_with_methane", "sum"),
        qa_pass_pixels=("qa_pass", "sum"),
        usable_observations=("usable", "sum"),
        no_coverage_pairs=("quality_diagnostic", lambda x: int((x == "No pixels in radius").sum())),
    ).reset_index()
    site_qc["valid_methane_pct"] = (100 * site_qc["pixels_with_methane"] / site_qc["total_pixels_in_radius"].replace(0, np.nan)).round(1)
    site_qc["qa_pass_pct_of_valid_methane"] = (100 * site_qc["qa_pass_pixels"] / site_qc["pixels_with_methane"].replace(0, np.nan)).round(1)
    site_qc["usable_pct_of_nearby_pixels"] = (100 * site_qc["usable_observations"] / site_qc["total_pixels_in_radius"].replace(0, np.nan)).round(1)
    st.subheader("Validation summary by landfill")
    st.dataframe(site_qc, width="stretch", hide_index=True)
    st.download_button(
        "Download quality validation summary (CSV)",
        site_qc.to_csv(index=False).encode("utf-8"),
        file_name="landfill_quality_validation_summary.csv",
        mime="text/csv"
    )
    with st.expander("How to interpret these checks"):
        st.markdown("""
        - **No pixels in radius:** the file's swath did not provide pixel centres within your chosen radius. It is not evidence of zero methane.
        - **Valid methane fraction:** percentage of nearby pixels with a finite methane value.
        - **QA pass fraction:** percentage of nearby pixels with valid methane that met your selected QA threshold.
        - **Usable fraction:** percentage of nearby pixels retained after methane-validity and QA filtering.
        - **NaN percentage:** a blank percentage usually means the denominator was zero, so the fraction cannot be calculated.
        - **Coordinate check:** numeric coordinates can be in range and still point to the wrong location; compare each site against a trusted map or official source.
        """)

if results.empty:
    st.warning(
        "No usable observations were found for the selected site(s) and file(s). Check coverage, radius, "
        "missing methane values and QA filtering. This does not mean methane is absent."
    )
    st.stop()

results["observation_time_utc_parsed"] = pd.to_datetime(
    results["observation_time_utc"], errors="coerce", utc=True
)
results = results.sort_values(
    ["landfill_name", "observation_time_utc_parsed", "distance_from_landfill_km"],
    na_position="last"
).reset_index(drop=True)

st.subheader("Additional observation-level checks")
valid_times = results["observation_time_utc_parsed"].dropna()
if not valid_times.empty:
    st.write(f"Parsed observation timestamps: {len(valid_times):,}/{len(results):,} ({100 * len(valid_times) / len(results):.1f}%).")
    st.write(f"Date range in retained observations (UTC): {valid_times.min().date()} to {valid_times.max().date()}.")
else:
    st.warning("No observation timestamps could be parsed; temporal coverage cannot be assessed.")
exact_key = ["landfill_name", "source_file", "scanline_index", "ground_pixel_index"]
exact_duplicate_rows = int(results.duplicated(subset=exact_key, keep="first").sum())
st.write(f"Exact duplicate pixel records within the same landfill/file: {exact_duplicate_rows:,}.")
if exact_duplicate_rows > 0:
    st.warning("Exact duplicate records were found. Review the combined CSV before calculating summaries.")
st.caption("This exact-duplicate check does not detect all possible overlapping satellite acquisitions from different files.")

st.header("5. Combined landfill-nearby observations")
st.caption("One row per usable pixel observation across all selected sites and files. Overlapping/duplicate acquisitions are not automatically deduplicated.")
st.dataframe(results.drop(columns=["observation_time_utc_parsed"]), width="stretch", hide_index=True)
st.download_button(
    "Download combined observations (CSV)",
    results.drop(columns=["observation_time_utc_parsed"]).to_csv(index=False).encode("utf-8"),
    file_name="combined_landfill_tropomi_ch4_observations.csv",
    mime="text/csv"
)

st.subheader("Observation summary by landfill")
site_summary = results.groupby("landfill_name", as_index=False).agg(
    usable_observations=("methane_mixing_ratio_ppb", "size"),
    median_xch4_ppb=("methane_mixing_ratio_ppb", "median"),
    mean_xch4_ppb=("methane_mixing_ratio_ppb", "mean"),
    median_qa=("qa_value", "median"),
    closest_pixel_km=("distance_from_landfill_km", "min"),
)
st.dataframe(site_summary, width="stretch", hide_index=True)
st.download_button(
    "Download landfill summary (CSV)",
    site_summary.to_csv(index=False).encode("utf-8"),
    file_name="landfill_summary.csv",
    mime="text/csv"
)

st.header("6. Geographic visualization")
fig, ax = plt.subplots(figsize=(9, 5))
sc = ax.scatter(
    results["pixel_longitude"], results["pixel_latitude"],
    c=results["methane_mixing_ratio_ppb"], s=28, cmap="viridis", alpha=0.8
)
targets = pd.DataFrame(sites, columns=["landfill_name", "target_latitude", "target_longitude"])
ax.scatter(
    targets["target_longitude"], targets["target_latitude"],
    marker="x", s=100, linewidths=2, color="red", label="Landfill target"
)
ax.set_xlabel("Longitude (degrees)")
ax.set_ylabel("Latitude (degrees)")
ax.set_title("Quality-screened TROPOMI XCH₄ observations across selected sites")
ax.legend()
fig.colorbar(sc, ax=ax, label="Methane mixing ratio (product units/scaling to verify)")
ax.grid(alpha=0.25)
st.pyplot(fig)
plt.close(fig)

st.header("7. Temporal analysis")
time_series = results.dropna(subset=["observation_time_utc_parsed"]).copy()
if time_series.empty:
    st.warning("Observation timestamps could not be parsed, so a time-series plot is unavailable.")
else:
    time_series["observation_date_utc"] = time_series["observation_time_utc_parsed"].dt.floor("D")
    daily = time_series.groupby(["landfill_name", "observation_date_utc"], as_index=False).agg(
        mean_xch4_ppb=("methane_mixing_ratio_ppb", "mean"),
        median_xch4_ppb=("methane_mixing_ratio_ppb", "median"),
        observation_count=("methane_mixing_ratio_ppb", "size"),
    )
    st.caption("Daily summaries are calculated separately for each landfill. Unequal pixel counts and repeated acquisitions can affect comparisons.")
    st.dataframe(daily, width="stretch", hide_index=True)
    selected_site_for_plot = st.selectbox(
        "Choose a landfill for its daily time series",
        options=sorted(daily["landfill_name"].unique())
    )
    plot_daily = daily[daily["landfill_name"] == selected_site_for_plot]
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    ax2.plot(plot_daily["observation_date_utc"], plot_daily["median_xch4_ppb"], marker="o")
    ax2.set_xlabel("Observation date (UTC)")
    ax2.set_ylabel("Median XCH₄ (product units/scaling to verify)")
    ax2.set_title(f"Daily median methane mixing ratio — {selected_site_for_plot}")
    ax2.grid(alpha=0.3)
    fig2.autofmt_xdate()
    st.pyplot(fig2)
    plt.close(fig2)
    st.download_button(
        "Download daily time-series summary (CSV)",
        daily.to_csv(index=False).encode("utf-8"),
        file_name="landfill_daily_xch4_summary.csv",
        mime="text/csv"
    )

if background_enabled:
    st.header("8. Exploratory background comparison")
    if background_outer_km <= radius_km:
        st.warning("Choose a background outer radius greater than the landfill search radius.")
    else:
        bg_rows = []
        bg_progress = st.progress(0, text="Extracting background-ring observations...")
        bg_jobs = len(uploaded_files) * len(sites)
        bg_job = 0
        for site_name, target_lat, target_lon in sites:
            for uploaded_file in uploaded_files:
                bg_progress.progress(bg_job / max(1, bg_jobs), text=f"Checking background for {site_name} — {uploaded_file.name}...")
                rows, _ = process_file(uploaded_file, site_name, target_lat, target_lon, background_outer_km, qa_min)
                for row in rows:
                    if radius_km < row["distance_from_landfill_km"] <= background_outer_km:
                        row["region"] = "background_ring"
                        bg_rows.append(row)
                bg_job += 1
        bg_progress.progress(1.0, text="Background extraction complete")
        bg_progress.empty()
        bg_df = pd.DataFrame(bg_rows)
        if bg_df.empty:
            st.warning("No valid background-ring observations were found for the selected files/sites.")
        else:
            landfill_region = results[results["distance_from_landfill_km"] <= radius_km]
            site_medians = landfill_region.groupby("landfill_name")["methane_mixing_ratio_ppb"].median()
            bg_medians = bg_df.groupby("landfill_name")["methane_mixing_ratio_ppb"].median()
            compare = pd.DataFrame({
                "landfill_name": sorted(set(site_medians.index) | set(bg_medians.index))
            })
            compare["landfill_radius_median_ppb"] = compare["landfill_name"].map(site_medians)
            compare["background_ring_median_ppb"] = compare["landfill_name"].map(bg_medians)
            compare["difference_site_minus_background_ppb"] = (
                compare["landfill_radius_median_ppb"] - compare["background_ring_median_ppb"]
            )
            compare["landfill_observation_count"] = compare["landfill_name"].map(
                landfill_region.groupby("landfill_name").size()
            )
            compare["background_observation_count"] = compare["landfill_name"].map(
                bg_df.groupby("landfill_name").size()
            )
            st.write("This is an exploratory contrast between nearby columns, not proof of landfill-source methane.")
            st.dataframe(compare, width="stretch", hide_index=True)
            st.download_button(
                "Download background comparison (CSV)",
                compare.to_csv(index=False).encode("utf-8"),
                file_name="landfill_background_comparison.csv",
                mime="text/csv"
            )
            st.download_button(
                "Download background-ring observations (CSV)",
                bg_df.to_csv(index=False).encode("utf-8"),
                file_name="landfill_background_ring_observations.csv",
                mime="text/csv"
            )

with st.expander("Metadata and scientific notes"):
    st.write("The methane variable is typically column-averaged dry-air methane mixing ratio (XCH₄). Verify exact units and scale interpretation from product metadata before reporting results.")
    st.write("The app filters pixel centres by distance; it does not test whether pixel footprint polygons intersect the landfill or radius boundary.")
    st.write("A single high XCH₄ observation does not prove landfill attribution. Reliable plume/emission estimates need validated background methods and additional meteorological/retrieval analysis.")
    st.write("CSV coordinates are used as supplied. Verify each landfill coordinate against a reliable location source; approximate coordinates/range midpoints are for exploration only.")
    st.write("TROPOMI L1B and EMIT L1B require separate geolocation, quality, spectral and retrieval pipelines; they are not supported by this L2 extractor.")
