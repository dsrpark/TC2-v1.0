"""
TC Detection Pipeline — orchestrates reader + Fortran core + post-processing.

Usage:
    from detect import TCDetector
    from config import MODELS

    detector = TCDetector(MODELS["ERA5"])
    tracks = detector.run(years=[2020])
    detector.save_tracks(tracks, "output/DBTRACK_ERA5_test.txt")
"""
import sys
import gc
import time
import numpy as np
import netCDF4 as nc4
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Optional
from dataclasses import dataclass

from .config_loader import (
    ModelConfig, CRI_VOR, CRI_LAT, CRI_GEN_LAT,
    MINDIST_KM, MINDHR,
    MAXD_DEG, PSL_SEARCH_KM, WSPD_SEARCH_KM, M_SEARCH_DEG,
    DT_HOURS, OCEANID, MAX_CENTERS, MAX_TRACKS, MAX_TRACKLEN,
)
from .reader import SmartReader

# Import Fortran module (required)
sys.path.insert(0, str(Path(__file__).parent / "fortran"))
from detect_core import calc_vorticity, find_centers, connect_tracks



def _precompute_near_ocean(mask, nlon, nlat, maxd, oceanid):
    """Precompute near-ocean mask: 1 if any ocean grid within 4*maxd, 0 otherwise."""
    r = 4 * maxd
    # is_ocean(nlon, nlat): True where mask == oceanid
    is_ocean = (mask == oceanid)
    # Collapse latitude: any ocean in lat window for each (lon, lat)
    # Use cumulative sum for fast window query along lat axis
    ocean_cumsum = np.zeros((nlon, nlat + 1), dtype=np.int32)
    ocean_cumsum[:, 1:] = np.cumsum(is_ocean.astype(np.int32), axis=1)
    # has_ocean_in_lat_window(i, j) = sum of is_ocean[i, j2lo:j2hi+1] > 0
    has_ocean_lat = np.zeros((nlon, nlat), dtype=bool)
    for j in range(nlat):
        j2lo = max(j - r, 0)
        j2hi = min(j + r, nlat - 1)
        has_ocean_lat[:, j] = (ocean_cumsum[:, j2hi + 1] - ocean_cumsum[:, j2lo]) > 0
    # Now expand along longitude with periodic wrapping
    # Pad has_ocean_lat periodically for rolling window
    padded = np.concatenate([has_ocean_lat[-r:, :], has_ocean_lat, has_ocean_lat[:r, :]], axis=0)
    # For each (i, j): any True in padded[i:i+2r+1, j]
    # Use cumulative sum along lon axis
    pad_cumsum = np.zeros((padded.shape[0] + 1, nlat), dtype=np.int32)
    pad_cumsum[1:, :] = np.cumsum(padded.astype(np.int32), axis=0)
    near_ocean = np.zeros((nlon, nlat), dtype=np.int32)
    for i in range(nlon):
        lo = i  # offset by r due to padding
        hi = i + 2 * r + 1
        near_ocean[i, :] = (pad_cumsum[hi, :] - pad_cumsum[lo, :]) > 0
    return near_ocean





# ============================================================
# Track data structure
# ============================================================
@dataclass
class Track:
    """One detected disturbance track."""
    track_id: int
    lon: np.ndarray
    lat: np.ndarray
    psl: np.ndarray       # Pa
    wspd: np.ndarray      # m/s
    time_idx: np.ndarray  # global timestep index
    year: np.ndarray
    month: np.ndarray
    day: np.ndarray
    hour: np.ndarray


def load_tracks(filepath: str) -> List[Track]:
    """Load Track objects from DBTRACK txt file."""
    tracks = []
    with open(filepath) as f:
        lines = f.readlines()
    i = 0
    while i < len(lines):
        parts = lines[i].split()
        if len(parts) == 3 and len(parts[0]) == 4:
            nstep = int(parts[1])
            tid = int(parts[2])
            lons, lats, psls, wspds = [], [], [], []
            years, months, days, hours = [], [], [], []
            for j in range(1, nstep + 1):
                if i + j >= len(lines):
                    break
                p = lines[i + j].split()
                years.append(int(p[0]))
                months.append(int(p[1]))
                days.append(int(p[2]))
                hours.append(int(p[3]))
                lons.append(float(p[4]))
                lats.append(float(p[5]))
                wspds.append(float(p[6]))
                psls.append(float(p[7]) * 100)  # hPa → Pa
            if lons:
                trk = Track(
                    track_id=tid,
                    lon=np.array(lons, dtype=np.float32),
                    lat=np.array(lats, dtype=np.float32),
                    psl=np.array(psls, dtype=np.float32),
                    wspd=np.array(wspds, dtype=np.float32),
                    time_idx=np.zeros(len(lons), dtype=np.int32),
                    year=np.array(years, dtype=np.int32),
                    month=np.array(months, dtype=np.int32),
                    day=np.array(days, dtype=np.int32),
                    hour=np.array(hours, dtype=np.int32),
                )
                tracks.append(trk)
            i += nstep + 1
        else:
            i += 1
    print(f"Loaded {len(tracks)} tracks ({sum(len(t.lon) for t in tracks)} points) from {filepath}")
    return tracks


# ============================================================
# Calendar utilities
# ============================================================

def _find_time_dim(ds):
    for name in ['time', 'Time', 'TIME']:
        if name in ds.variables:
            return name
    return 'time'




# ============================================================
# Best-track loading and matching
# ============================================================


# ============================================================
# Main detector class
# ============================================================

class TCDetector:
    """
    End-to-end TC detection pipeline.

    Usage:
        detector = TCDetector(model_config)
        tracks = detector.run(years=[2020])
        detector.save_tracks(tracks, "output/result.txt")
    """

    def __init__(self, cfg: ModelConfig):
        self.cfg = cfg
        self.reader = SmartReader(cfg)

    def run(self, years: Optional[List[int]] = None,
            debug: bool = False) -> List[Track]:
        """
        Run full detection pipeline.

        Args:
            years: List of years to process. Default: cfg.styr to cfg.edyr.
        """
        if years is None:
            years = list(range(self.cfg.styr, self.cfg.edyr + 1))

        print(f"=== TC Detection: {self.cfg.name} ({years[0]}-{years[-1]}) ===")

        # Read coordinates and mask
        lon, lat = self.reader.read_lonlat(years[0])
        mask = self.reader.read_mask()
        nlon, nlat = len(lon), len(lat)

        # Auto-detect maxd (2.5 degrees in grid points)
        dx_deg = abs(lon[1] - lon[0]) if len(lon) > 1 else 1.0
        maxd = max(int(MAXD_DEG / dx_deg), 1)
        print(f"  Grid: {nlon}x{nlat}, resolution ~{dx_deg:.4f} deg, maxd={maxd}")

        # Precompute near-ocean mask (time-invariant)
        near_ocean = _precompute_near_ocean(mask, nlon, nlat, maxd, OCEANID)

        # Process each year: detect centers
        all_clon, all_clat, all_cpsl, all_cwspd = [], [], [], []
        all_ncenter = []

        for year in years:
            print(f"\n--- Year {year} ---")
            t0 = time.perf_counter()
            ua, va = self.reader.read_winds_850(year)
            psl_data = self.reader.read_psl(year)
            ntime = ua.shape[2]  # actual timesteps from data
            t_read = time.perf_counter() - t0

            # Compute vorticity
            t0 = time.perf_counter()
            print("  Computing vorticity...")
            vor = calc_vorticity(ua, va, lon, lat, nlon, nlat, ntime)

            # Wind speed
            wspd = np.sqrt(ua**2 + va**2)
            t_vor = time.perf_counter() - t0

            # Count vorticity extrema (diagnostic, slow)
            if debug:
                from scipy.ndimage import maximum_filter, minimum_filter
                lat_mask = np.abs(lat) <= CRI_LAT
                n_vorex = 0
                for t in range(ntime):
                    v = vor[:, :, t]
                    above_thr = np.abs(v) >= CRI_VOR
                    sz = 2 * maxd + 1
                    local_max = maximum_filter(v, size=(sz, sz), mode='wrap')
                    local_min = minimum_filter(v, size=(sz, sz), mode='wrap')
                    is_ext = np.zeros_like(v, dtype=bool)
                    is_ext[:, lat >= 0] = (v[:, lat >= 0] == local_max[:, lat >= 0])
                    is_ext[:, lat < 0] = (v[:, lat < 0] == local_min[:, lat < 0])
                    n_vorex += int((above_thr & is_ext & lat_mask[None, :]).sum())
                print(f"  Vorticity extrema: {n_vorex} (over {ntime} timesteps)")

            # Find centers
            max_centers = MAX_CENTERS
            m_search = int(np.ceil(M_SEARCH_DEG / dx_deg))
            print(f"  Finding disturbance centers (m_search={m_search})...")
            t0 = time.perf_counter()
            c_lon, c_lat, c_psl, c_wspd, c_vlon, c_vlat, nctr = find_centers(
                vor, psl_data, wspd, lon, lat,
                CRI_VOR, CRI_LAT, maxd,
                m_search, PSL_SEARCH_KM, WSPD_SEARCH_KM,
                max_centers, near_ocean)
            t_centers = time.perf_counter() - t0

            n_cap_hit = int((nctr >= max_centers).sum())
            print(f"  Centers: max={nctr.max()}/step, total={nctr.sum()}")
            print(f"  [Time] read={t_read:.1f}s  vorticity={t_vor:.1f}s  centers={t_centers:.1f}s")
            if n_cap_hit > 0:
                print(f"  WARNING: max_centers={max_centers} cap hit in {n_cap_hit}/{ntime} timesteps!")

            all_clon.append(c_lon)
            all_clat.append(c_lat)
            all_cpsl.append(c_psl)
            all_cwspd.append(c_wspd)
            all_ncenter.append(nctr)

            del ua, va, psl_data, vor, wspd, c_vlon, c_vlat
            gc.collect()
            rss_gb = int(open('/proc/self/status').read().split('VmRSS:')[1].split()[0]) / 1e6
            print(f"  [Memory] RSS = {rss_gb:.1f} GB")

        # Concatenate all years
        all_clon = np.concatenate(all_clon, axis=0)
        all_clat = np.concatenate(all_clat, axis=0)
        all_cpsl = np.concatenate(all_cpsl, axis=0)
        all_cwspd = np.concatenate(all_cwspd, axis=0)
        all_ncenter = np.concatenate(all_ncenter, axis=0)

        total_ntime = all_clon.shape[0]
        max_centers = all_clon.shape[1]

        # Connect tracks
        print(f"\nConnecting tracks across {total_ntime} timesteps...")
        t0 = time.perf_counter()
        t_lon, t_lat, t_psl, t_wspd, t_tidx, t_len, ntrack = connect_tracks(
            all_clon, all_clat, all_cpsl, all_cwspd, all_ncenter,
            MINDIST_KM, MINDHR, DT_HOURS,
            mask, lon, lat, OCEANID, CRI_GEN_LAT,
            MAX_TRACKS, MAX_TRACKLEN)
        t_connect = time.perf_counter() - t0

        print(f"  Total tracks found: {ntrack}")
        print(f"  [Time] connect={t_connect:.1f}s")
        if ntrack >= MAX_TRACKS:
            raise RuntimeError(
                f"connect_tracks hit max_tracks={MAX_TRACKS}. "
                f"Results are incomplete. Increase max_tracks in config."
            )

        # Build global time axis from actual netCDF time coordinates
        all_datetimes = []
        for year in years:
            if self.cfg.file_layout == "cmip6":
                # ESGF: collect timesteps from all files covering this year, de-dup,
                # then SORT chronologically to match the data array. _read_3d_esgf
                # reorders the data by (month,day,hour,minute); the scenario start-year
                # boundary step (Jan-1 00Z) lives at the END of the preceding
                # historical file, so plain file-scan order would place it last here
                # while the data has it first — misassigning every timestamp in that
                # year (a +6h shift with the 00Z teleported to year-end). Sorting fixes it.
                files = self.reader._esgf_find_all("psl", year)
                seen = set()
                year_dts = []
                for path in files:
                    ds = nc4.Dataset(str(path))
                    time_var = ds.variables[_find_time_dim(ds)]
                    tunits = getattr(time_var, 'units', '')
                    tcal = getattr(time_var, 'calendar', 'standard')
                    dts = nc4.num2date(time_var[:], tunits, tcal)
                    for dt in dts:
                        if dt.year == year:
                            # Match _read_3d_esgf's de-dup/sort key exactly (incl. minute).
                            key = (dt.month, dt.day, dt.hour, getattr(dt, "minute", 0))
                            if key not in seen:
                                seen.add(key)
                                year_dts.append(dt)
                    ds.close()
                year_dts.sort(key=lambda d: (d.month, d.day, d.hour, getattr(d, "minute", 0)))
                all_datetimes.extend(year_dts)
            elif self.cfg.file_layout == "split_monthly":
                for month in range(1, 13):
                    path = self.reader._nc_path("psl", year, month=month)
                    if not path.exists():
                        continue
                    ds = nc4.Dataset(str(path))
                    time_var = ds.variables[_find_time_dim(ds)]
                    tunits = getattr(time_var, 'units', '')
                    if "since" in tunits:
                        tcal = getattr(time_var, 'calendar', 'standard')
                        dts = nc4.num2date(time_var[:], tunits, tcal)
                    else:
                        # Non-CF: parse YYYYMMDD.fraction
                        import datetime
                        dts = []
                        for val in time_var[:]:
                            fval = float(val)
                            if 10000000 < fval < 99999999:
                                idate = int(fval)
                                frac = fval - idate
                                dts.append(datetime.datetime(
                                    idate // 10000,
                                    (idate % 10000) // 100,
                                    idate % 100,
                                    round(frac * 24)))
                            elif dts:
                                dts.append(dts[-1] + datetime.timedelta(hours=6))
                            else:
                                dts.append(datetime.datetime(year, month, 1, 0))
                    all_datetimes.extend(dts)
                    ds.close()
            else:
                path = self.reader._nc_path("psl", year)
                ds = nc4.Dataset(str(path))
                time_var = ds.variables[_find_time_dim(ds)]
                tunits = getattr(time_var, 'units', '')
                tcal = getattr(time_var, 'calendar', 'standard')
                dts = nc4.num2date(time_var[:], tunits, tcal)
                all_datetimes.extend(dts)
                ds.close()

        # Guard: the datetime axis must align 1:1 with the data/center axis. A
        # mismatch means the ESGF time collection diverged from _read_3d_esgf's
        # ordering/dedup (the 2015-boundary class of bug) — fail fast, don't mislabel.
        if len(all_datetimes) != total_ntime:
            raise RuntimeError(
                f"time-axis mismatch for {self.cfg.name}: {len(all_datetimes)} datetimes "
                f"vs {total_ntime} data timesteps — check ESGF time collection in detect.py")

        tracks = []
        for i in range(ntrack):
            tlen = t_len[i]
            tr_tidx = t_tidx[i, :tlen]
            yr = np.zeros(tlen, dtype=np.int32)
            mo = np.zeros(tlen, dtype=np.int32)
            dy = np.zeros(tlen, dtype=np.int32)
            hr = np.zeros(tlen, dtype=np.int32)
            for j in range(tlen):
                dt = all_datetimes[int(tr_tidx[j]) - 1]
                yr[j] = dt.year; mo[j] = dt.month; dy[j] = dt.day; hr[j] = dt.hour

            tracks.append(Track(
                track_id=i + 1,
                lon=t_lon[i, :tlen], lat=t_lat[i, :tlen],
                psl=t_psl[i, :tlen], wspd=t_wspd[i, :tlen],
                time_idx=tr_tidx, year=yr, month=mo, day=dy, hour=hr,
            ))

        print(f"\n=== Done: {len(tracks)} tracks detected ===")
        return tracks

    def save_tracks(self, tracks: List[Track], outpath: str) -> None:
        """Save tracks in original Fortran output format for comparison."""
        Path(outpath).parent.mkdir(parents=True, exist_ok=True)
        with open(outpath, "w") as f:
            for trk in tracks:
                nstep = len(trk.lon)
                f.write(f"{trk.year[0]:04d}    {nstep:6d}    {trk.track_id:6d}\n")
                for i in range(nstep):
                    f.write(f"{trk.year[i]:04d} {trk.month[i]:02d} {trk.day[i]:02d} "
                            f"{trk.hour[i]:02d}  {trk.lon[i]:8.2f} {trk.lat[i]:8.2f} "
                            f"{trk.wspd[i]:8.2f} {trk.psl[i] / 100:8.2f}   0\n")
        print(f"Saved {len(tracks)} tracks to {outpath}")

    def save_tracks_csv(self, tracks: List[Track], outpath: str) -> None:
        """Save tracks as CSV (easy for pandas/02model)."""
        rows = []
        for trk in tracks:
            for i in range(len(trk.lon)):
                rows.append({
                    "track_id": trk.track_id,
                    "year": trk.year[i], "month": trk.month[i],
                    "day": trk.day[i], "hour": trk.hour[i],
                    "lon": trk.lon[i], "lat": trk.lat[i],
                    "wspd": trk.wspd[i], "psl_hPa": trk.psl[i] / 100,
                })
        df = pd.DataFrame(rows)
        df.to_csv(outpath, index=False)
        print(f"Saved {len(tracks)} tracks ({len(df)} points) to {outpath}")
