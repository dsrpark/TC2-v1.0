"""
Environmental Parameter Calculation

Computes environmental indices around detected TC centers:
- 850 hPa wind speed (max, azimuthal mean)
- Vertical wind shear (250-850 hPa)
- 850 hPa vorticity (max, mean, inner/outer)
- Warm core (250 hPa, 250-500 hPa mean)
- Sea level pressure (min, azimuthal mean)

Corresponds to CAL_PARAMETER.f90 in the original code.
"""
import sys
import os
import shutil
import hashlib
import fcntl
import errno
import numpy as np
from collections import defaultdict
from pathlib import Path
from typing import List

from .config_loader import (
    ModelConfig, DT_HOURS, ENV_REGION_DEG, ENV_WSPD_RADIUS_KM,
    ENV_VOR_INNER_KM, ENV_VOR_OUTER_KM, ENV_VOR_OUTER_START_KM,
    ENV_PSL_RADIUS_KM, ENV_WARMCORE_RADIUS_KM,
)
from .reader import SmartReader
from .detect import Track

sys.path.insert(0, str(Path(__file__).parent / "fortran"))
from detect_core import calc_env_all


# ── Env-stage integrity helpers ────────────────────────────────────────────────
# Guard against reusing a stale per-year tmp file from a *different* DBTRACK run.
# The old skip-check only matched model/year/point-count, so a re-detected DBTRACK
# with the same yearly point count but different track structure silently reused
# stale tmp -> corrupt envi (e.g. ACCESS-CM2 ssp585 2017/2053, 2026-06).

def _point_key(trk, p_idx):
    """Canonical per-point key: track_id:YYYYMMDDHH:lon:lat (2dp, matching written precision)."""
    return (f"{trk.track_id}:{int(trk.year[p_idx]):04d}{int(trk.month[p_idx]):02d}"
            f"{int(trk.day[p_idx]):02d}{int(trk.hour[p_idx]):02d}:"
            f"{trk.lon[p_idx]:.2f}:{trk.lat[p_idx]:.2f}")


def _fingerprint(keys):
    """Order-independent sha256 (first 12 hex) over a list of point keys."""
    h = hashlib.sha256()
    for k in sorted(keys):
        h.update(k.encode())
        h.update(b"\n")
    return h.hexdigest()[:12]


def _year_keys(tracks, year_points):
    """Keys for one year. year_points entries: (t_idx, p_idx, ix, iy)."""
    return [_point_key(tracks[t_idx], p_idx) for (t_idx, p_idx, _ix, _iy) in year_points]


def _global_keys(tracks):
    """Keys for every point of every track (run-level fingerprint input)."""
    return [_point_key(trk, p) for trk in tracks for p in range(len(trk.lon))]


def _haversine_km_vectorized(lon1, lat1, lon2_grid, lat2_grid):
    """Vectorized haversine distance in km. float32 to match Fortran real(4)."""
    pi = np.float32(3.141592)
    deg2rad = pi / np.float32(180.)
    lat1_r = np.float32(lat1) * deg2rad
    lat2_r = lat2_grid.astype(np.float32) * deg2rad
    dy = (lat2_grid.astype(np.float32) - np.float32(lat1)) * deg2rad
    dx = np.abs(lon2_grid.astype(np.float32) - np.float32(lon1))
    dx = np.where(dx >= np.float32(180.), np.float32(360.) - dx, dx)
    dx = dx * deg2rad
    a = np.sin(dy * np.float32(0.5))**2 + \
        np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dx * np.float32(0.5))**2
    return np.float32(6371.) * np.float32(2.) * np.arctan2(
        np.sqrt(a), np.sqrt(np.float32(1.) - a))


def _build_time_lookup(reader, year):
    """Build (month, day, hour) -> time_index lookup from netCDF time coordinate.
    Works with any CMIP calendar (365_day, 360_day, proleptic_gregorian, etc.).
    For ESGF layouts, collects only timesteps matching the requested year.
    """
    import netCDF4 as nc4

    if reader.cfg.file_layout == "cmip6":
        files = reader._esgf_find_all("psl", year)
        year_dts = []
        seen = set()
        for path in files:
            ds = nc4.Dataset(str(path))
            tname = next((n for n in ['time', 'Time', 'TIME'] if n in ds.variables), 'time')
            tvar = ds.variables[tname]
            tunits = getattr(tvar, 'units', '')
            tcal = getattr(tvar, 'calendar', 'standard')
            dts = nc4.num2date(tvar[:], tunits, tcal)
            for dt in dts:
                if dt.year == year:
                    # De-dup with the same key reader._read_3d_esgf uses (incl. minute),
                    # so this lookup's indices align with the reordered data array.
                    key = (dt.month, dt.day, dt.hour, getattr(dt, "minute", 0))
                    if key not in seen:
                        seen.add(key)
                        year_dts.append(dt)
            ds.close()
    elif reader.cfg.file_layout == "split_monthly":
        import datetime
        year_dts = []
        for month in range(1, 13):
            path = reader._nc_path("psl", year, month=month)
            if not path.exists():
                continue
            ds = nc4.Dataset(str(path))
            tname = next((n for n in ['time', 'Time', 'TIME'] if n in ds.variables), 'time')
            tvar = ds.variables[tname]
            tunits = getattr(tvar, 'units', '')
            if "since" in tunits:
                tcal = getattr(tvar, 'calendar', 'standard')
                dts = nc4.num2date(tvar[:], tunits, tcal)
                year_dts.extend(dts)
            else:
                for val in tvar[:]:
                    fval = float(val)
                    if 10000000 < fval < 99999999:
                        idate = int(fval)
                        frac = fval - idate
                        year_dts.append(datetime.datetime(
                            idate // 10000,
                            (idate % 10000) // 100,
                            idate % 100,
                            round(frac * 24)))
                    elif year_dts:
                        year_dts.append(year_dts[-1] + datetime.timedelta(hours=6))
            ds.close()
    else:
        path = reader._nc_path("psl", year)
        ds = nc4.Dataset(str(path))
        tvar = ds.variables['time']
        tunits = getattr(tvar, 'units', '')
        tcal = getattr(tvar, 'calendar', 'standard')
        year_dts = list(nc4.num2date(tvar[:], tunits, tcal))
        ds.close()

    # Order chronologically. For ssp runs the start-year 00Z step comes from the
    # preceding historical file and is collected last; reader._read_3d_esgf reorders
    # timesteps chronologically, so this index lookup (used to map env params onto
    # the read array) must match that order.
    year_dts.sort(key=lambda dt: (int(dt.month), int(dt.day), int(dt.hour),
                                  int(getattr(dt, "minute", 0))))
    lookup = {}
    for i, dt in enumerate(year_dts):
        lookup[(int(dt.month), int(dt.day), int(dt.hour))] = i
    return lookup


class EnvironmentalCalculator:
    """
    Calculate environmental parameters around detected track centers.

    Usage:
        calc = EnvironmentalCalculator(cfg)
        env_data = calc.compute(tracks)
        calc.save_envi(tracks, env_data, "output/DBTRACK_ERA5_envi.txt")
    """

    def __init__(self, cfg: ModelConfig):
        self.cfg = cfg
        self.reader = SmartReader(cfg)
        self._dist_cache = None  # initialized as list in init_env_grid()
        self._di_arr = None      # pre-computed offset array

    def init_env_grid(self, lon, lat, dnx, dny):
        """Pre-compute grid constants for env calculation.
        Must be called once before _compute_point_env().
        """
        self._nlat = len(lat)
        self._nlon = len(lon)
        self._dnx = dnx
        self._dny = dny
        self._di_arr = np.arange(-dnx, dnx + 1)
        self._dist_cache = [None] * self._nlat

        # Pre-compute reference lonv for distance grid (ix-independent)
        idx_arr = self._di_arr % self._nlon
        ref_lonv = lon[idx_arr].copy()
        ref_lonv[self._di_arr < 0] -= 360.0
        ref_lonv[self._di_arr >= self._nlon] += 360.0
        self._ref_lonv = ref_lonv

    def _get_distkm_grid(self, iy, lat):
        """Get distance grid with list-based iy cache (O(1) lookup)."""
        if self._dist_cache[iy] is not None:
            return self._dist_cache[iy]

        dnx = self._dnx
        dny = self._dny
        ny = 2 * dny + 1

        # Build latv
        j_indices = np.arange(iy - dny, iy + dny + 1)
        latv = np.full(ny, np.nan, dtype=np.float32)
        valid = (j_indices >= 0) & (j_indices < self._nlat)
        latv[valid] = lat[j_indices[valid]]

        # Vectorized distance grid
        lon_grid, lat_grid = np.meshgrid(self._ref_lonv, latv, indexing='ij')
        distkm_grid = np.full((2 * dnx + 1, ny), np.float32(9999.), dtype=np.float32)
        lat_valid = ~np.isnan(lat_grid)
        if lat_valid.any():
            dist = _haversine_km_vectorized(
                self._ref_lonv[dnx], latv[dny],
                lon_grid[lat_valid], lat_grid[lat_valid])
            distkm_grid[lat_valid] = dist

        self._dist_cache[iy] = (distkm_grid, latv)
        return distkm_grid, latv

    def _compute_point_env(self, fields, ix, iy, lon, lat, nlon, dnx, dny):
        """Compute env parameters for a single point given pre-loaded 2D fields."""
        missing = 1e20

        # Get cached distance grid and latv
        distkm_grid, latv = self._get_distkm_grid(iy, lat)

        # Build lon index array and lonv (shared across crop + vorticity)
        lon_idx = (ix + self._di_arr) % nlon
        lonv = lon[lon_idx].copy()
        raw_idx = ix + self._di_arr
        lonv[raw_idx < 0] -= 360.0
        lonv[raw_idx >= nlon] += 360.0

        # Crop 7 fields with shared lon_idx (no Python loop)
        j_start = iy - dny
        j_end = iy + dny + 1
        if j_start >= 0 and j_end <= lat.shape[0]:
            u850 = fields["ua_850"][lon_idx, j_start:j_end].copy()
            v850 = fields["va_850"][lon_idx, j_start:j_end].copy()
            u250 = fields["ua_250"][lon_idx, j_start:j_end].copy()
            v250 = fields["va_250"][lon_idx, j_start:j_end].copy()
            t250 = fields["ta_250"][lon_idx, j_start:j_end].copy()
            t500 = fields["ta_500"][lon_idx, j_start:j_end].copy()
            psl_reg = fields["psl"][lon_idx, j_start:j_end].copy()
        else:
            _shape = (2 * dnx + 1, 2 * dny + 1)
            u850 = np.full(_shape, np.nan, dtype=np.float32)
            v850 = np.full(_shape, np.nan, dtype=np.float32)
            u250 = np.full(_shape, np.nan, dtype=np.float32)
            v250 = np.full(_shape, np.nan, dtype=np.float32)
            t250 = np.full(_shape, np.nan, dtype=np.float32)
            t500 = np.full(_shape, np.nan, dtype=np.float32)
            psl_reg = np.full(_shape, np.nan, dtype=np.float32)

        # Replace NaN with missing
        for arr in [u850, v850, u250, v250, t250, t500, psl_reg]:
            arr[np.isnan(arr)] = missing

        # Single Fortran call: vorticity + derived + azimuthal means + metrics
        bin_km = np.float32(100.)
        naz = 11
        wspd_az_max, vws_az_lg, vor_cnt, psl_az_cnt, \
            warmcore_250, warmcore_250_500 = calc_env_all(
                u850, v850, u250, v250, t250, t500, psl_reg,
                distkm_grid, lonv, latv,
                np.float32(missing), bin_km, naz,
                np.float32(ENV_VOR_INNER_KM), np.float32(ENV_WARMCORE_RADIUS_KM),
                int(ENV_WSPD_RADIUS_KM / bin_km),
                int(ENV_PSL_RADIUS_KM / bin_km),
                int(ENV_VOR_OUTER_START_KM / bin_km),
                int(ENV_VOR_OUTER_KM / bin_km),
            )

        return {
            "wspd_az_max": wspd_az_max,
            "vws_az_lg": vws_az_lg,
            "vor_cnt": vor_cnt,
            "psl_az_cnt": psl_az_cnt,
            "warmcore_250": warmcore_250,
            "warmcore_250_500": warmcore_250_500,
        }

    def _acquire_env_lock(self, lock_path: Path):
        """Concurrency guard (#6) via an advisory POSIX lock (`fcntl.lockf`).

        The kernel holds the lock against the open fd and releases it automatically
        when the process exits — for ANY reason — so there is no stale lock to detect,
        no pid-liveness guessing, and no reclaim/delete race (closes the rename race).
        We never unlink the lock file; only the lock STATE matters, so a leftover empty
        file is harmless and cannot be mistaken for a held lock.

        Returns the open fd, which the caller must keep until release.
        (POSIX locks are coordinated across hosts by the NFS lock manager when the
        export supports it; if not, this degrades to same-host scope — the content
        preflights #4/#7 remain the cross-run safety net regardless.)"""
        fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR, 0o644)
        try:
            fcntl.lockf(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as e:
            if e.errno in (errno.EACCES, errno.EAGAIN):
                # Genuinely held by another process -> refuse.
                try:
                    holder = os.read(fd, 64).decode(errors="replace").strip() or "?"
                except Exception:
                    holder = "?"
                os.close(fd)
                raise RuntimeError(
                    f"env stage already running (holder {holder}) for {lock_path.parent}; "
                    f"refusing to run concurrently.")
            # Locking unsupported on this filesystem (e.g. ENOLCK/EOPNOTSUPP on some
            # NFS exports). Don't falsely block every run — degrade to no lock; the
            # content preflights (#4/#7) remain the cross-run corruption guard.
            print(f"  WARNING: advisory env lock unavailable ({e}); proceeding without lock",
                  flush=True)
            return fd
        try:
            os.ftruncate(fd, 0)
            os.write(fd, f"{os.getpid()}\n".encode())
        except OSError:
            pass
        return fd

    def _release_env_lock(self, fd) -> None:
        """Release the advisory lock and close the fd (#4/#6). The file is left on
        disk on purpose — unlinking would reopen a create race."""
        if fd is None:
            return
        try:
            fcntl.lockf(fd, fcntl.LOCK_UN)
        except OSError:
            pass
        finally:
            try:
                os.close(fd)
            except OSError:
                pass

    def compute_and_save(self, tracks: List[Track], outpath: str) -> None:
        """Lock-guarded entry point: acquire env lock, run the stage, ALWAYS release
        the lock (#4, #6) even on exception."""
        if not tracks:
            print("No tracks to compute env parameters for")
            Path(outpath).parent.mkdir(parents=True, exist_ok=True)
            open(outpath, "w").close()
            return
        outdir = Path(outpath).parent
        outdir.mkdir(parents=True, exist_ok=True)  # lock create needs parent to exist
        base = Path(outpath).stem
        lock_path = outdir / f".{base}.env.lock"
        lock_fd = self._acquire_env_lock(lock_path)
        try:
            self._compute_and_save_locked(tracks, outpath)
        finally:
            self._release_env_lock(lock_fd)

    def _compute_and_save_locked(self, tracks: List[Track], outpath: str) -> None:
        """
        Compute environmental parameters and save to file.
        Processes year-by-year to limit memory usage.
        Supports resume: skips years whose tmp file already exists.
        """
        first_year = min(int(y) for trk in tracks for y in trk.year)
        lon, lat = self.reader.read_lonlat(first_year)
        dx_deg = abs(lon[1] - lon[0]) if len(lon) > 1 else 1.0
        dnx = int(np.ceil(ENV_REGION_DEG / dx_deg))
        dny = dnx
        nlon = len(lon)
        self.init_env_grid(lon, lat, dnx, dny)

        outdir = Path(outpath).parent
        base = Path(outpath).stem  # e.g. DBTRACK_ERA5_anal_1998-2024_envi

        # Run-level fingerprint of the *current* DBTRACK track set. Embedding it in
        # the tmp-dir name means a re-detected DBTRACK gets a different scratch dir,
        # so stale tmp from another run can never be reused (prevention #3).
        run_fp = _fingerprint(_global_keys(tracks))
        tmp_dir = outdir / f"{base}_tmp_{run_fp}"
        tmp_dir.mkdir(parents=True, exist_ok=True)

        # DBTRACK mtime for the freshness guard (#2): DBTRACK is the envi base name
        # minus the trailing "_envi".
        dbtrack_name = base[:-5] if base.endswith("_envi") else base
        dbtrack_file = outdir / f"{dbtrack_name}.txt"
        dbtrack_mtime = dbtrack_file.stat().st_mtime if dbtrack_file.exists() else 0.0

        # Drop orphan tmp dirs from earlier/other DBTRACK runs (disk hygiene + #3).
        for old in outdir.glob(f"{base}_tmp*"):
            if old.is_dir() and old != tmp_dir:
                shutil.rmtree(old, ignore_errors=True)

        def _rss_mb():
            try:
                return int(open("/proc/self/status").read().split("VmRSS:")[1].split()[0]) // 1024
            except Exception:
                return 0

        # Index points by year -> {year: [(track_idx, point_idx, ix, iy), ...]}
        print("Computing environmental parameters...", flush=True)
        print(f"  Indexing {len(tracks)} tracks... (RSS={_rss_mb()} MB)", flush=True)
        points_by_year = defaultdict(list)
        for t_idx, trk in enumerate(tracks):
            for p_idx in range(len(trk.lon)):
                ix = int(np.argmin(np.abs(lon - trk.lon[p_idx])))
                iy = int(np.argmin(np.abs(lat - trk.lat[p_idx])))
                year = int(trk.year[p_idx])
                points_by_year[year].append((t_idx, p_idx, ix, iy))

        all_years = sorted(points_by_year.keys())
        total_points = sum(len(trk.lon) for trk in tracks)
        points_done = 0
        print(f"  Indexing done: {total_points} points, years {all_years[0]}-{all_years[-1]} (RSS={_rss_mb()} MB)", flush=True)

        # Fixed fields for TC2 method — tied to _compute_point_env() field names
        field_specs = [
            ("ua", 850), ("va", 850),
            ("ua", 250), ("va", 250),
            ("ta", 250), ("ta", 500),
            ("psl", None),
        ]

        # --- Year-by-year computation ---
        for year in all_years:
            tmp_file = tmp_dir / f"{year}.txt"
            year_points = points_by_year[year]

            year_fp = _fingerprint(_year_keys(tracks, year_points))
            expected_header = f"# {self.cfg.name} {year} {len(year_points)} sha256={year_fp}"

            # Resume check: reuse tmp only if it was written from an IDENTICAL DBTRACK
            # (per-year fingerprint, #1) and is not older than DBTRACK (#2).
            if tmp_file.exists():
                with open(tmp_file) as fchk:
                    header = fchk.readline().rstrip()
                    n_lines = sum(1 for _ in fchk)
                fresh = tmp_file.stat().st_mtime >= dbtrack_mtime
                if header == expected_header and n_lines == len(year_points) and fresh:
                    points_done += len(year_points)
                    print(f"  Year {year}: skip ({n_lines} points, tmp verified)", flush=True)
                    continue
                # Stale / mismatched / older than DBTRACK — recompute
                tmp_file.unlink()

            # Build time lookup from netCDF (handles any CMIP calendar)
            time_lookup = _build_time_lookup(self.reader, year)

            # Group this year's points by time_idx
            by_time = defaultdict(list)
            skip_count = 0
            for t_idx, p_idx, ix, iy in year_points:
                trk = tracks[t_idx]
                key = (int(trk.month[p_idx]), int(trk.day[p_idx]), int(trk.hour[p_idx]))
                it = time_lookup.get(key)
                if it is None:
                    skip_count += 1
                    continue
                by_time[it].append((t_idx, p_idx, ix, iy))
            if skip_count > 0:
                print(f"  WARNING: {skip_count} points skipped (not in time axis)", flush=True)

            ds_cache = {}
            year_done = 0
            n_times = len(by_time)
            t_processed = 0
            print(f"  Year {year}: {len(year_points)} points, {n_times} timesteps (RSS={_rss_mb()} MB)", flush=True)
            tmp_part = tmp_file.with_name(tmp_file.name + ".part")  # atomic write (#5)
            try:
                with open(tmp_part, "w") as f:
                    f.write(expected_header + "\n")
                    for it in sorted(by_time.keys()):
                        # Read 7 2D fields for this timestep
                        fields = {}
                        for var_key, lev_hpa in field_specs:
                            field_name = f"{var_key}_{lev_hpa}" if lev_hpa else var_key
                            fields[field_name] = self.reader.read_env_2d(var_key, year, lev_hpa, it, ds_cache)

                        for t_idx, p_idx, ix, iy in by_time[it]:
                            trk = tracks[t_idx]
                            env = self._compute_point_env(fields, ix, iy, lon, lat, nlon, dnx, dny)
                            f.write(f"{trk.track_id} {trk.year[p_idx]:04d} {trk.month[p_idx]:02d} "
                                    f"{trk.day[p_idx]:02d} {trk.hour[p_idx]:02d} "
                                    f"{trk.lon[p_idx]:8.2f} {trk.lat[p_idx]:8.2f} "
                                    f"{env['wspd_az_max']:8.2f} {env['vws_az_lg']:8.2f} "
                                    f"{env['vor_cnt']:8.2f} {env['psl_az_cnt']:8.2f} "
                                    f"{env['warmcore_250']:8.2f} {env['warmcore_250_500']:8.2f} "
                                    f"  0\n")
                            year_done += 1

                        t_processed += 1
                        if t_processed == 1 or t_processed % 200 == 0:
                            print(f"    {t_processed}/{n_times} timesteps, {year_done} points", flush=True)
                os.replace(tmp_part, tmp_file)
            finally:
                if tmp_part.exists():
                    tmp_part.unlink()
                for v in ds_cache.values():
                    if hasattr(v, "close"):
                        v.close()

            points_done += year_done
            print(f"  Year {year} done: {year_done} points ({points_done}/{total_points} total)", flush=True)

        # --- Concat: read all tmp files, group by track_id, write final envi.txt ---
        print("  Assembling final envi file...")
        # Read all tmp lines into per-track buckets
        # Use list of (point_sort_key, line) per track to maintain chronological order
        track_lines = defaultdict(list)
        for year in all_years:
            tmp_file = tmp_dir / f"{year}.txt"
            with open(tmp_file) as f:
                for line in f:
                    if line.startswith("#"):
                        continue
                    parts = line.split()
                    tid = int(parts[0])
                    # sort key: year month day hour
                    sort_key = (int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4]))
                    track_lines[tid].append((sort_key, line))

        # Preflight (#4): the assembled tmp track set must EXACTLY equal the current
        # DBTRACK — bidirectional. Subset-only would pass when points are MISSING
        # (e.g. skip_count>0 dropping points -> partial envi). Here every track's
        # (time,lon,lat) key set must match both ways, so stale/extra AND missing/
        # partial tmp both fail fast; a silently partial envi is never written.
        # Sorted-list (multiset) comparison, not set — a duplicated (time,lon,lat)
        # point would change the count but not the set, so set-equality could miss it
        # (#5). Sorted lists catch count differences too.
        def _track_keylist(trk):
            return sorted(
                (int(trk.year[p]), int(trk.month[p]), int(trk.day[p]), int(trk.hour[p]),
                 round(float(trk.lon[p]), 2), round(float(trk.lat[p]), 2))
                for p in range(len(trk.lon))
            )
        cur_tids = {trk.track_id for trk in tracks}
        problems = []
        for trk in tracks:
            want = _track_keylist(trk)
            got = sorted(
                (int(p[1]), int(p[2]), int(p[3]), int(p[4]),
                 round(float(p[5]), 2), round(float(p[6]), 2))
                for p in (line.split() for _sk, line in track_lines.get(trk.track_id, []))
            )
            if got != want:
                problems.append((trk.track_id, len(want), len(got)))
        orphan_tids = [tid for tid in track_lines if tid not in cur_tids]
        if problems or orphan_tids:
            raise RuntimeError(
                f"env preflight FAILED for {outpath}: "
                f"{len(problems)} track(s) with mismatched point set "
                f"(tid,want,got e.g. {problems[:3]}); "
                f"{len(orphan_tids)} orphan tmp tid(s) (e.g. {orphan_tids[:3]}). "
                f"Stale/partial tmp in {tmp_dir}.")

        # Write final file in track order, atomically (#5)
        Path(outpath).parent.mkdir(parents=True, exist_ok=True)
        out_part = Path(str(outpath) + ".part")
        with open(out_part, "w") as f:
            for trk in tracks:
                tid = trk.track_id
                lines = track_lines.get(tid, [])
                lines.sort(key=lambda x: x[0])
                nstep = len(lines)
                if nstep == 0:
                    continue
                first_year = lines[0][0][0]
                f.write(f"{first_year:04d}    {nstep:6d}    {tid:6d}\n")
                for _, line in lines:
                    # Strip track_id from tmp format, reformat to envi format
                    parts = line.split()
                    # parts: tid yr mo dy hr lon lat wspd vws vor psl wc250 wc500 flag
                    f.write(f"{int(parts[1]):04d} {int(parts[2]):02d} {int(parts[3]):02d} "
                            f"{int(parts[4]):02d}  {float(parts[5]):8.2f} {float(parts[6]):8.2f} "
                            f"{float(parts[7]):8.2f} {float(parts[8]):8.2f} "
                            f"{float(parts[9]):8.2f} {float(parts[10]):8.2f} "
                            f"{float(parts[11]):8.2f} {float(parts[12]):8.2f} "
                            f"{int(parts[13]):3d}\n")

        os.replace(out_part, outpath)
        print(f"Saved env parameters to {outpath}")
