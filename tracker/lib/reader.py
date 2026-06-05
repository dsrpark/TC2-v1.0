"""
Smart netCDF reader using netCDF4 library directly.

Reads any netCDF file and normalizes to standard internal representation:
  - Longitude: 0~360, ascending
  - Latitude:  90~-90 (N→S)
  - Pressure:  Pa
  - Wind:      m/s
  - Temperature: K

All returned arrays are in Fortran memory order: (nlon, nlat, ntime).
"""
import re
import numpy as np
import netCDF4 as nc
from pathlib import Path
from typing import Optional, Tuple, List
from .config_loader import ModelConfig, STANDARD_UNITS

# CMIP6/HighResMIP ESGF filename: ..._{start}-{end}.nc
_ESGF_TIME_RE = re.compile(r'_(\d{10,12})-(\d{10,12})\.nc$')


# ============================================================
# Variable / dimension name detection
# ============================================================
VARNAME_PATTERNS = {
    "ua":   ["ua", "u", "U", "UGRD", "u_wind", "uwnd"],
    "va":   ["va", "v", "V", "VGRD", "v_wind", "vwnd"],
    "ta":   ["ta", "t", "T", "TMP", "air_temperature", "temp"],
    "psl":  ["psl", "msl", "PRMSL", "slp", "SLP", "mslp"],
    "mask": ["lsm", "land", "LAND", "landmask", "sftlf"],
}

DIM_PATTERNS = {
    "lat":  ["lat", "latitude", "y", "LAT", "LATITUDE"],
    "lon":  ["lon", "longitude", "x", "LON", "LONGITUDE"],
    "lev":  ["lev", "level", "plev", "pressure", "isobaricInhPa", "p"],
    "time": ["time", "TIME", "Time"],
}

CF_STANDARD_NAMES = {
    "eastward_wind": "ua",
    "northward_wind": "va",
    "air_temperature": "ta",
    "air_pressure_at_mean_sea_level": "psl",
    "air_pressure_at_sea_level": "psl",
    "land_binary_mask": "mask",
    "land_area_fraction": "mask",
}


def _find_var(ds, key: str, explicit: Optional[str] = None) -> Optional[str]:
    """Find variable name: explicit → CF standard_name → pattern."""
    if explicit and explicit in ds.variables:
        return explicit
    for vname in ds.variables:
        sn = getattr(ds.variables[vname], "standard_name", "")
        if sn in CF_STANDARD_NAMES and CF_STANDARD_NAMES[sn] == key:
            return vname
    for pattern in VARNAME_PATTERNS.get(key, []):
        if pattern in ds.variables:
            return pattern
    return None


def _find_dim(ds, dim_type: str, explicit: Optional[str] = None) -> Optional[str]:
    """Find dimension/coordinate name."""
    if explicit and explicit in ds.variables:
        return explicit
    for pattern in DIM_PATTERNS.get(dim_type, []):
        if pattern in ds.variables:
            return pattern
    return None


def _to_float32(data: np.ndarray) -> np.ndarray:
    """Convert to float32 to match original Fortran real(4) precision."""
    if isinstance(data, np.ma.MaskedArray):
        data = data.filled(np.nan)
    return data.astype(np.float32)


# ============================================================
# Unit conversion
# ============================================================
def _convert_units(data: np.ndarray, src_unit: str, var_type: str) -> np.ndarray:
    """Convert to standard internal units if needed."""
    dst = STANDARD_UNITS.get(var_type)
    if not dst:
        return data
    src = src_unit.strip().replace("**", "").replace("  ", " ")
    dst_clean = dst.replace("-", "")
    src_clean = src.replace("-", "")
    if src_clean.lower() == dst_clean.lower() or src.lower() == dst.lower():
        return data

    conversions = {
        "hpa": ("pressure", lambda x: x * np.float32(100.0)),
        "mb": ("pressure", lambda x: x * np.float32(100.0)),
        "knots": ("wind", lambda x: x * np.float32(0.514444)),
        "kt": ("wind", lambda x: x * np.float32(0.514444)),
        "degc": ("temperature", lambda x: x + np.float32(273.15)),
        "degrees_c": ("temperature", lambda x: x + np.float32(273.15)),
        "celsius": ("temperature", lambda x: x + np.float32(273.15)),
    }
    for key, (vtype, fn) in conversions.items():
        if src.lower().startswith(key) and var_type == vtype:
            return fn(data)

    return data


class SmartReader:
    """
    Reads netCDF data with automatic variable/unit/coordinate detection.
    Uses netCDF4 directly (no xarray dependency at runtime).

    All returned data arrays are (nlon, nlat, ntime) in Fortran order,
    matching the original Fortran code's memory layout.
    """

    def __init__(self, cfg: ModelConfig):
        self.cfg = cfg
        self.ncdir = Path(cfg.ncdir)
        self._lon = None
        self._lat = None
        self._lat_is_n2s = None  # True if raw file has N→S
        self._lev_idx_cache = {}  # instance-level cache for level indices
        self._esgf_cache = {}    # cache for ESGF file lookups

    def _nc_path(self, var_key: str, year: int, month: Optional[int] = None,
                 level_hpa: Optional[int] = None) -> Path:
        """Return path to file for var_key, year, and optionally month/level.

        file_pattern placeholders: {var}, {year}, {month}, {level}
        For split-level layouts (e.g., FNL), use pattern like "{var}{level}/{year}{month}.nc"
        """
        var_name = getattr(self.cfg, f"var_{var_key}", None) or var_key
        if self.cfg.file_layout == "cmip6":
            return self._esgf_find_primary(var_key, year)
        if self.cfg.file_layout == "split_monthly" and month is None:
            month = 1
        fmt = {"var": var_name, "year": f"{year:04d}"}
        if month is not None:
            fmt["month"] = f"{month:02d}"
        fmt["level"] = str(level_hpa) if level_hpa else ""
        return self.ncdir / self.cfg.file_pattern.format(**fmt)

    def _monthly_paths(self, var_key: str, year: int,
                       level_hpa: Optional[int] = None) -> List[Path]:
        """Return list of 12 monthly file paths for a year."""
        return [self._nc_path(var_key, year, m, level_hpa) for m in range(1, 13)]

    # --------------------------------------------------------
    # ESGF file discovery (CMIP6, HighResMIP, etc.)
    # --------------------------------------------------------
    def _esgf_var_dir(self, var_key: str) -> Path:
        """Return the variable directory under ncdir."""
        var_name = getattr(self.cfg, f"var_{var_key}", None) or var_key
        sfc_vars = set(self.cfg.sfc_vars)
        group = "sfc" if (var_key in sfc_vars or var_name in sfc_vars) else "pres"
        var_dir = self.ncdir / group / var_name
        if not var_dir.is_dir():
            raise FileNotFoundError(f"Variable directory not found: {var_dir}")
        return var_dir

    def _historical_var_dir(self, var_key: str) -> Optional[Path]:
        """Return the historical-experiment counterpart of this variable dir.

        CMIP6 ScenarioMIP: an ssp run branches from historical at the start year
        (e.g. 2015), and CMIP6 chunk files end at Jan 1 00Z. So an ssp variable's
        very first timestep (start-year 00Z) lives at the END of the preceding
        historical file. Returning the historical dir lets _esgf_find_all reach
        that one boundary timestep. General to every ssp model (not a per-model
        exception). Returns None when not an ssp run or the dir is absent.
        """
        parts = list(self.ncdir.parts)
        try:
            i = parts.index("CMIP6")
        except ValueError:
            return None
        if i + 1 >= len(parts):
            return None
        exp = parts[i + 1]
        if not exp.startswith("ssp"):
            return None
        parts[i + 1] = "historical"
        var_name = getattr(self.cfg, f"var_{var_key}", None) or var_key
        sfc_vars = set(self.cfg.sfc_vars)
        group = "sfc" if (var_key in sfc_vars or var_name in sfc_vars) else "pres"
        hist_dir = Path(*parts) / group / var_name
        return hist_dir if hist_dir.is_dir() else None

    def _esgf_find_all(self, var_key: str, year: int) -> List[Path]:
        """Find all ESGF files containing any timestep in the given year.

        ESGF files often end at Jan 1 00Z of the next year, so the previous
        file may contain the first timestep (00Z) of the requested year.
        """
        cache_key = f"{var_key}_{year}"
        if cache_key in self._esgf_cache:
            return self._esgf_cache[cache_key]

        var_dir = self._esgf_var_dir(var_key)
        # Also scan the preceding historical dir for ssp runs so the start-year
        # 00Z boundary timestep (stored at the end of the last historical file)
        # is reachable. General to all ssp models — the time-range filter below
        # admits only the single historical file ending at this year's Jan 1 00Z.
        search_dirs = [var_dir]
        hist_dir = self._historical_var_dir(var_key)
        if hist_dir is not None:
            search_dirs.append(hist_dir)

        result = []
        for sdir in search_dirs:
            for f in sorted(sdir.glob("*.nc")):
                m = _ESGF_TIME_RE.search(f.name)
                if not m:
                    continue
                s, e = m.group(1), m.group(2)
                sy, ey = int(s[:4]), int(e[:4])
                em, ed = int(e[4:6]), int(e[6:8])
                # File ending YYYY0101 00Z: last real data year is YYYY-1
                if em == 1 and ed <= 1:
                    data_end_year = ey  # but contains 00Z of ey
                else:
                    data_end_year = ey
                if sy <= year <= data_end_year:
                    result.append(f)
                elif ey == year and em == 1:
                    result.append(f)

        if not result:
            raise FileNotFoundError(
                f"No ESGF file for {var_key} year {year} in {var_dir}")
        self._esgf_cache[cache_key] = result
        return result

    def _esgf_find_primary(self, var_key: str, year: int) -> Path:
        """Find the primary ESGF file for a year (for coord reading etc)."""
        files = self._esgf_find_all(var_key, year)
        for f in files:
            m = _ESGF_TIME_RE.search(f.name)
            if m and int(m.group(1)[:4]) <= year:
                return f
        return files[-1]

    def _esgf_build_year_map(self, var_key: str, year: int, ds_cache: dict):
        """Build mapping from year-relative index to (Dataset, file-local index).

        Returns list of (ds, file_local_index) for each timestep in the year.
        Cached per var_key+year.
        """
        map_key = f"_esgf_map_{var_key}_{year}"
        if map_key in ds_cache:
            return ds_cache[map_key]

        files = self._esgf_find_all(var_key, year)
        # Collect with a chronological sort key. Steps may come from more than one
        # file (e.g. the start-year 00Z from the preceding historical file), so we
        # must order by actual timestamp, not file/append order.
        entries = []  # [(sortkey, ds_handle, local_index), ...]
        seen = set()
        for path in files:
            cache_key = f"{var_key}_{year}_{path.name}"
            if cache_key not in ds_cache:
                ds_cache[cache_key] = nc.Dataset(str(path))
            ds = ds_cache[cache_key]
            time_var = ds.variables[_find_dim(ds, "time") or "time"]
            cal = getattr(time_var, "calendar", "standard")
            times = nc.num2date(time_var[:], time_var.units, calendar=cal)
            for i, t in enumerate(times):
                if t.year == year:
                    key = (t.month, t.day, t.hour, getattr(t, "minute", 0))
                    if key not in seen:
                        seen.add(key)
                        entries.append((key, ds, i))

        entries.sort(key=lambda e: e[0])
        year_map = [(ds, i) for (_, ds, i) in entries]

        ds_cache[map_key] = year_map
        return year_map

    def _esgf_resolve_timestep(self, var_key: str, year: int, it: int,
                                ds_cache: dict):
        """Resolve year-relative time index to (Dataset, file-local index)."""
        year_map = self._esgf_build_year_map(var_key, year, ds_cache)
        if it < 0 or it >= len(year_map):
            raise IndexError(f"Time index {it} out of range for {var_key} year {year} "
                             f"(have {len(year_map)} timesteps)")
        return year_map[it]

    # --------------------------------------------------------
    # Monthly file resolution
    # --------------------------------------------------------
    def _monthly_build_year_map(self, var_key: str, year: int, ds_cache: dict,
                               level_hpa: Optional[int] = None):
        """Build mapping from year-relative index to (Dataset, file-local index) for monthly files."""
        lev_tag = f"_{level_hpa}" if level_hpa else ""
        map_key = f"_monthly_map_{var_key}{lev_tag}_{year}"
        if map_key in ds_cache:
            return ds_cache[map_key]

        year_map = []
        for month in range(1, 13):
            path = self._nc_path(var_key, year, month, level_hpa)
            if not path.exists():
                continue
            cache_key = f"{var_key}{lev_tag}_{year}_{month:02d}"
            if cache_key not in ds_cache:
                ds_cache[cache_key] = nc.Dataset(str(path))
            ds = ds_cache[cache_key]
            ntime = len(ds.variables[_find_dim(ds, "time") or "time"])
            for i in range(ntime):
                year_map.append((ds, i))

        ds_cache[map_key] = year_map
        return year_map

    def _monthly_resolve_timestep(self, var_key: str, year: int, it: int,
                                  ds_cache: dict, level_hpa: Optional[int] = None):
        """Resolve year-relative time index to (Dataset, file-local index) for monthly files."""
        year_map = self._monthly_build_year_map(var_key, year, ds_cache, level_hpa)
        if it < 0 or it >= len(year_map):
            raise IndexError(f"Time index {it} out of range for {var_key} year {year} "
                             f"(have {len(year_map)} timesteps)")
        return year_map[it]

    def _lat_is_n2s_for_ds(self, ds) -> bool:
        """Return True if this file's raw latitude axis is north-to-south."""
        lat_name = _find_dim(ds, "lat", self.cfg.dim_lat)
        lat = ds.variables[lat_name][:]
        return bool(lat[0] > lat[-1])

    # --------------------------------------------------------
    # Coordinates
    # --------------------------------------------------------
    def read_lonlat(self, year: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Read and cache lon/lat. Always returns N→S latitude."""
        if self._lon is not None:
            return self._lon, self._lat

        yr = year or self.cfg.styr
        path = self._nc_path("psl", yr)
        ds = nc.Dataset(str(path))
        lon_name = _find_dim(ds, "lon", self.cfg.dim_lon)
        lat_name = _find_dim(ds, "lat", self.cfg.dim_lat)

        lon = ds.variables[lon_name][:].astype(np.float32)
        lat = ds.variables[lat_name][:].astype(np.float32)
        ds.close()

        # Normalize lon to 0~360
        if np.any(lon < 0):
            lon = lon % 360.0

        # Check lat direction
        self._lat_is_n2s = (lat[0] > lat[-1])
        if not self._lat_is_n2s:
            lat = lat[::-1]

        self._lon = lon
        self._lat = lat
        return self._lon, self._lat

    # --------------------------------------------------------
    # Land mask
    # --------------------------------------------------------
    def read_mask(self) -> np.ndarray:
        """Read land mask. Returns (nlon, nlat) where 0=ocean."""
        ds = nc.Dataset(self.cfg.maskfile)
        var_name = _find_var(ds, "mask", self.cfg.var_mask)
        if var_name is None:
            raise ValueError(f"Cannot find mask variable in {self.cfg.maskfile}")

        lat_is_n2s = self._lat_is_n2s_for_ds(ds)
        data = ds.variables[var_name][:]
        data = _to_float32(data)
        ds.close()

        # Squeeze out time/extra dims → (lat, lon)
        while data.ndim > 2:
            data = data[0]

        # Binarize: land >= threshold → 1, else → 0
        # ESGF sftlf: 0-100%, legacy masks: 0-1 fraction or binary 0/1
        if np.nanmax(data) > 1.5:
            data = np.where(data >= 50.0, 1.0, 0.0).astype(np.float32)
        else:
            data = np.where(data >= 0.5, 1.0, 0.0).astype(np.float32)

        # Ensure N→S using the mask file's own latitude axis.
        if not lat_is_n2s:
            data = data[::-1, :]

        # (lat, lon) → (lon, lat) for Fortran compat
        data = data.T.copy()
        return data

    # --------------------------------------------------------
    # 3D variables (time, [lev], lat, lon)
    # --------------------------------------------------------
    def _read_3d(self, var_key: str, year: int,
                 level_pa: Optional[float] = None,
                 var_type: str = "wind") -> np.ndarray:
        """
        Read a variable for one year, optionally selecting a pressure level.
        For ESGF layouts, reads from multiple files and slices to [Jan 1 00Z, Dec 31 18Z].

        Returns:
            (nlon, nlat, ntime) float32 array
        """
        if self.cfg.file_layout == "cmip6":
            return self._read_3d_esgf(var_key, year, level_pa, var_type)
        if self.cfg.file_layout == "split_monthly":
            return self._read_3d_monthly(var_key, year, level_pa, var_type)
        return self._read_3d_single(var_key, year, level_pa, var_type)

    def _split_level(self, level_pa: Optional[float]) -> Optional[int]:
        """If file_pattern contains {level}, return hPa value; else None."""
        if level_pa and "{level}" in self.cfg.file_pattern:
            return int(level_pa / 100)
        return None

    def _read_3d_single(self, var_key: str, year: int,
                        level_pa: Optional[float] = None,
                        var_type: str = "wind") -> np.ndarray:
        """Read from a single yearly file."""
        lev_hpa = self._split_level(level_pa)
        path = self._nc_path(var_key, year, level_hpa=lev_hpa)
        file_level_pa = None if lev_hpa else level_pa
        print(f"  READ {path}" + (f" at {level_pa/100:.0f} hPa" if level_pa else ""))

        ds = nc.Dataset(str(path))
        lat_is_n2s = self._lat_is_n2s_for_ds(ds)
        data = self._extract_from_ds(ds, var_key, year, file_level_pa, var_type)
        ds.close()
        return self._finalize(data, lat_is_n2s)

    def _read_3d_monthly(self, var_key: str, year: int,
                         level_pa: Optional[float] = None,
                         var_type: str = "wind") -> np.ndarray:
        """Read from 12 monthly files and concatenate."""
        lev_hpa = self._split_level(level_pa)
        file_level_pa = None if lev_hpa else level_pa
        print(f"  READ split_monthly {var_key} {year}" +
              (f" at {level_pa/100:.0f} hPa" if level_pa else "") +
              " (12 files)")
        chunks = []
        for month in range(1, 13):
            path = self._nc_path(var_key, year, month=month, level_hpa=lev_hpa)
            if not path.exists():
                continue
            ds = nc.Dataset(str(path))
            lat_is_n2s = self._lat_is_n2s_for_ds(ds)
            chunk = self._extract_from_ds(ds, var_key, year, file_level_pa, var_type)
            ds.close()
            if chunk.ndim > 0 and chunk.shape[0] > 0:
                chunks.append(self._finalize(chunk, lat_is_n2s))
        if not chunks:
            raise ValueError(f"No data for {var_key} year {year}")
        return np.concatenate(chunks, axis=2)

    def _read_3d_esgf(self, var_key: str, year: int,
                      level_pa: Optional[float] = None,
                      var_type: str = "wind") -> np.ndarray:
        """Read from ESGF file(s), collecting unique timesteps for the year."""
        files = self._esgf_find_all(var_key, year)
        print(f"  READ ESGF {var_key} {year}" +
              (f" at {level_pa/100:.0f} hPa" if level_pa else "") +
              f" ({len(files)} file{'s' if len(files) > 1 else ''})")

        chunks = []
        all_keys = []   # chronological sort keys, aligned with the concatenated time axis
        seen = set()
        for path in files:
            ds = nc.Dataset(str(path))
            try:
                lat_is_n2s = self._lat_is_n2s_for_ds(ds)
                chunk, chunk_keys = self._extract_from_ds_dedup(
                    ds, var_key, year, level_pa, var_type, seen)
                if chunk is not None and chunk.shape[0] > 0:
                    chunks.append(self._finalize(chunk, lat_is_n2s))
                    all_keys.extend(chunk_keys)
                    seen.update(chunk_keys)
            finally:
                ds.close()

        if not chunks:
            raise ValueError(f"No data for {var_key} year {year}")

        data = np.concatenate(chunks, axis=2) if len(chunks) > 1 else chunks[0]

        # Timesteps may be collected across multiple files — e.g. an ssp run's
        # start-year 00Z boundary step comes from the preceding historical file and
        # is read last. Reorder the time axis chronologically so the detection core
        # (and the env-param time lookup in parameters.py) see a monotonic sequence.
        order = sorted(range(len(all_keys)), key=lambda k: all_keys[k])
        if order != list(range(len(all_keys))):
            data = np.asfortranarray(data[:, :, order])
        return data

    def _extract_from_ds_dedup(self, ds, var_key: str, year: int,
                               level_pa: Optional[float], var_type: str,
                               seen: set):
        """Like _extract_from_ds but skips timesteps already in `seen`.
        Returns (data_array, set_of_keys_added)."""
        var_name = _find_var(ds, var_key, getattr(self.cfg, f"var_{var_key}", None))
        if var_name is None:
            raise ValueError(f"Cannot find variable '{var_key}' in {ds.filepath()}")

        var = ds.variables[var_name]
        dims = var.dimensions
        units = getattr(var, "units", "")
        scale_factor = getattr(var, "scale_factor", None)
        add_offset = getattr(var, "add_offset", None)
        fill_val = getattr(var, "_FillValue", None)
        var.set_auto_maskandscale(False)

        # Find unique timestep indices for this year not already seen
        time_dim = _find_dim(ds, "time") or "time"
        time_var = ds.variables[time_dim]
        cal = getattr(time_var, "calendar", "standard")
        times = nc.num2date(time_var[:], time_var.units, calendar=cal)

        indices = []
        new_keys = []      # ordered, aligned 1:1 with `indices` (for chronological reorder)
        local = set()
        for i, t in enumerate(times):
            if t.year == year:
                key = (t.month, t.day, t.hour, getattr(t, "minute", 0))
                if key not in seen and key not in local:
                    local.add(key)
                    indices.append(i)
                    new_keys.append(key)

        if not indices:
            return np.empty((0,), dtype=np.float32), new_keys

        # Level selection
        lev_name = _find_dim(ds, "lev", self.cfg.dim_lev)
        lev_idx = None
        if level_pa is not None and lev_name and lev_name in dims:
            lev_vals = ds.variables[lev_name][:].astype(np.float32)
            if lev_vals.max() < 2000:
                lev_vals_pa = lev_vals * np.float32(100)
            else:
                lev_vals_pa = lev_vals
            lev_idx = int(np.argmin(np.abs(lev_vals_pa - np.float32(level_pa))))
            actual = lev_vals_pa[lev_idx]
            if abs(actual - level_pa) > 100:
                raise ValueError(f"No level near {level_pa/100:.0f} hPa")

        # Read selected timesteps in one request when they are contiguous.
        # CMIP6 yearly reads are usually contiguous; per-timestep reads make
        # detection I/O-bound on network or compressed NetCDF storage.
        slices = [slice(None)] * var.ndim
        if indices == list(range(indices[0], indices[-1] + 1)):
            slices[0] = slice(indices[0], indices[-1] + 1)
            if lev_idx is not None:
                lev_dim_pos = dims.index(lev_name)
                slices[lev_dim_pos] = lev_idx
            data = var[tuple(slices)]
        else:
            slabs = []
            for idx in indices:
                slices = [slice(None)] * var.ndim
                slices[0] = idx
                if lev_idx is not None:
                    lev_dim_pos = dims.index(lev_name)
                    slices[lev_dim_pos] = lev_idx
                slabs.append(var[tuple(slices)])
            data = np.stack(slabs, axis=0)

        # Fill mask, scale/offset, unit conversion
        fill_mask = (data == fill_val) if fill_val is not None else None
        data = data.astype(np.float32)
        if scale_factor is not None:
            data = data * np.float32(scale_factor) + np.float32(add_offset or 0)
        if fill_mask is not None:
            data[fill_mask] = np.nan
        if units:
            data = _convert_units(data, units, var_type)

        return data, new_keys

    def _extract_from_ds(self, ds, var_key: str, year: int,
                         level_pa: Optional[float], var_type: str) -> np.ndarray:
        """Extract data for one year from an open Dataset. Returns (time, lat, lon)."""
        var_name = _find_var(ds, var_key, getattr(self.cfg, f"var_{var_key}", None))
        if var_name is None:
            raise ValueError(f"Cannot find variable '{var_key}' in {ds.filepath()}")

        var = ds.variables[var_name]
        dims = var.dimensions

        units = getattr(var, "units", "")
        scale_factor = getattr(var, "scale_factor", None)
        add_offset = getattr(var, "add_offset", None)
        fill_val = getattr(var, "_FillValue", None)
        var.set_auto_maskandscale(False)

        # Time slicing: select only timesteps in the requested year
        time_dim = _find_dim(ds, "time") or "time"
        time_var = ds.variables[time_dim]
        time_units = getattr(time_var, "units", "")
        if "since" in time_units:
            cal = getattr(time_var, "calendar", "standard")
            times = nc.num2date(time_var[:], time_units, calendar=cal)
            indices = [i for i, t in enumerate(times) if t.year == year]
            if not indices:
                return np.empty((0,), dtype=np.float32)
            time_slice = slice(indices[0], indices[-1] + 1)
        else:
            # Non-CF time axis (e.g., YYYYMMDD.fraction) — read all timesteps
            time_slice = slice(None)

        # Level selection
        lev_name = _find_dim(ds, "lev", self.cfg.dim_lev)
        slices = [slice(None)] * var.ndim
        slices[0] = time_slice

        if level_pa is not None and lev_name and lev_name in dims:
            lev_vals = ds.variables[lev_name][:].astype(np.float32)
            if lev_vals.max() < 2000:
                lev_vals_pa = lev_vals * np.float32(100)
            else:
                lev_vals_pa = lev_vals
            lev_idx = int(np.argmin(np.abs(lev_vals_pa - np.float32(level_pa))))
            actual = lev_vals_pa[lev_idx]
            if abs(actual - level_pa) > 100:
                raise ValueError(f"No level near {level_pa/100:.0f} hPa, closest={actual/100:.0f}")
            lev_dim_pos = dims.index(lev_name)
            slices[lev_dim_pos] = lev_idx

        data = var[tuple(slices)]

        # Fill mask, scale/offset, unit conversion
        fill_mask = (data == fill_val) if fill_val is not None else None
        data = data.astype(np.float32)
        if scale_factor is not None:
            data = data * np.float32(scale_factor) + np.float32(add_offset or 0)
        if fill_mask is not None:
            data[fill_mask] = np.nan
        if units:
            data = _convert_units(data, units, var_type)

        return data

    def _finalize(self, data: np.ndarray, lat_is_n2s: Optional[bool] = None) -> np.ndarray:
        """Transpose to (lon, lat, time), flip lat if needed."""
        # Preserve the leading time dimension even when a file contributes
        # only one timestep; squeeze singleton level/extra dimensions only.
        squeeze_axes = tuple(i for i, size in enumerate(data.shape) if i != 0 and size == 1)
        if squeeze_axes:
            data = np.squeeze(data, axis=squeeze_axes)
        if data.ndim == 3:
            data = np.transpose(data, (2, 1, 0))
        elif data.ndim == 2:
            data = data.T[:, :, np.newaxis]
        if lat_is_n2s is None:
            lat_is_n2s = self._lat_is_n2s
        if not lat_is_n2s:
            data = np.flip(data, axis=1)
        return np.asfortranarray(data)

    # --------------------------------------------------------
    # Convenience methods
    # --------------------------------------------------------
    def read_winds_850(self, year: int) -> Tuple[np.ndarray, np.ndarray]:
        """Read 850 hPa u,v winds in m/s. Returns (ua, va) shape (nlon, nlat, ntime)."""
        self.read_lonlat(year)  # ensure lat direction is known
        ua = self._read_3d("ua", year, level_pa=np.float32(85000.), var_type="wind")
        va = self._read_3d("va", year, level_pa=np.float32(85000.), var_type="wind")
        return ua, va

    def read_psl(self, year: int) -> np.ndarray:
        """Read sea level pressure in Pa. Shape (nlon, nlat, ntime)."""
        self.read_lonlat(year)
        return self._read_3d("psl", year, level_pa=None, var_type="pressure")

    # --------------------------------------------------------
    # Batch env read (2D field per timestep, Dataset cached)
    # --------------------------------------------------------
    def read_env_2d(self, var_key: str, year: int, lev_hpa: Optional[int],
                    it: int, ds_cache: dict) -> np.ndarray:
        """
        Read a full 2D (lon, lat) field for a single timestep.
        it = time index within the year (0-based, as built by _build_time_lookup).
        Caches Dataset handles in ds_cache to avoid repeated open/close.
        Returns (nlon, nlat) float32 array in standard orientation.
        """
        self.read_lonlat(year)
        level_pa = np.float32(lev_hpa * 100) if lev_hpa else None

        split_lev = self._split_level(level_pa)
        if self.cfg.file_layout == "cmip6":
            ds, file_it = self._esgf_resolve_timestep(var_key, year, it, ds_cache)
        elif self.cfg.file_layout == "split_monthly":
            ds, file_it = self._monthly_resolve_timestep(var_key, year, it, ds_cache, split_lev)
        else:
            path = self._nc_path(var_key, year, level_hpa=split_lev)
            cache_key = f"{var_key}_{split_lev or ''}_{year}"
            if cache_key not in ds_cache:
                ds_cache[cache_key] = nc.Dataset(str(path))
            ds = ds_cache[cache_key]
            file_it = it
        if split_lev:
            level_pa = None

        var_name = _find_var(ds, var_key, getattr(self.cfg, f"var_{var_key}", None))
        var = ds.variables[var_name]
        dims = var.dimensions
        lev_name = _find_dim(ds, "lev", self.cfg.dim_lev)
        lat_is_n2s = self._lat_is_n2s_for_ds(ds)

        # Read units/scale before disabling auto scale
        units = getattr(var, "units", "")
        scale_factor = getattr(var, "scale_factor", None)
        add_offset = getattr(var, "add_offset", None)
        fill_val = getattr(var, "_FillValue", None)
        var.set_auto_maskandscale(False)

        # Determine level index (cached in dict)
        lev_idx = None
        if level_pa is not None and lev_name and lev_name in dims:
            lev_cache_key = f"{var_key}_{year}_{lev_hpa}"
            if lev_cache_key in self._lev_idx_cache:
                lev_idx = self._lev_idx_cache[lev_cache_key]
            else:
                lev_vals = ds.variables[lev_name][:].astype(np.float32)
                if lev_vals.max() < 2000:
                    lev_vals *= np.float32(100)
                lev_idx = int(np.argmin(np.abs(lev_vals - level_pa)))
                self._lev_idx_cache[lev_cache_key] = lev_idx

        # Read 2D slice (generic slicer for any dimension order)
        slices = [slice(None)] * var.ndim
        # Time is always first dimension
        slices[0] = file_it
        if lev_idx is not None:
            lev_dim_pos = dims.index(lev_name)
            slices[lev_dim_pos] = lev_idx
        data_2d = var[tuple(slices)]

        data_2d = data_2d.squeeze()

        # Build fill mask on raw data before any conversion
        fill_mask = None
        if fill_val is not None:
            fill_mask = (data_2d == fill_val)

        # Apply scale/offset manually in float32
        data_2d = data_2d.astype(np.float32)
        if scale_factor is not None:
            data_2d = data_2d * np.float32(scale_factor) + np.float32(add_offset or 0)

        if fill_mask is not None:
            data_2d[fill_mask] = np.nan

        var_type = {"ua": "wind", "va": "wind", "ta": "temperature", "psl": "pressure"}.get(var_key)
        if units and var_type:
            data_2d = _convert_units(data_2d, units, var_type)

        # (lat, lon) → (lon, lat)
        data_2d = data_2d.T
        if not lat_is_n2s:
            data_2d = np.flip(data_2d, axis=1)

        return data_2d

    # --------------------------------------------------------
    # Regional crop (for env parameter calculation) — legacy
    # --------------------------------------------------------
    def read_env_region(self, var_key: str, year: int, lev_hpa: Optional[int],
                        ix: int, iy: int, it: int, dnx: int, dny: int) -> np.ndarray:
        """
        Read a cropped (2*dnx+1, 2*dny+1) region around grid point (ix, iy)
        at time index `it`. Handles periodic longitude wrapping.
        """
        self.read_lonlat(year)
        level_pa = np.float32(lev_hpa * 100) if lev_hpa else None
        path = self._nc_path(var_key, year)

        ds = nc.Dataset(str(path))
        var_name = _find_var(ds, var_key, getattr(self.cfg, f"var_{var_key}", None))
        var = ds.variables[var_name]
        dims = var.dimensions
        lev_name = _find_dim(ds, "lev", self.cfg.dim_lev)
        lat_is_n2s = self._lat_is_n2s_for_ds(ds)

        # Read metadata before disabling auto scale
        units = getattr(var, "units", "")
        scale_factor = getattr(var, "scale_factor", None)
        add_offset = getattr(var, "add_offset", None)
        fill_val = getattr(var, "_FillValue", None)
        var.set_auto_maskandscale(False)

        nlon = len(self._lon)

        # Determine level index
        lev_idx = None
        if level_pa is not None and lev_name and lev_name in dims:
            lev_vals = ds.variables[lev_name][:].astype(np.float32)
            if lev_vals.max() < 2000:
                lev_vals *= np.float32(100)
            lev_idx = int(np.argmin(np.abs(lev_vals - level_pa)))

        # Read 2D slice (generic slicer for any dimension order)
        slices = [slice(None)] * var.ndim
        slices[0] = it
        if lev_idx is not None:
            lev_dim_pos = dims.index(lev_name)
            slices[lev_dim_pos] = lev_idx
        data_2d = var[tuple(slices)]

        data_2d = data_2d.squeeze()

        # Build fill mask on raw data
        fill_mask = None
        if fill_val is not None:
            fill_mask = (data_2d == fill_val)

        # Apply scale/offset in float32
        data_2d = data_2d.astype(np.float32)
        if scale_factor is not None:
            data_2d = data_2d * np.float32(scale_factor) + np.float32(add_offset or 0)

        if fill_mask is not None:
            data_2d[fill_mask] = np.nan

        var_type = {"ua": "wind", "va": "wind", "ta": "temperature", "psl": "pressure"}.get(var_key)
        if units and var_type:
            data_2d = _convert_units(data_2d, units, var_type)

        ds.close()

        # data_2d is (lat, lon) → transpose to (lon, lat)
        data_2d = data_2d.T

        # Flip lat if needed
        if not lat_is_n2s:
            data_2d = np.flip(data_2d, axis=1)

        # Crop with periodic longitude
        region = np.full((2 * dnx + 1, 2 * dny + 1), np.nan, dtype=np.float32)
        for di in range(-dnx, dnx + 1):
            ii = (ix + di) % nlon
            j_start = iy - dny
            j_end = iy + dny + 1
            if j_start >= 0 and j_end <= data_2d.shape[1]:
                region[di + dnx, :] = data_2d[ii, j_start:j_end]

        return region
