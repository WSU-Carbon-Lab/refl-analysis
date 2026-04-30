import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from periodictable import xsf
import pint
from scipy.optimize import curve_fit

ureg = pint.UnitRegistry()


class NexafsDirectory:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"Path {self.path} does not exist")
        self.izero_store: dict[pd.Timestamp, dict] = {}
        self._selected_izero_timestamp: pd.Timestamp | None = None
        self._last_sample_context: dict | None = None
        self._scan_and_process_izero()
        self.dataset = None
        self.pre_edge = (None, 280)
        self.post_edge = (360, None)

    def set_pre(self, pre_edge):
        if isinstance(pre_edge, (int, float)):
            self.pre_edge = (None, pre_edge)
        elif isinstance(pre_edge, (tuple, list)) and len(pre_edge) == 2:
            self.pre_edge = (pre_edge[0], pre_edge[1])
        else:
            raise ValueError("pre_edge must be a float, int, or length-2 (min, max) tuple/list.")

    def set_post(self, post_edge):
        if isinstance(post_edge, (int, float)):
            self.post_edge = (post_edge, None)
        elif isinstance(post_edge, (tuple, list)) and len(post_edge) == 2:
            self.post_edge = (post_edge[0], post_edge[1])
        else:
            raise ValueError("post_edge must be a float, int, or length-2 (min, max) tuple/list.")

    @staticmethod
    def read_nexafs(file_path: Path) -> pd.DataFrame:
        with open(file_path) as f:
            lines = f.readlines()
        header_idx = next(
            (i for i, line in enumerate(lines) if "Time of Day" in line), None
        )
        if header_idx is None:
            raise ValueError(f"Could not find 'Time of Day' header in {file_path}")
        from io import StringIO
        table_text = "".join(lines[header_idx:])
        df = pd.read_csv(StringIO(table_text), sep=r"\t", engine="python")
        df["Beamline Energy"] = df["Beamline Energy"].round(1)
        return df

    @staticmethod
    def compute_izero(izer_df: pd.DataFrame) -> pd.DataFrame:
        df = pd.DataFrame()
        df["Energy"] = izer_df["Beamline Energy"]
        df["Photodiode"] = izer_df["Photodiode"]
        df["AI 3 Izero"] = izer_df["AI 3 Izero"]
        df["I0"] = izer_df["AI 3 Izero"] / izer_df["Photodiode"]
        if "Timestamp" in izer_df.columns:
            df["Timestamp"] = izer_df["Timestamp"]
        return df

    def _scan_and_process_izero(self):
        izero_files = list(self.path.glob("izero*"))
        new_store: dict[pd.Timestamp, dict] = {}
        for file in izero_files:
            raw_df = self.read_nexafs(file)
            processed = self.compute_izero(raw_df)
            if "Timestamp" in processed.columns:
                timestamp = pd.to_datetime(processed["Timestamp"].iloc[0])
            else:
                timestamp = pd.to_datetime(file.stat().st_mtime, unit="s")
            energy = processed["Energy"].astype(float)
            epu = None
            if "EPU Polarization" in raw_df.columns:
                epu_val = raw_df["EPU Polarization"].iloc[0]
                if pd.notna(epu_val):
                    epu = float(epu_val)
            new_store[timestamp] = {
                "df": processed,
                "energy_min": float(energy.min()),
                "energy_max": float(energy.max()),
                "epu_polarization": epu,
            }
        self.izero_store = new_store

    def plot_izero(self, ai_to_plot="I0"):
        self._scan_and_process_izero()
        if not self.izero_store:
            print("No izero scans found.")
            return
        fig, ax = plt.subplots()
        for timestamp, entry in sorted(self.izero_store.items()):
            label = timestamp.strftime("%Y-%m-%d %H:%M:%S")
            entry["df"].plot(x="Energy", y=ai_to_plot, ax=ax, label=label)
        ax.set_xlabel("Energy")
        ax.set_ylabel(ai_to_plot)
        ax.set_title("Izero Scans")
        ax.legend()
        ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
        return ax

    def get_izero_by_timestamp(self, timestamp: pd.Timestamp) -> pd.DataFrame | None:
        self._scan_and_process_izero()
        entry = self.izero_store.get(timestamp)
        return entry["df"] if entry else None

    def list_izero_timestamps(self):
        self._scan_and_process_izero()
        return sorted(self.izero_store.keys())

    def _select_izero_nearest(
        self,
        sample_timestamp: pd.Timestamp,
        energy_range: tuple[float, float] | None = None,
        epu_polarization: float | None = None,
    ) -> pd.Timestamp | None:
        if not self.izero_store:
            return None
        candidates = []
        sample_min, sample_max = (energy_range or (0, np.inf))
        for ts, entry in self.izero_store.items():
            e_min, e_max = entry["energy_min"], entry["energy_max"]
            if energy_range and (sample_max < e_min or sample_min > e_max):
                continue
            epu = entry.get("epu_polarization")
            if epu_polarization is not None and epu is not None:
                if not np.isclose(epu_polarization, epu, rtol=1e-5, atol=1e-5):
                    continue
            dt = abs((ts - sample_timestamp).total_seconds())
            candidates.append((dt, ts))
        if not candidates:
            candidates = [(abs((ts - sample_timestamp).total_seconds()), ts) for ts in self.izero_store]
        candidates.sort(key=lambda x: x[0])
        return candidates[0][1] if candidates else None

    @property
    def izero(self) -> pd.DataFrame | None:
        self._scan_and_process_izero()
        if not self.izero_store:
            return None
        if self._selected_izero_timestamp is not None:
            entry = self.izero_store.get(self._selected_izero_timestamp)
            return entry["df"] if entry else None
        if self._last_sample_context:
            ts = self._select_izero_nearest(
                sample_timestamp=self._last_sample_context["timestamp"],
                energy_range=self._last_sample_context.get("energy_range"),
                epu_polarization=self._last_sample_context.get("epu_polarization"),
            )
            if ts is not None:
                return self.izero_store[ts]["df"]
        latest_ts = max(self.izero_store)
        return self.izero_store[latest_ts]["df"]

    def set_izero_by_timestamp(self, timestamp: pd.Timestamp | None) -> None:
        self._scan_and_process_izero()
        if timestamp is None:
            self._selected_izero_timestamp = None
        else:
            if timestamp not in self.izero_store:
                raise ValueError(f"Timestamp {timestamp} not found in available izero scans.")
            self._selected_izero_timestamp = timestamp

    def set_izero_nearest(
        self,
        sample_timestamp: pd.Timestamp,
        energy_range: tuple[float, float] | None = None,
        epu_polarization: float | None = None,
    ) -> pd.Timestamp | None:
        self._scan_and_process_izero()
        ts = self._select_izero_nearest(
            sample_timestamp=sample_timestamp,
            energy_range=energy_range,
            epu_polarization=epu_polarization,
        )
        if ts is not None:
            self._selected_izero_timestamp = ts
        return ts

    def list_izero_options(self) -> pd.DataFrame:
        self._scan_and_process_izero()
        rows = []
        for ts in sorted(self.izero_store):
            entry = self.izero_store[ts]
            rows.append({
                "timestamp": ts,
                "energy_min": entry["energy_min"],
                "energy_max": entry["energy_max"],
                "epu_polarization": entry.get("epu_polarization"),
            })
        return pd.DataFrame(rows)

    def list_samples(self):
        sample_names = set()
        for file in self.path.glob("*.txt"):
            if file.name.startswith("izero"):
                continue
            split = file.stem.split("_")
            sample = "_".join(split[:-2])
            sample_names.add(sample)
        return sorted(sample_names)

    def get_sample_file_info(self, sample_name):
        files = []
        for file in self.path.glob("*.txt"):
            if file.name.startswith("izero"):
                continue
            split = file.stem.split("_")
            if "_".join(split[:-2]) != sample_name:
                continue
            try:
                angle = float(split[-2].strip("deg"))
                experiment_hash = int(split[-1])
                files.append({"file": file, "angle": angle, "experiment": experiment_hash})
            except Exception:
                continue
        return files

    def process_tey(self, df, izero_df):
        processed = pd.DataFrame()
        processed["Energy"] = df["Beamline Energy"]
        processed["Intensity"] = df["AI 3 Izero"] / df["TEY signal"]
        izero_df = izero_df.copy()
        izero_df["Energy"] = izero_df["Energy"].astype(float)
        processed = processed.merge(izero_df, on="Energy", how="left")
        processed["PD Corrected"] = processed["I0"] / processed["Intensity"]
        return processed

    @staticmethod
    def mu(energy: np.ndarray, formula: str, units: str = "g/cm^2") -> np.ndarray:
        """
        Calculate the mass attenuation coefficient μ(E) for a formula at given energies.

        Parameters
        ----------
        energy : np.ndarray
            Energy in eV.
        formula : str
            Chemical formula or element symbol, e.g., 'C8H8', 'Si', 'O'.
        units : str
            Output units, default = "g/cm^2".

        Returns
        -------
        np.ndarray
            μ in the requested units.
        """
        energy = np.asarray(energy)
        energy_kev = energy * 1e-3
        n = xsf.index_of_refraction(formula, energy=energy_kev, density=1.0)
        beta = -n.imag
        hc = 12.3984193
        wavelength_A = hc / energy_kev
        wavelength_cm = wavelength_A * 1e-8
        mu_over_rho = 4 * np.pi * beta / wavelength_cm

        mu_qty = mu_over_rho * ureg("g/cm^2")
        mu_converted = mu_qty.to(units)

        return mu_converted.magnitude

    @staticmethod
    def build_mu_arrays(energy: np.ndarray, formula: str, units: str = "g/cm^2") -> dict[str, np.ndarray]:
        """
        Precompute mu arrays for chemical formula, Si, and O at given energies.

        Parameters
        ----------
        energy : np.ndarray
            Energy in eV.
        formula : str
            Chemical formula for the main species, e.g., 'C8H8'.
        units : str
            Output units, default = "g/cm^2".

        Returns
        -------
        dict[str, np.ndarray]
            Keys: "chemical", "Si", "O". Values: mu arrays.
        """
        return {
            "chemical": NexafsDirectory.mu(energy, formula, units),
            "Si": NexafsDirectory.mu(energy, "Si", units),
            "O": NexafsDirectory.mu(energy, "O", units),
        }

    def _parse_edge_region(self, df, region, edge_type="pre"):
        energy = df["Energy"]
        if edge_type == "pre":
            if region[0] is not None and region[1] is not None:
                mask = (energy >= region[0]) & (energy <= region[1])
            elif region[0] is None and region[1] is not None:
                mask = energy < region[1]
            elif region[0] is not None and region[1] is None:
                mask = energy >= region[0]
            else:
                raise ValueError("At least one endpoint for pre_edge must be specified.")
        elif edge_type == "post":
            if region[0] is not None and region[1] is not None:
                mask = (energy >= region[0]) & (energy <= region[1])
            elif region[0] is not None and region[1] is None:
                mask = energy > region[0]
            elif region[0] is None and region[1] is not None:
                mask = energy <= region[1]
            else:
                raise ValueError("At least one endpoint for post_edge must be specified.")
        else:
            raise ValueError("edge_type must be 'pre' or 'post'")
        return df[mask]

    def normalize_tey(self, df, formula: str, pre_edge=None, post_edge=None):
        df = df.copy()
        df["Bare Atom"] = self.mu(df["Energy"], formula)
        pre = pre_edge if pre_edge is not None else self.pre_edge
        post = post_edge if post_edge is not None else self.post_edge
        if isinstance(pre, (int, float)):
            pre = (None, pre)
        pre = tuple(pre)
        if isinstance(post, (int, float)):
            post = (post, None)
        post = tuple(post)
        df_pre = self._parse_edge_region(df, pre, edge_type="pre")
        xpre = df_pre["Energy"].to_numpy()
        ypre = df_pre["PD Corrected"].to_numpy()
        barepre = df_pre["Bare Atom"].to_numpy()
        if len(xpre) < 2:
            raise ValueError("Pre-edge region too small for line fit")
        coef, intercept = np.polyfit(xpre, ypre, 1)
        coef_bare, intercept_bare = np.polyfit(xpre, barepre, 1)
        baseline = coef * df["Energy"] + intercept
        baseline_bare = coef_bare * df["Energy"] + intercept_bare
        df["Norm Abs"] = df["PD Corrected"] - baseline
        df["Bare Atom Step"] = df["Bare Atom"] - baseline_bare
        df_post = self._parse_edge_region(df, post, edge_type="post")
        if len(df_post) < 2:
            df["Mass Abs."] = df["Norm Abs"]
            return df
        post_normabs = df_post["Norm Abs"].median()
        pre_normabs = df_pre["Norm Abs"].median()
        edge_jump_normabs = post_normabs - pre_normabs
        post_bare = df_post["Bare Atom Step"].median()
        pre_bare = df_pre["Bare Atom Step"].median()
        edge_jump_bare = post_bare - pre_bare
        scale = edge_jump_bare / edge_jump_normabs if not np.isclose(edge_jump_normabs, 0) else 1.0
        df["Mass Abs."] = df["Norm Abs"] * scale
        return df

    def normalize_sample_angles(self, dfs, formula: str, pre_edge=None, post_edge=None):
        for df in dfs:
            df["Bare Atom"] = self.mu(df["Energy"], formula)
        pre = pre_edge if pre_edge is not None else self.pre_edge
        post = post_edge if post_edge is not None else self.post_edge
        if isinstance(pre, (int, float)):
            pre = (None, pre)
        pre = tuple(pre)
        if isinstance(post, (int, float)):
            post = (post, None)
        post = tuple(post)
        post_region_len = [len(self._parse_edge_region(df, post, edge_type="post")) for df in dfs]
        min_post_points = 5
        complete_mask = np.array([n >= min_post_points for n in post_region_len])
        complete_dfs = [df for df, c in zip(dfs, complete_mask) if c]
        if len(complete_dfs) == 0:
            all_pre = pd.concat([
                self._parse_edge_region(df, pre, edge_type="pre").assign(idx=i)
                for i, df in enumerate(dfs)
            ])
            xpre = all_pre["Energy"].to_numpy()
            ypre = all_pre["PD Corrected"].to_numpy()
            barepre = all_pre["Bare Atom"].to_numpy()
            if len(xpre) >= 2:
                coef, intercept = np.polyfit(xpre, ypre, 1)
                coef_bare, intercept_bare = np.polyfit(xpre, barepre, 1)
                for df in dfs:
                    baseline = coef * df["Energy"] + intercept
                    baseline_bare = coef_bare * df["Energy"] + intercept_bare
                    df["Norm Abs"] = df["PD Corrected"] - baseline
                    df["Bare Atom Step"] = df["Bare Atom"] - baseline_bare
            for df in dfs:
                df["Mass Abs."] = df.get("Norm Abs", df["PD Corrected"].copy())
            return dfs
        all_pre = pd.concat([
            self._parse_edge_region(df, pre, edge_type="pre").assign(idx=i)
            for i, df in enumerate(complete_dfs)
        ])
        xpre = all_pre["Energy"].to_numpy()
        ypre = all_pre["PD Corrected"].to_numpy()
        barepre = all_pre["Bare Atom"].to_numpy()
        if len(xpre) < 2:
            raise ValueError("Pre-edge region too small for line fit")
        coef, intercept = np.polyfit(xpre, ypre, 1)
        coef_bare, intercept_bare = np.polyfit(xpre, barepre, 1)
        for df in dfs:
            baseline = coef * df["Energy"] + intercept
            baseline_bare = coef_bare * df["Energy"] + intercept_bare
            df["Norm Abs"] = df["PD Corrected"] - baseline
            df["Bare Atom Step"] = df["Bare Atom"] #- baseline_bare
        pre_norm = pd.concat([
            self._parse_edge_region(df, pre, edge_type="pre")[["Norm Abs", "Bare Atom Step"]]
            for df in complete_dfs
        ])
        post_norm = pd.concat([
            self._parse_edge_region(df, post, edge_type="post")[["Norm Abs", "Bare Atom Step"]]
            for df in complete_dfs
        ])
        pre_median_normabs = pre_norm["Norm Abs"].median()
        post_median_normabs = post_norm["Norm Abs"].median()
        edge_jump_normabs = post_median_normabs - pre_median_normabs
        pre_median_bare = pre_norm["Bare Atom Step"].median()
        post_median_bare = post_norm["Bare Atom Step"].median()
        edge_jump_bare = post_median_bare - pre_median_bare
        scale = edge_jump_bare / edge_jump_normabs if not np.isclose(edge_jump_normabs, 0) else 1.0
        E0 = pre[1] if pre[1] is not None else pre[0]
        post_first = self._parse_edge_region(complete_dfs[0], post, edge_type="post")
        k = 0.0
        if len(post_first) >= 2:
            E_post = post_first["Energy"].to_numpy()
            slope_bare = np.polyfit(E_post, post_first["Bare Atom Step"].to_numpy(), 1)[0]
            slope_massabs = np.polyfit(E_post, (post_first["Norm Abs"].to_numpy() * scale), 1)[0]
            k = slope_bare - slope_massabs
        for i, df in enumerate(dfs):
            if complete_mask[i]:
                df["Mass Abs."] = df["Norm Abs"] * scale
                if k != 0.0:
                    mask = df["Energy"] > E0
                    df.loc[mask, "Mass Abs."] += (df.loc[mask, "Energy"] - E0) * k
            else:
                df["Mass Abs."] = df["Norm Abs"].copy()
        return dfs

    def process_sample(
        self,
        sample_name: str | None = None,
        formula: str = "C8H8",
        pre_edge: float | tuple[float | None, float | None] = 284.0,
        post_edge: float | tuple[float | None, float | None] = 320.0,
        mode: str = "si_only",
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Full pipeline: load, TEY absorption (I0/izeros), Si background fit.
        Result agrees with bare atom in pre/post edge regions (user-configurable bounds).
        Returns (concatenated DataFrame, summary DataFrame).
        """
        samples = [sample_name] if sample_name else self.list_samples()
        if not samples:
            return pd.DataFrame(), pd.DataFrame()
        all_dfs = []
        all_summaries = []
        for s in samples:
            dfs = self.get_sample_dfs(s, formula=formula, pre_edge=pre_edge, post_edge=post_edge)
            dfs, summary = self.fit_background_subtraction(
                dfs, formula=formula, pre_edge=pre_edge, post_edge=post_edge, mode=mode
            )
            all_dfs.extend(dfs)
            all_summaries.append(summary)
        combined = pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()
        summary_combined = pd.concat(all_summaries, ignore_index=True) if all_summaries else pd.DataFrame()
        return combined, summary_combined

    def get_sample_dfs(self, sample_name, formula: str = "C8H8", pre_edge=None, post_edge=None):
        files = self.get_sample_file_info(sample_name)
        if files:
            first_df = self.read_nexafs(files[0]["file"])
            energy = first_df["Beamline Energy"].astype(float)
            epu = None
            if "EPU Polarization" in first_df.columns:
                epu_val = first_df["EPU Polarization"].iloc[0]
                if pd.notna(epu_val):
                    epu = float(epu_val)
            if "Timestamp" in first_df.columns:
                sample_ts = pd.to_datetime(first_df["Timestamp"].iloc[0])
            else:
                sample_ts = pd.to_datetime(files[0]["file"].stat().st_mtime, unit="s")
            self._last_sample_context = {
                "timestamp": sample_ts,
                "energy_range": (float(energy.min()), float(energy.max())),
                "epu_polarization": epu,
            }
        izero = self.izero
        if izero is None:
            raise RuntimeError("No izero scan available.")
        df_list = []
        for info in files:
            df = self.read_nexafs(info["file"])
            processed = self.process_tey(df, izero)
            processed["Sample"] = sample_name
            processed["Angle"] = info["angle"]
            processed["Experiment"] = info["experiment"]
            df_list.append(processed)
        return self.normalize_sample_angles(df_list, formula, pre_edge, post_edge)

    def process_all(self, formula: str = "C8H8", pre_edge=None, post_edge=None):
        dataset_records = []
        for sample_name in self.list_samples():
            dfs = self.get_sample_dfs(sample_name, formula=formula, pre_edge=pre_edge, post_edge=post_edge)
            for df in dfs:
                record = df.copy()
                record["Sample"] = sample_name
                record.setdefault("Angle", None)
                record.setdefault("Experiment", None)
                dataset_records.append(record)
        self.dataset = pd.concat(dataset_records, ignore_index=True)
        return self.dataset

    def view_dataset(self) -> pd.DataFrame | None:
        if self.dataset is None:
            print("Dataset not yet processed. Run process_all() first.")
            return None
        return self.dataset

    def plot_sample(
        self,
        sample_name,
        ycol="Mass Abs.",
        formula: str = "C8H8",
        pre_edge=None,
        post_edge=None,
        ax=None,
        cmap="viridis",
        show_bare_atom=False,
        bare_atom_kwargs=None,
        apply_si_subtraction: bool = False,
        normalization_mode: str = "si_only",
    ):
        dfs = self.get_sample_dfs(sample_name, formula=formula, pre_edge=pre_edge, post_edge=post_edge)
        if apply_si_subtraction:
            pre = pre_edge if pre_edge is not None else self.pre_edge
            post = post_edge if post_edge is not None else self.post_edge
            pre_val = float(pre) if isinstance(pre, (int, float)) else (pre[1] if pre[1] is not None else 284.0)
            post_val = float(post) if isinstance(post, (int, float)) else (post[0] if post[0] is not None else 320.0)
            dfs, _ = self.fit_background_subtraction(
                dfs, formula=formula, pre_edge=pre_val, post_edge=post_val, mode=normalization_mode
            )
            if ycol == "Mass Abs.":
                ycol = "Si-subtracted"
        if ax is None:
            fig, ax = plt.subplots()
        angles = [df["Angle"].iloc[0] for df in dfs]
        norm = plt.Normalize(min(angles), max(angles))
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        if show_bare_atom:
            plot_kwargs = {"color": "black", "lw": 1}
            if bare_atom_kwargs is not None:
                plot_kwargs.update(bare_atom_kwargs)
            ax.plot(dfs[0]["Energy"], dfs[0]["Bare Atom Step"], **plot_kwargs)
        for df in dfs:
            angle = df["Angle"].iloc[0]
            color = sm.to_rgba(angle)
            ax.plot(df["Energy"], df[ycol], label=f"{angle:.1f} deg", color=color)
        ax.set_xlabel("Energy")
        ax.set_ylabel(ycol)
        ax.set_title(f"Sample: {sample_name}")
        ax.legend(title="Angle", handlelength=0.5, fontsize=10, ncol=2, frameon=True, fancybox=False, framealpha=1)
        plt.colorbar(sm, ax=ax, pad=0.02).set_label("Angle (deg)")
        ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
        return ax

    def normalization_quality(
        self,
        dfs,
        pre_edge=None,
        post_edge=None,
        ref_col="Bare Atom Step",
        ycol="Mass Abs.",
        scale_ref_percent=None,
    ):
        pre = pre_edge if pre_edge is not None else self.pre_edge
        post = post_edge if post_edge is not None else self.post_edge
        if isinstance(pre, (int, float)):
            pre = (None, pre)
        pre = tuple(pre)
        if isinstance(post, (int, float)):
            post = (post, None)
        post = tuple(post)
        all_resid_pre = []
        all_resid_post = []
        for df in dfs:
            r = df[ycol] - df[ref_col]
            pre_df = self._parse_edge_region(df, pre, edge_type="pre")
            post_df = self._parse_edge_region(df, post, edge_type="post")
            if len(pre_df):
                all_resid_pre.append(r.loc[pre_df.index])
            if len(post_df):
                all_resid_post.append(r.loc[post_df.index])
        concat_pre = pd.concat(all_resid_pre) if all_resid_pre else pd.Series(dtype=float)
        concat_post = pd.concat(all_resid_post) if all_resid_post else pd.Series(dtype=float)
        rms_pre = float(np.sqrt(np.mean(concat_pre ** 2))) if len(concat_pre) else np.nan
        rms_post = float(np.sqrt(np.mean(concat_post ** 2))) if len(concat_post) else np.nan
        if scale_ref_percent is None and len(dfs) > 0:
            post_ref = pd.concat([
                self._parse_edge_region(df, post, edge_type="post")[ref_col] for df in dfs
            ])
            scale_ref_percent = float(np.median(np.abs(post_ref))) if len(post_ref) else 1.0
        rms_pct_pre = 100.0 * rms_pre / scale_ref_percent if scale_ref_percent else np.nan
        rms_pct_post = 100.0 * rms_post / scale_ref_percent if scale_ref_percent else np.nan
        return {
            "rms_pre": rms_pre,
            "rms_post": rms_post,
            "rms_pct_pre": rms_pct_pre,
            "rms_pct_post": rms_pct_post,
            "residuals_pre": all_resid_pre,
            "residuals_post": all_resid_post,
            "scale_ref": scale_ref_percent,
        }

    def _resolve_edge_region(self, edge_val, default_upper: float | None, default_lower: float | None, edge_type: str) -> tuple[float | None, float | None]:
        if edge_val is None:
            return (None, default_upper) if edge_type == "pre" else (default_lower, None)
        if isinstance(edge_val, (int, float)):
            return (None, float(edge_val)) if edge_type == "pre" else (float(edge_val), None)
        if isinstance(edge_val, (tuple, list)) and len(edge_val) == 2:
            return (edge_val[0] if edge_val[0] is not None else None, edge_val[1] if edge_val[1] is not None else None)
        raise ValueError(f"{edge_type}_edge must be float or (start, stop) tuple")

    def fit_background_subtraction(
        self,
        dfs: pd.DataFrame | list[pd.DataFrame],
        formula: str = "C8H8",
        pre_edge: float | tuple[float | None, float | None] = 284.0,
        post_edge: float | tuple[float | None, float | None] = 320.0,
        mode: str = "si_only",
    ) -> tuple[list[pd.DataFrame], pd.DataFrame]:
        if isinstance(dfs, pd.DataFrame):
            dfs = [dfs]
        pre_region = self._resolve_edge_region(pre_edge, 284.0, None, "pre")
        post_region = self._resolve_edge_region(post_edge, None, 320.0, "post")
        energy_all = np.concatenate([df["Energy"].values for df in dfs])
        energy_grid = np.linspace(energy_all.min(), energy_all.max(), 1000)
        mu_arrays = self.build_mu_arrays(energy_grid, formula)
        summary_rows = []
        result_dfs = []
        for df in dfs:
            df = df.copy()
            pre_df = self._parse_edge_region(df, pre_region, edge_type="pre")
            post_df = self._parse_edge_region(df, post_region, edge_type="post")
            pre_post = df.loc[pre_df.index.union(post_df.index)]
            if len(pre_post) < 2:
                result_dfs.append(df)
                continue
            sf = 1 / (pre_post["Bare Atom"] / pre_post["PD Corrected"]).mean()

            def _background_si_only(energy, density, si_comp):
                mc = np.interp(energy, energy_grid, mu_arrays["chemical"])
                ms = np.interp(energy, energy_grid, mu_arrays["Si"])
                return (mc + ms * si_comp) * density

            def _background_si_oxygen(energy, density, si_comp, o_comp):
                mc = np.interp(energy, energy_grid, mu_arrays["chemical"])
                ms = np.interp(energy, energy_grid, mu_arrays["Si"])
                mo = np.interp(energy, energy_grid, mu_arrays["O"])
                return (mc + ms * si_comp + mo * o_comp) * density

            if mode == "si_only":
                p0 = (sf, 0.0)
                bounds = ((0, 0), (np.inf, np.inf))
                popt, pcov = curve_fit(
                    _background_si_only,
                    pre_post["Energy"].values,
                    pre_post["PD Corrected"].values,
                    p0=p0,
                    bounds=bounds,
                )
                density, si_comp = popt[0], popt[1]
                density_err, si_comp_err = np.sqrt(np.diag(pcov))
                o_comp, o_comp_err = np.nan, np.nan
            else:
                p0 = (sf, 0.0, 0.0)
                bounds = ((0, 0, 0), (np.inf, np.inf, np.inf))
                popt, pcov = curve_fit(
                    _background_si_oxygen,
                    pre_post["Energy"].values,
                    pre_post["PD Corrected"].values,
                    p0=p0,
                    bounds=bounds,
                )
                density, si_comp, o_comp = popt[0], popt[1], popt[2]
                errs = np.sqrt(np.diag(pcov))
                density_err, si_comp_err, o_comp_err = errs[0], errs[1], errs[2]
                if o_comp_err > abs(o_comp):
                    warnings.warn(
                        "Oxygen composition error bars exceed the value itself; results may be unreliable",
                        UserWarning,
                        stacklevel=2,
                    )
            df["Density Scaled"] = df["PD Corrected"] / density
            e_full = df["Energy"].values
            mu_si_full = np.interp(e_full, energy_grid, mu_arrays["Si"])
            df["Si-subtracted"] = df["Density Scaled"] - mu_si_full * si_comp
            if mode == "si_and_oxygen":
                mu_o_full = np.interp(e_full, energy_grid, mu_arrays["O"])
                df["O-subtracted"] = df["Si-subtracted"] - mu_o_full * o_comp
            sample = df["Sample"].iloc[0] if "Sample" in df.columns else ""
            angle = df["Angle"].iloc[0] if "Angle" in df.columns else np.nan
            exp = df["Experiment"].iloc[0] if "Experiment" in df.columns else np.nan
            row = {
                "Sample": sample,
                "Angle": angle,
                "Experiment": exp,
                "density": density,
                "density_err": density_err,
                "si_comp": si_comp,
                "si_comp_err": si_comp_err,
            }
            if mode == "si_and_oxygen":
                row["o_comp"] = o_comp
                row["o_comp_err"] = o_comp_err
            summary_rows.append(row)
            result_dfs.append(df)
        summary_df = pd.DataFrame(summary_rows)
        return result_dfs, summary_df
