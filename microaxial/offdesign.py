"""Screening-level off-design compressor map and surge-margin estimation.

Method
------
A mean-line stage-stacking model built directly on the design-point velocity
triangles of an :class:`~microaxial.compressor.AxialCompressor`.

For a given corrected speed and mass flow the model marches stage by stage:

1. Continuity at the stage inlet annulus (fixed by the design geometry) gives
   the axial velocity ``Cx`` and hence the flow coefficient ``phi = Cx/U``.
2. With the stator exit angle ``alpha1`` frozen at its design value and the
   rotor relative exit angle drifting with incidence,
   ``beta2 = beta2_d + k_dev*(beta1 - beta1_d)``, Euler's equation gives the
   work coefficient ``psi = 1 - phi*(tan(alpha1) + tan(beta2))``.
3. The polytropic efficiency falls quadratically with the change in rotor
   relative inlet angle (incidence proxy):
   ``eta_p = eta_p,d * (1 - k_inc * (beta1 - beta1_d)^2)``.
4. Stage pressure ratio follows from temperature rise and ``eta_p``.

Limits on each speed line
-------------------------
* **Stall / surge side**: lowest flow at which every rotor and stator
  Lieblein diffusion factor stays at or below ``df_limit`` (frozen design
  solidity and exit angles), or the peak of the speed line if that occurs at
  higher flow.
* **Choke side**: either annulus choking (Mach 1 on ``Cx``) or rotor-throat
  choking on relative total conditions. The throat area is calibrated so the
  design point sits ``design_choke_margin`` below throat choke.

Running line and surge margin
-----------------------------
``PR_run = 1 + (PR_d - 1) * (m_c/m_d)^n`` through the design point
(default ``n = 2``, a choked downstream nozzle). Surge margin at each speed is

    SM = (PR_surge / m_surge) / (PR_run / m_run) - 1

i.e. the usual constant-speed definition on the PR/flow ratio.

Where no stall-free flow exists at a speed (front stages stalled while rear
stages choke - typical of fixed-geometry, high-pressure-ratio axials at part
speed) the speed is reported with ``stall_free_range = False``: that is the
signature of needing bleed or variable stators, not a numerical failure.

Limitations (read before trusting numbers)
------------------------------------------
This is a *screening* tool for comparing design candidates and for sanity
checking margins. It has no tip-clearance, endwall-blockage, Reynolds, or
inter-stage bleed model, uses a single mean-line per stage, and the
efficiency/incidence coefficient is a generic value. Absolute surge lines will
differ from rig data; trends with rpm, loading and solidity are the useful part.
Calibrate ``k_inc``, ``df_limit`` and ``design_choke_margin`` against test data
or 3-D CFD when available.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from microaxial.compressor import AxialCompressor


@dataclass
class OffDesignSettings:
    """Tunable assumptions of the off-design model."""
    k_inc: float = 5.0                  # efficiency loss per rad^2 of rotor incidence change
    df_limit: float = 0.60              # Lieblein stall-onset diffusion factor
    design_choke_margin: float = 0.12   # design flow sits this far below throat choke
    running_line_exponent: float = 2.0  # PR_run - 1 ~ (m_c/m_d)^n
    min_efficiency: float = 0.30        # floor on polytropic efficiency
    deviation_slope: float = 0.30       # d(beta2)/d(beta1): rotor exit-angle drift with incidence


@dataclass
class _StageData:
    area_flow: float        # inlet annulus flow area (m^2)
    mean_radius: float
    reaction: float
    phi_d: float
    psi_d: float
    eta_p_d: float
    k_euler: float          # tan(alpha1) + tan(beta2) at design = (1 - psi)/phi
    tan_alpha1: float
    tan_beta2: float        # magnitude, rotor relative exit
    beta1_d: float          # design rotor relative inlet angle magnitude (rad)
    sigma_rotor: float
    sigma_stator: float
    throat_factor: float = 1.0


class CompressorMap:
    """Off-design map generator bound to one design."""

    def __init__(self, design: AxialCompressor, settings: Optional[OffDesignSettings] = None):
        self.design = design
        self.s = settings or OffDesignSettings()
        self.gamma = design.gamma
        self.R = design.gas_constant
        self.cp = self.R * self.gamma / (self.gamma - 1.0)
        self.Tt1 = design.inlet_total_temperature
        self.Pt1 = design.inlet_total_pressure
        self.rpm_d = design.rpm
        self.m_d = design.mass_flow_rate
        self.pr_d = design.pressure_ratio
        self._stages = self._extract_stages()
        self._area_out = design.outlet_flow_station.physical_area / (1.0 + design.outlet_flow_station.blockage)
        self._calibrate_throats()

    # ------------------------------------------------------------------ #
    # Setup
    # ------------------------------------------------------------------ #
    def _extract_stages(self) -> list[_StageData]:
        out: list[_StageData] = []
        for st in self.design.stages:
            fs = st.inlet_flow_station
            phi, psi, R = st.flow_coefficient, st.loading_coefficient, st.reaction
            tan_a1 = (1.0 - R - 0.5 * psi) / phi
            tan_b2 = (R - 0.5 * psi) / phi
            beta1 = np.arctan(1.0 / phi - tan_a1)
            out.append(_StageData(
                area_flow=float(fs.physical_area / (1.0 + fs.blockage)),
                mean_radius=float(st.mean_radius), reaction=float(R),
                phi_d=float(phi), psi_d=float(psi), eta_p_d=float(st.polytropic_efficiency),
                k_euler=float((1.0 - psi) / phi), tan_alpha1=float(tan_a1), tan_beta2=float(abs(tan_b2)),
                beta1_d=float(beta1),
                sigma_rotor=float(st.rotor.solidity), sigma_stator=float(st.stator.solidity),
            ))
        return out

    def _cx_from_flow(self, m: float, area: float, Tt: float, Pt: float, tan_a: float = 0.0) -> Optional[float]:
        """Subsonic axial velocity from continuity; None if the annulus is choked."""
        g, R, cp = self.gamma, self.R, self.cp
        k = 1.0 + tan_a * tan_a  # |C|^2 = Cx^2 (1 + tan^2 alpha): stage-inlet swirl is included

        def flux(cx: float) -> float:
            t = Tt - k * cx * cx / (2 * cp)
            return Pt / (R * Tt) * (t / Tt) ** (1.0 / (g - 1.0)) * cx

        target = m / area
        cx_choke = np.sqrt(2.0 * g * R * Tt / (g + 1.0) / k)  # |C| = critical velocity (M=1)
        lo, hi = 1e-3, cx_choke
        if flux(hi) < target:
            return None
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if flux(mid) < target:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    def _throat_flow_limit(self, st: _StageData, cx: float, U: float, Tt: float, Pt: float) -> float:
        """Rotor-throat choked mass flow on relative total conditions (kg/s)."""
        g, R, cp = self.gamma, self.R, self.cp
        t = Tt - cx * cx * (1.0 + st.tan_alpha1 ** 2) / (2 * cp)
        p = Pt * (t / Tt) ** (g / (g - 1.0))
        c_theta = cx * st.tan_alpha1
        w2 = cx * cx + (U - c_theta) ** 2
        t0r = t + w2 / (2 * cp)
        p0r = p * (t0r / t) ** (g / (g - 1.0))
        f = (2.0 / (g + 1.0)) ** ((g + 1.0) / (2.0 * (g - 1.0)))
        return st.throat_factor * st.area_flow * p0r * np.sqrt(g / (R * t0r)) * f

    def _calibrate_throats(self) -> None:
        """Set each rotor's throat area so the design point is `design_choke_margin` below choke."""
        Tt, Pt = self.Tt1, self.Pt1
        U = 2 * np.pi * self.rpm_d / 60.0
        for st in self._stages:
            cx = self._cx_from_flow(self.m_d, st.area_flow, Tt, Pt, st.tan_alpha1)
            st.throat_factor = 1.0
            raw = self._throat_flow_limit(st, cx, U * st.mean_radius, Tt, Pt)
            st.throat_factor = self.m_d * (1.0 + self.s.design_choke_margin) / raw
            # advance the thermodynamic state with the design work for the next stage
            dT = st.psi_d * (U * st.mean_radius) ** 2 / self.cp
            pr = ((Tt + dT) / Tt) ** (st.eta_p_d * self.gamma / (self.gamma - 1.0))
            Tt, Pt = Tt + dT, Pt * pr

    # ------------------------------------------------------------------ #
    # Single operating point
    # ------------------------------------------------------------------ #
    def evaluate(self, speed_fraction: float, mass_flow: float) -> dict:
        """Evaluate the stage stack at one speed and mass flow (design inlet conditions).

        Returns a dict with pressure ratio, efficiency, per-stage state and limit flags.
        """
        s = self.s
        omega = 2 * np.pi * self.rpm_d * speed_fraction / 60.0
        Tt, Pt = self.Tt1, self.Pt1
        Tt_in = Tt
        work = 0.0
        stalled: list[int] = []
        choked_annulus = False
        choked_throat = False
        phis: list[float] = []
        max_df = 0.0
        for i, st in enumerate(self._stages):
            U = omega * st.mean_radius
            cx = self._cx_from_flow(mass_flow, st.area_flow, Tt, Pt, st.tan_alpha1)
            if cx is None:
                choked_annulus = True
                break
            phi = cx / U
            phis.append(phi)
            # incidence proxy: rotor relative inlet angle change
            beta1 = np.arctan(1.0 / phi - st.tan_alpha1)
            beta2 = np.arctan(st.tan_beta2) + s.deviation_slope * (beta1 - st.beta1_d)
            tan_b2 = np.tan(beta2)
            psi = 1.0 - phi * (st.tan_alpha1 + tan_b2)
            if psi <= 0:
                return {"valid": False, "pr": np.nan, "eta": np.nan, "phis": phis, "reason": "negative work"}
            eta_p = st.eta_p_d * max(1.0 - s.k_inc * (beta1 - st.beta1_d) ** 2, s.min_efficiency / st.eta_p_d)
            dT = psi * U * U / self.cp
            pr_stage = ((Tt + dT) / Tt) ** (eta_p * self.gamma / (self.gamma - 1.0))
            # diffusion factors with frozen design solidity and exit angles
            b1, b2 = beta1, beta2
            df_r = 1 - np.cos(b1) / np.cos(b2) + np.cos(b1) / 2 * st.sigma_rotor * (np.tan(b1) - np.tan(b2))
            a2 = np.arctan(1.0 / phi - tan_b2)
            a1 = np.arctan(st.tan_alpha1)
            df_s = 1 - np.cos(a2) / np.cos(a1) + np.cos(a2) / 2 * st.sigma_stator * (np.tan(a2) - np.tan(a1))
            max_df = max(max_df, df_r, df_s)
            if max(df_r, df_s) > s.df_limit:
                stalled.append(i + 1)
            if mass_flow > self._throat_flow_limit(st, cx, U, Tt, Pt):
                choked_throat = True
            Tt, Pt = Tt + dT, Pt * pr_stage
            work += dT
        else:
            pr = Pt / self.Pt1
            eta_is = (pr ** ((self.gamma - 1) / self.gamma) - 1.0) / (Tt / Tt_in - 1.0) if Tt > Tt_in else np.nan
            return {
                "valid": True, "pr": float(pr), "eta_isentropic": float(eta_is), "tt_ratio": float(Tt / Tt_in),
                "phis": phis, "stalled_stages": stalled, "choked": choked_throat, "max_df": float(max_df),
            }
        return {"valid": False, "pr": np.nan, "choked": True, "reason": "annulus choke", "phis": phis}

    # ------------------------------------------------------------------ #
    # Speed lines and map
    # ------------------------------------------------------------------ #
    def speed_line(self, speed_fraction: float, n_points: int = 160,
                   flow_range: tuple[float, float] = (0.35, 1.45)) -> pd.DataFrame:
        """Evaluate one speed line; adds `surge` / `choke` flags and the limiting boundaries."""
        flows = np.linspace(flow_range[0], flow_range[1], n_points) * self.m_d * speed_fraction
        rows = []
        for m in flows:
            r = self.evaluate(speed_fraction, float(m))
            rows.append({
                "speed_fraction": speed_fraction, "corrected_flow": float(m),
                "pressure_ratio": r.get("pr", np.nan), "isentropic_efficiency": r.get("eta_isentropic", np.nan),
                "valid": r["valid"], "stalled": bool(r.get("stalled_stages")) if r["valid"] else False,
                "stalled_stages": ",".join(map(str, r.get("stalled_stages", []))) if r["valid"] else "",
                "choked": bool(r.get("choked", False)),
            })
        df = pd.DataFrame(rows)
        return df

    @staticmethod
    def _limits(line: pd.DataFrame) -> tuple[Optional[int], Optional[int]]:
        """Index of the surge point and the choke point on a speed line."""
        ok = line["valid"] & ~line["choked"]
        if not ok.any():
            return None, None
        usable = line[ok]
        # choke = highest usable flow
        i_choke = int(usable.index[-1])
        # stall-free region: flows above the last stalled usable point
        not_stalled = usable[~usable["stalled"]]
        if not_stalled.empty:
            return None, i_choke
        first_ok = int(not_stalled.index[0])
        # a stalled point can sit above first_ok (non-monotone); take the highest stalled index below choke
        stalled_idx = usable[usable["stalled"]].index
        if len(stalled_idx):
            first_ok = max(first_ok, int(stalled_idx.max()) + 1) if int(stalled_idx.max()) < i_choke else first_ok
        first_ok = min(first_ok, i_choke)
        # speed-line peak (positive slope) may bind before the DF limit
        seg = line.loc[first_ok:i_choke]
        pr = seg["pressure_ratio"].to_numpy()
        i_peak = int(seg.index[int(np.nanargmax(pr))])
        return max(first_ok, i_peak), i_choke

    def build(self, speed_fractions: Sequence[float] = (0.60, 0.70, 0.80, 0.90, 1.00, 1.05),
              n_points: int = 160) -> "MapResult":
        """Compute speed lines, surge/choke boundaries, running line and surge margin."""
        lines: list[pd.DataFrame] = []
        rows = []
        n = self.s.running_line_exponent
        for N in speed_fractions:
            line = self.speed_line(N, n_points=n_points)
            i_surge, i_choke = self._limits(line)
            lines.append(line)
            row = {"speed_fraction": N, "corrected_speed_rpm": N * self.rpm_d}
            row["stall_free_range"] = i_surge is not None
            if i_surge is None:
                rows.append(row)
                continue
            surge = line.loc[i_surge]
            row.update({"surge_flow": surge["corrected_flow"], "surge_pr": surge["pressure_ratio"]})
            if i_choke is not None:
                ch = line.loc[i_choke]
                row.update({"choke_flow": ch["corrected_flow"], "choke_pr": ch["pressure_ratio"]})
            # operating point: intersection of speed line with running line (between surge and choke)
            seg = line.loc[i_surge:i_choke] if i_choke is not None else line.loc[i_surge:]
            m = seg["corrected_flow"].to_numpy()
            pr = seg["pressure_ratio"].to_numpy()
            run = 1.0 + (self.pr_d - 1.0) * (m / self.m_d) ** n
            diff = pr - run
            idx = np.where(np.sign(diff[:-1]) * np.sign(diff[1:]) < 0)[0]
            if len(idx):
                k = int(idx[-1])
                t = diff[k] / (diff[k] - diff[k + 1])
                m_op = m[k] + t * (m[k + 1] - m[k])
                pr_op = pr[k] + t * (pr[k + 1] - pr[k])
                row.update({
                    "run_flow": float(m_op), "run_pr": float(pr_op),
                    "surge_margin_pct": float(((row["surge_pr"] / row["surge_flow"]) / (pr_op / m_op) - 1.0) * 100.0),
                })
            rows.append(row)
        return MapResult(design=self.design, lines=pd.concat(lines, ignore_index=True),
                         boundaries=pd.DataFrame(rows), settings=self.s)


@dataclass
class MapResult:
    design: AxialCompressor
    lines: pd.DataFrame
    boundaries: pd.DataFrame
    settings: OffDesignSettings = field(default_factory=OffDesignSettings)

    @property
    def design_surge_margin(self) -> Optional[float]:
        """Surge margin (%) at 100 % speed, if the running line crosses the speed line."""
        b = self.boundaries
        r = b[np.isclose(b["speed_fraction"], 1.0)]
        if r.empty or "surge_margin_pct" not in r or pd.isna(r.iloc[0].get("surge_margin_pct")):
            return None
        return float(r.iloc[0]["surge_margin_pct"])

    def surge_margin_table(self) -> pd.DataFrame:
        cols = [c for c in ["speed_fraction", "corrected_speed_rpm", "stall_free_range", "surge_flow", "surge_pr",
                            "run_flow", "run_pr", "surge_margin_pct", "choke_flow"] if c in self.boundaries]
        return self.boundaries[cols].copy()

    def plot(self, path: Optional[str] = None):
        """Plotly compressor map: speed lines, surge line, running line, design point."""
        import plotly.graph_objects as go
        fig = go.Figure()
        for N, grp in self.lines.groupby("speed_fraction"):
            g = grp[grp["valid"] & ~grp["choked"]]
            b = self.boundaries[np.isclose(self.boundaries["speed_fraction"], N)]
            if not b.empty and "surge_flow" in b and not pd.isna(b.iloc[0].get("surge_flow")):
                g = g[g["corrected_flow"] >= b.iloc[0]["surge_flow"] * 0.97]
            fig.add_trace(go.Scatter(x=g["corrected_flow"], y=g["pressure_ratio"], mode="lines",
                                     name=f"{N*100:.0f}% N"))
        b = self.boundaries.dropna(subset=["surge_flow"]) if "surge_flow" in self.boundaries else self.boundaries.iloc[0:0]
        if len(b):
            fig.add_trace(go.Scatter(x=b["surge_flow"], y=b["surge_pr"], mode="lines+markers",
                                     name="Surge line (est.)", line=dict(color="red", dash="dash")))
        if "run_flow" in self.boundaries:
            r = self.boundaries.dropna(subset=["run_flow"])
            fig.add_trace(go.Scatter(x=r["run_flow"], y=r["run_pr"], mode="lines+markers",
                                     name="Running line", line=dict(color="black")))
        fig.add_trace(go.Scatter(x=[self.design.mass_flow_rate], y=[self.design.pressure_ratio], mode="markers",
                                 name="Design point", marker=dict(symbol="star", size=14, color="gold",
                                                                  line=dict(color="black", width=1))))
        fig.update_layout(title="Estimated compressor map (screening-level mean-line model)",
                          xaxis_title="Corrected mass flow (kg/s)", yaxis_title="Total pressure ratio",
                          template="plotly_white")
        if path:
            if path.endswith(".html"):
                fig.write_html(path)
            else:
                fig.write_image(path)
        return fig


def estimate_map(design: AxialCompressor, settings: Optional[OffDesignSettings] = None,
                 speed_fractions: Sequence[float] = (0.60, 0.70, 0.80, 0.90, 1.00, 1.05)) -> MapResult:
    """Convenience wrapper: build a :class:`MapResult` for a design."""
    return CompressorMap(design, settings).build(speed_fractions)
