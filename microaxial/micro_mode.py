"""Micro gas turbine design mode.

Two tools for small axial compressors (UAV / micro-turbojet class, roughly
0.2-2 kg/s, 40k-100k rpm):

* :func:`micro_sizing` - first-pass sizing from engine-level requirements
  (mass flow, pressure ratio, rpm) to a starting annulus and stage count.
* :func:`evaluate_design` - screening of a finished mean-line design against
  recommended ranges for small, low-Reynolds-number axial compressors.

The ranges are textbook screening heuristics (Lieblein diffusion factor,
de Haller limit, Howell/Cohen-Rogers-Saravanamuttoo stage-loading practice),
tightened where small scale makes things harder (low Reynolds number, short
blades, tip-clearance sensitivity). They flag designs worth a second look;
they are not pass/fail certification limits.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from microaxial.compressor import AxialCompressor


# --------------------------------------------------------------------------- #
# Recommended ranges
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Range:
    low: float
    high: float
    note: str

    def classify(self, value: float, margin: float = 0.10) -> str:
        """'ok' inside the range, 'warn' within `margin` (relative) outside, else 'fail'."""
        span = max(abs(self.high - self.low), 1e-12)
        if self.low <= value <= self.high:
            return "ok"
        if self.low - margin * span <= value <= self.high + margin * span:
            return "warn"
        return "fail"


MICRO_RANGES: dict[str, Range] = {
    "flow_coefficient": Range(0.45, 0.70, "phi = Cx/U. Below ~0.45 blades get highly cambered; above ~0.7 work capability drops."),
    "stage_loading": Range(0.25, 0.50, "psi = dh/U^2. Above ~0.5 de Haller / diffusion limits usually bind."),
    "stage_pressure_ratio": Range(1.10, 1.40, "Per-stage total pressure ratio. Small axials rarely exceed ~1.35 without transonic rotors."),
    "stage_temperature_rise": Range(15.0, 55.0, "Per-stage total temperature rise (K)."),
    "reaction": Range(0.45, 0.70, "Degree of reaction. 0.5 symmetric; higher unloads the stator."),
    "de_haller": Range(0.72, 1.00, "Minimum over span, rotor and stator. Below 0.72 expect separation (Rabe / de Haller criterion)."),
    "diffusion_factor": Range(0.20, 0.55, "Maximum over span (Lieblein). Stall onset typically near 0.6."),
    "rotor_tip_relative_mach": Range(0.40, 1.15, "Inlet relative Mach at rotor tip. Above ~0.9 shock loss starts; a >1.2 rotor is a transonic design."),
    "tip_speed": Range(150.0, 400.0, "Rotor tip speed (m/s). Upper bound set by disk / blade stress for small Ti or Al wheels."),
    "chord_reynolds": Range(1.0e5, 1.0e7, "Chord Reynolds number. Below ~1e5 profile loss rises steeply (laminar separation)."),
    "blade_height_mm": Range(8.0, 60.0, "Blade height (mm). Below ~8 mm tip clearance dominates losses and stall margin."),
    "hub_to_tip_ratio": Range(0.55, 0.85, "Row hub-to-tip. Small machines run high to keep blade height practical and the shaft stiff."),
    "aspect_ratio": Range(0.9, 2.5, "Height / chord. Low values suit short blades; high values risk flutter and bending."),
    "solidity": Range(1.0, 1.8, "Chord / pitch."),
    "thickness_to_chord": Range(0.04, 0.10, "Max thickness / chord. Thin sections for transonic rows, thicker for robust micro-scale manufacturing."),
}


# --------------------------------------------------------------------------- #
# Design checks
# --------------------------------------------------------------------------- #
def _row(stage: int, item: str, value: float, key: str, unit: str = "") -> dict:
    rng = MICRO_RANGES[key]
    return {
        "stage": stage,
        "parameter": item,
        "value": float(value),
        "unit": unit,
        "recommended_min": rng.low,
        "recommended_max": rng.high,
        "status": rng.classify(float(value)),
        "note": rng.note,
    }


def evaluate_design(design: AxialCompressor) -> pd.DataFrame:
    """Return a long-form table of all micro-mode checks (one row per stage and parameter)."""
    rows: list[dict] = []
    for st in design.stages:
        n = st.stage_number
        rows += [
            _row(n, "flow coefficient", st.flow_coefficient, "flow_coefficient"),
            _row(n, "stage loading", st.loading_coefficient, "stage_loading"),
            _row(n, "stage pressure ratio", st.pressure_ratio, "stage_pressure_ratio"),
            _row(n, "stage temperature rise", st.temperature_rise, "stage_temperature_rise", "K"),
            _row(n, "reaction", st.reaction, "reaction"),
            _row(n, "rotor min de Haller", st.rotor_minimum_de_haller, "de_haller"),
            _row(n, "stator min de Haller", st.stator_minimum_de_haller, "de_haller"),
            _row(n, "rotor max diffusion factor", st.rotor_maximum_diffusion_factor, "diffusion_factor"),
            _row(n, "stator max diffusion factor", st.stator_maximum_diffusion_factor, "diffusion_factor"),
            _row(n, "rotor tip relative Mach", st.rotor_tip_relative_mach_number, "rotor_tip_relative_mach"),
            _row(n, "rotor tip speed", st.tip_speed, "tip_speed", "m/s"),
            _row(n, "rotor chord Reynolds", st.rotor.chord_reynolds_number, "chord_reynolds"),
            _row(n, "stator chord Reynolds", st.stator.chord_reynolds_number, "chord_reynolds"),
            _row(n, "blade height", st.minimum_blade_height * 1e3, "blade_height_mm", "mm"),
            _row(n, "rotor hub-to-tip", st.rotor.hub_to_tip_ratio, "hub_to_tip_ratio"),
            _row(n, "rotor aspect ratio", st.rotor.aspect_ratio, "aspect_ratio"),
            _row(n, "stator aspect ratio", st.stator.aspect_ratio, "aspect_ratio"),
            _row(n, "rotor solidity", st.rotor.solidity, "solidity"),
            _row(n, "stator solidity", st.stator.solidity, "solidity"),
            _row(n, "rotor thickness/chord", st.rotor.max_thickness_to_chord, "thickness_to_chord"),
            _row(n, "stator thickness/chord", st.stator.max_thickness_to_chord, "thickness_to_chord"),
        ]
    return pd.DataFrame(rows)


def check_summary(design: AxialCompressor) -> dict[str, int]:
    """Counts of ok / warn / fail across all checks."""
    df = evaluate_design(design)
    counts = df["status"].value_counts().to_dict()
    return {k: int(counts.get(k, 0)) for k in ("ok", "warn", "fail")}


# --------------------------------------------------------------------------- #
# First-pass sizing
# --------------------------------------------------------------------------- #
@dataclass
class MicroSizing:
    """Result of :func:`micro_sizing`. All lengths in metres, speeds in m/s."""
    tip_radius: float
    hub_radius: float
    mean_radius: float
    mean_blade_speed: float
    tip_speed: float
    axial_velocity: float
    total_temperature_rise: float
    stage_temperature_rise: float
    num_stages: int
    inlet_tip_relative_mach: float
    polytropic_efficiency: float

    def as_dict(self) -> dict:
        return self.__dict__.copy()


def micro_sizing(
    mass_flow_rate: float,
    pressure_ratio: float,
    rpm: float,
    *,
    inlet_total_temperature: float = 288.15,
    inlet_total_pressure: float = 101325.0,
    gamma: float = 1.4,
    gas_constant: float = 287.0,
    polytropic_efficiency: float = 0.85,
    target_flow_coefficient: float = 0.55,
    target_stage_loading: float = 0.40,
    hub_to_tip_ratio: float = 0.68,
    inlet_blockage: float = 0.05,
) -> MicroSizing:
    """First-pass annulus and stage-count sizing for a small axial compressor.

    The mean-line is constrained by continuity with a chosen flow coefficient
    phi = Cx/U_mean, hub-to-tip ratio and rotational speed. Stage work follows from
    the loading coefficient, psi = cp*dT/U_mean^2.

    The tip radius solves  m = rho * Cx * A_flow  with  Cx = phi * w * r_mean,
    which gives a cubic in r_tip, evaluated by fixed-point iteration on rho.
    """
    cp = gas_constant * gamma / (gamma - 1.0)
    w = 2.0 * np.pi * rpm / 60.0
    ht = hub_to_tip_ratio
    r_ratio_mean = (1.0 + ht) / 2.0

    cx = 150.0
    rho = inlet_total_pressure / (gas_constant * inlet_total_temperature)
    rt = 0.05
    for _ in range(60):
        t_static = inlet_total_temperature - cx**2 / (2 * cp)
        p_static = inlet_total_pressure * (t_static / inlet_total_temperature) ** (gamma / (gamma - 1))
        rho = p_static / (gas_constant * t_static)
        a_flow = mass_flow_rate / (rho * cx)
        a_phys = a_flow * (1.0 + inlet_blockage)
        rt = np.sqrt(a_phys / (np.pi * (1.0 - ht**2)))
        cx_new = target_flow_coefficient * w * rt * r_ratio_mean
        if abs(cx_new - cx) < 1e-6:
            cx = cx_new
            break
        cx = 0.5 * cx + 0.5 * cx_new

    rm = rt * r_ratio_mean
    u_m = w * rm
    u_t = w * rt

    # polytropic efficiency -> overall temperature rise
    dt_total = inlet_total_temperature * (pressure_ratio ** ((gamma - 1) / (gamma * polytropic_efficiency)) - 1.0)
    dt_stage = target_stage_loading * u_m**2 / cp
    n_stages = int(np.ceil(dt_total / dt_stage))

    t_static = inlet_total_temperature - cx**2 / (2 * cp)
    a_in = np.sqrt(gamma * gas_constant * t_static)
    w_tip = np.sqrt(cx**2 + u_t**2)

    return MicroSizing(
        tip_radius=float(rt),
        hub_radius=float(ht * rt),
        mean_radius=float(rm),
        mean_blade_speed=float(u_m),
        tip_speed=float(u_t),
        axial_velocity=float(cx),
        total_temperature_rise=float(dt_total),
        stage_temperature_rise=float(dt_stage),
        num_stages=n_stages,
        inlet_tip_relative_mach=float(w_tip / a_in),
        polytropic_efficiency=float(polytropic_efficiency),
    )
