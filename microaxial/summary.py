"""Stage-by-stage summary tables for quick design review."""
from __future__ import annotations

import numpy as np
import pandas as pd

from microaxial.compressor import AxialCompressor
from microaxial.micro_mode import check_summary


def machine_summary(design: AxialCompressor) -> pd.Series:
    """One-column overview of the whole machine."""
    st0 = design.stages[0]
    last = design.stages[-1]
    data = {
        "Mass flow (kg/s)": design.mass_flow_rate,
        "Corrected mass flow, ISA SL (kg/s)": design.inlet_corrected_mass_flow,
        "Speed (rpm)": design.rpm,
        "Overall pressure ratio": design.pressure_ratio,
        "Overall total temperature rise (K)": design.overall_temperature_rise,
        "Isentropic efficiency": design.isentropic_efficiency,
        "Polytropic efficiency": design.polytropic_efficiency,
        "Shaft power (kW)": design.shaft_power,
        "Stages": design.num_stages,
        "Mean stage pressure ratio": design.mean_stage_pressure_ratio,
        "Inlet tip radius (mm)": design.inlet_tip_radius * 1e3,
        "Inlet hub radius (mm)": design.inlet_hub_radius * 1e3,
        "Inlet tip speed (m/s)": design.inlet_tip_speed,
        "First-rotor tip relative Mach": design.inlet_tip_relative_mach_number,
        "First-stage blade height (mm)": st0.rotor.height * 1e3,
        "Last-stage blade height (mm)": last.stator.height * 1e3,
    }
    counts = check_summary(design)
    data.update({f"Micro-mode checks: {k}": v for k, v in counts.items()})
    return pd.Series(data, name="value")


def stage_summary_df(design: AxialCompressor) -> pd.DataFrame:
    """One row per stage with the quantities a compressor designer reviews first."""
    rows = []
    cum_pr = 1.0
    for st in design.stages:
        cum_pr *= st.pressure_ratio
        r, s = st.rotor, st.stator
        rows.append({
            "stage": st.stage_number,
            "dTt (K)": st.temperature_rise,
            "Tt out (K)": st.outlet_total_temperature,
            "PR stage": st.pressure_ratio,
            "PR cumul.": cum_pr,
            "phi": st.flow_coefficient,
            "psi": st.loading_coefficient,
            "R": st.reaction,
            "U mean (m/s)": st.blade_velocity,
            "U tip (m/s)": st.tip_speed,
            "R M_rel,tip": r.tip_relative_mach_number,
            "R dH min": r.minimum_de_haller,
            "R DF max": r.maximum_diffusion_factor,
            "R Re_c": r.chord_reynolds_number,
            "R Z": r.num_blades,
            "R chord (mm)": r.chord * 1e3,
            "R h (mm)": r.height * 1e3,
            "S dH min": s.minimum_de_haller,
            "S DF max": s.maximum_diffusion_factor,
            "S Re_c": s.chord_reynolds_number,
            "S Z": s.num_blades,
            "S chord (mm)": s.chord * 1e3,
            "S h (mm)": s.height * 1e3,
        })
    return pd.DataFrame(rows).set_index("stage")


def format_summary(design: AxialCompressor) -> str:
    """Printable summary: machine overview followed by the stage table."""
    with pd.option_context("display.float_format", lambda x: f"{x:.3f}", "display.width", 220,
                           "display.max_columns", 50):
        head = machine_summary(design).to_string()
        tab = stage_summary_df(design).to_string()
    return f"MACHINE\n{head}\n\nSTAGES  (R = rotor, S = stator, dH = de Haller, DF = diffusion factor)\n{tab}\n"
