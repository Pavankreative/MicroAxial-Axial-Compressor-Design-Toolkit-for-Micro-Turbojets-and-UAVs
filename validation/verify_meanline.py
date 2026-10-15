"""Independent cross-check of the MicroAxial mean-line model (verification, not experimental validation).

Re-derives the overall thermodynamics and the velocity-triangle identities with a few lines of plain NumPy,
using only the design inputs, and compares them with what the package reports for the bundled design.
Run:  python validation/verify_meanline.py
Exit code is non-zero if any check fails its tolerance.

Note: these checks re-derive the same governing equations, so they catch implementation errors (units, indexing,
sign conventions) but cannot show that the equations match real hardware. See docs in README "Validation".
"""
import sys
import numpy as np
from microaxial.cli.state import MicroAxialDesign

TOL = 1e-6
rows = []


def check(name, independent, package, tol=TOL):
    rel = abs(independent - package) / max(abs(independent), 1e-12)
    rows.append((name, independent, package, rel, rel <= tol))


d = MicroAxialDesign.from_file("microaxial/designs/micro_turbojet_default.json").definition
g, R = d.gamma, d.gas_constant
PR, eta_c, T1 = d.pressure_ratio, d.isentropic_efficiency, d.inlet_total_temperature

# 1. Overall thermodynamics from the definition of polytropic efficiency
eta_p = (g - 1) * np.log(PR) / (g * np.log((eta_c + PR ** ((g - 1) / g) - 1) / eta_c))
T2 = T1 * PR ** ((g - 1) / (g * eta_p))
check("polytropic efficiency", eta_p, d.polytropic_efficiency)
check("outlet total temperature (K)", T2, d.outlet_total_temperature)
cp = g * R / (g - 1)
check("shaft power (kW) = m cp dT", d.mass_flow_rate * cp * (T2 - T1) / 1e3, d.shaft_power, tol=2e-3)  # package reports kW

# 2. Stage chain: equal split, polytropic pressure ratio per stage, product equals overall PR
n = len(d.stages)
dT = (T2 - T1) / n
Tt, cum = T1, 1.0
for s in d.stages:
    pr = ((Tt + dT) / Tt) ** (eta_p * g / (g - 1))
    cum *= pr
    check(f"stage {s.stage_number} dTt (K)", dT, s.temperature_rise)
    check(f"stage {s.stage_number} pressure ratio", pr, s.pressure_ratio, tol=1e-4)
    Tt += dT
check("cumulative pressure ratio", cum, PR, tol=1e-4)

# 3. Velocity-triangle identities at mean line
for s in d.stages:
    U = s.blade_velocity
    c1, c2 = s.inlet_flow_station, s.mid_flow_station
    Cx = c1.meridional_velocity
    phi, psi = Cx / U, cp * s.temperature_rise / U**2
    check(f"stage {s.stage_number} flow coefficient", phi, s.flow_coefficient, tol=1e-4)
    check(f"stage {s.stage_number} loading coefficient", psi, s.loading_coefficient, tol=2e-3)
    # Euler work: U (C_theta2 - C_theta1) = cp dT
    euler = U * (c2.tangential_velocity - c1.tangential_velocity)
    check(f"stage {s.stage_number} Euler work (J/kg)", cp * s.temperature_rise, euler, tol=2e-3)
    # Closed-form angle relation  psi = 1 - phi*(tan a1 + tan b2)  with  tan b2 = (R - psi/2)/phi
    a1 = np.arctan((1 - s.reaction - psi / 2) / phi)
    b2 = np.arctan((s.reaction - psi / 2) / phi)
    check(f"stage {s.stage_number} Euler identity psi", psi, 1 - phi * (np.tan(a1) + np.tan(b2)), tol=1e-9)

# 4. Continuity at the machine inlet
c = d.inlet_flow_station
check("inlet continuity m = rho*Cx*A", d.mass_flow_rate, c.density * c.meridional_velocity * c.flow_area, tol=1e-4)

width = max(len(r[0]) for r in rows)
print(f"{'check':<{width}}  {'independent':>14}  {'package':>14}  {'rel.err':>9}  status")
for name, a, b, rel, ok in rows:
    print(f"{name:<{width}}  {a:14.6g}  {b:14.6g}  {rel:9.2e}  {'PASS' if ok else 'FAIL'}")
failed = [r for r in rows if not r[4]]
print(f"\n{len(rows) - len(failed)}/{len(rows)} checks passed")
sys.exit(1 if failed else 0)
