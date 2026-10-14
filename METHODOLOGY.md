# MicroAxial methodology

This document records the equations, sign conventions and assumptions used by the
mean-line analysis, the blade-row metrics, the micro-mode checks and the off-design
estimator. Symbols: `U` blade speed, `Cx` axial velocity, `C` absolute velocity,
`W` relative velocity, `alpha` absolute flow angle, `beta` relative flow angle
(angles from axial), `R` degree of reaction, `phi` flow coefficient, `psi` stage loading.

## 1. Overall thermodynamics

* Overall isentropic efficiency `eta_c` and pressure ratio `PR` give the polytropic efficiency

      eta_p = (gamma-1) ln(PR) / (gamma ln[(eta_c + PR^((gamma-1)/gamma) - 1)/eta_c])

* Outlet total temperature: `Tt2 = Tt1 * PR^((gamma-1)/(gamma*eta_p))`.
* Stage work is the given (or equal-split) stage temperature rise `dT`; stage pressure ratio
  is `(1 + dT/Tt)^(eta_p*gamma/(gamma-1))`.
* Shaft power = `m * cp * (Tt2 - Tt1)`.

## 2. Stage velocity triangles (mean line)

With constant mean radius and constant axial velocity:

    phi = Cx / U                 (flow coefficient)
    psi = cp * dT / U^2          (stage loading, also written dh/U^2)

    tan(alpha1) = (1 - R - psi/2) / phi        stator exit / rotor inlet absolute angle
    tan(alpha2) = (1 - R + psi/2) / phi        rotor exit absolute angle
    tan(beta1)  = 1/phi - tan(alpha1)          rotor relative inlet (magnitude)
    tan(beta2)  = (R - psi/2) / phi            rotor relative exit (magnitude)

Euler work: `psi = 1 - phi * (tan(alpha1) + tan(beta2))`.

Annulus sizing follows continuity at each stage inlet with the **swirl-inclusive static state**
(`|C| = Cx / cos(alpha1)`); the inlet hub-to-tip ratio fixes the mean radius, which is held constant
through the machine. Blockage factors scale flow area to physical area.

## 3. Radial equilibrium

Free-vortex distribution `r * Ctheta = const` across `num_streams` stream tubes
(`num_streams` odd, so a mean-span stream exists). Hub-region loading and diffusion exceed the
mean-line values, which is why the maximum over span is what the checks use.

## 4. Blade-row metrics

| Quantity | Definition |
|---|---|
| Solidity | `sigma = c / s = 1 / spacing_to_chord` |
| Aspect ratio | `h / c` (height over chord) |
| Lieblein diffusion factor | `DF = 1 - cos(b1)/cos(b2) + (cos(b1)/2) * sigma * (tan(b1) - tan(b2))`, evaluated on **angle magnitudes** |
| de Haller number | rotor `W2/W1`, stator `C2/C1` (per stream; row minimum reported) |
| Relative Mach (inlet) | rotor `W1/a1`, stator `C1/a1`, with `a1` the inlet static speed of sound |
| Tip relative Mach | the above evaluated at the tip stream |
| Tip speed | `2*pi*N/60 * r_tip` |
| Chord Reynolds number | `rho * W1 * c / mu`, `mu` from Sutherland's law, `W1` the relative (rotor) or absolute (stator) inlet velocity at mean span |

Sign convention note: in the rotating frame the relative angle is negative for a compressor
rotor. DF and the angle-based limits are defined on magnitudes.

## 5. Metal angles

Inlet/outlet flow angles map to metal angles through the Johnsen-Bullock incidence/deviation
correlations (default), or are used directly (`EqualsFlowAngles`). Airfoils are generated from
a double-circular-arc (DCA) section parameterised by chord, camber, leading-edge radius, maximum
thickness and stagger.

## 6. Micro-mode ranges

`microaxial.micro_mode.MICRO_RANGES` lists the recommended window for each screened parameter and a
one-line rationale. The ranges collect common practice (de Haller 0.72, Lieblein DF stall onset
about 0.6, loading 0.25-0.5, flow coefficient 0.45-0.7, subsonic-to-lightly-transonic tip Mach) and
tighten blade height and Reynolds number for small machines. A value inside the window is `ok`; up to 10 % of
the window width outside is `warn`; beyond that `fail`. They are screening heuristics, not limits.

## 7. Off-design estimator

See the module docstring of `microaxial/offdesign.py` for the complete statement. In short: stage-stacked
mean-line, Euler work with frozen stator exit angle and rotor exit angle drifting with incidence
(`k_dev = 0.3`), quadratic incidence loss on polytropic efficiency (`k_inc = 5`), stall at DF = 0.6 on rotor
or stator, choke by annulus Mach 1 or calibrated rotor throat, running line `PR-1 ~ (m/m_d)^2`,
surge margin `(PR_s/m_s)/(PR_run/m_run) - 1`. The model reproduces the design point exactly by construction.

### Known limitations

* No tip clearance, endwall blockage growth, bleed, variable geometry, or inter-stage Reynolds effects.
* Single mean line per stage. Hub-stall and tip-leakage-driven surge are not captured.
* Efficiency and deviation coefficients are generic; calibrate with rig data or 3-D CFD where possible.
* Speed lines of a stage-stacked model are steep and the stable window is narrow; treat the
  surge-line location and margin as comparative (design A vs design B), not as predictions.
* Constant mean radius and constant axial velocity are built into the mean-line geometry.
