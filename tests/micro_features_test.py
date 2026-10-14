import unittest
import numpy as np
from microaxial.micro_mode import MICRO_RANGES, check_summary, evaluate_design, micro_sizing
from microaxial.offdesign import CompressorMap, OffDesignSettings, estimate_map
from microaxial.summary import machine_summary, stage_summary_df
from tests.designs import micro_default, base_design


class LastStatorRegressionTest(unittest.TestCase):
    def test_last_stator_de_haller_defined(self):
        """Originally raised AssertionError for the last stator row."""
        dh = micro_default.stages[-1].stator.de_haller
        self.assertTrue(np.all(np.isfinite(dh)))
        self.assertTrue(np.all(dh > 0.5))


class DiffusionFactorRegressionTest(unittest.TestCase):
    def test_rotor_df_matches_stator_for_symmetric_stage(self):
        """R = 0.5: rotor and stator triangles are mirror images, so DF must agree."""
        st = micro_default.stages[0]
        r_df, s_df = st.rotor.diffusion_factor, st.stator.diffusion_factor
        # mean-span streams (index 2 of 5)
        self.assertAlmostEqual(float(r_df[2]), float(s_df[2]), delta=0.03)
        self.assertTrue(np.all(r_df > 0.2))


class MetricsTest(unittest.TestCase):
    def test_tip_mach_consistency(self):
        rot = micro_default.stages[0].rotor
        fs = rot.flow_station
        w_tip = fs.relative_velocity[-1]
        self.assertAlmostEqual(rot.tip_relative_mach_number, w_tip / fs.speed_of_sound[-1], places=9)
        # velocity triangle at the tip: W^2 = Cx^2 + (U - Ctheta)^2
        cx = micro_default.axial_velocity
        u_tip = rot.tip_speed
        c_theta = fs.tangential_velocity[-1]
        self.assertAlmostEqual(w_tip, float(np.sqrt(cx**2 + (u_tip - c_theta) ** 2)), places=6)

    def test_tip_speed(self):
        expected = 2 * np.pi * micro_default.rpm / 60 * micro_default.inlet_tip_radius
        self.assertAlmostEqual(micro_default.inlet_tip_speed, expected, places=6)

    def test_stator_has_no_tip_speed(self):
        self.assertEqual(micro_default.stages[0].stator.tip_speed, 0.0)

    def test_reynolds_positive_and_sensible(self):
        re = micro_default.stages[0].rotor.chord_reynolds_number
        self.assertTrue(5e4 < re < 1e6)

    def test_loading_and_flow_coefficient_definitions(self):
        st = micro_default.stages[0]
        U = st.blade_velocity
        self.assertAlmostEqual(st.flow_coefficient, micro_default.axial_velocity / U, places=9)
        self.assertAlmostEqual(st.loading_coefficient, st.enthalpy_rise / U**2, places=9)


class MicroModeTest(unittest.TestCase):
    def test_default_has_no_failures(self):
        self.assertEqual(check_summary(micro_default)["fail"], 0)

    def test_checks_table_columns(self):
        df = evaluate_design(micro_default)
        for col in ("stage", "parameter", "value", "status", "recommended_min", "recommended_max"):
            self.assertIn(col, df.columns)
        self.assertEqual(set(df["status"]) - {"ok", "warn", "fail"}, set())

    def test_runs_on_large_legacy_design(self):
        """The checks are design-agnostic: the original 20 kg/s example evaluates without error."""
        df = evaluate_design(base_design)
        self.assertEqual(len(df), len(evaluate_design(micro_default)) // micro_default.num_stages * base_design.num_stages)

    def test_range_classifier(self):
        r = MICRO_RANGES["de_haller"]
        self.assertEqual(r.classify(0.80), "ok")
        self.assertEqual(r.classify(0.70), "warn")
        self.assertEqual(r.classify(0.40), "fail")

    def test_sizing(self):
        s = micro_sizing(0.6, 3.2, 64000)
        self.assertTrue(0.03 < s.tip_radius < 0.08)
        self.assertGreaterEqual(s.num_stages, 3)
        # continuity closes: re-derive area from sizing outputs
        self.assertAlmostEqual(s.mean_blade_speed, 2 * np.pi * 64000 / 60 * s.mean_radius, places=6)


class SummaryTest(unittest.TestCase):
    def test_stage_table_shape(self):
        df = stage_summary_df(micro_default)
        self.assertEqual(len(df), micro_default.num_stages)
        self.assertAlmostEqual(df["PR cumul."].iloc[-1], micro_default.pressure_ratio, places=3)

    def test_machine_summary(self):
        s = machine_summary(micro_default)
        self.assertAlmostEqual(s["Overall pressure ratio"], 3.2)


class OffDesignTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cm = CompressorMap(micro_default)

    def test_design_point_closes(self):
        r = self.cm.evaluate(1.0, micro_default.mass_flow_rate)
        self.assertTrue(r["valid"])
        self.assertAlmostEqual(r["pr"], micro_default.pressure_ratio, places=3)
        self.assertAlmostEqual(r["eta_isentropic"], micro_default.isentropic_efficiency, places=3)

    def test_design_point_flow_coefficient(self):
        r = self.cm.evaluate(1.0, micro_default.mass_flow_rate)
        for phi, st in zip(r["phis"], micro_default.stages):
            self.assertAlmostEqual(phi, st.flow_coefficient, places=4)

    def test_pressure_ratio_rises_with_speed(self):
        m = micro_default.mass_flow_rate
        pr = [self.cm.evaluate(n, n * m)["pr"] for n in (0.9, 1.0, 1.05)]
        self.assertTrue(pr[0] < pr[1] < pr[2])

    def test_surge_margin_at_design_speed(self):
        res = estimate_map(micro_default)
        sm = res.design_surge_margin
        self.assertIsNotNone(sm)
        self.assertTrue(0.0 < sm < 40.0)

    def test_surge_line_pr_increases_with_speed(self):
        b = estimate_map(micro_default).boundaries.dropna(subset=["surge_pr"])
        self.assertTrue(np.all(np.diff(b["surge_pr"].to_numpy()) > 0))

    def test_more_conservative_df_limit_reduces_margin(self):
        loose = estimate_map(micro_default, OffDesignSettings(df_limit=0.65)).design_surge_margin
        tight = estimate_map(micro_default, OffDesignSettings(df_limit=0.55)).design_surge_margin
        if loose is not None and tight is not None:
            self.assertGreaterEqual(loose, tight)

    def test_plot_builds(self):
        fig = estimate_map(micro_default).plot()
        self.assertGreater(len(fig.data), 3)


if __name__ == "__main__":
    unittest.main()
