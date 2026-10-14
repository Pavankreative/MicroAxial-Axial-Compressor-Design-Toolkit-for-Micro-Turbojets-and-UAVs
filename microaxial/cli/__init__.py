from typing import Any, Callable, Optional

import click
import json
import sys

from microaxial.cli.design import design
from microaxial.cli.analyze import analyze
from microaxial.cli.cad import cad


class JsonFormatter:
    """Context object that tracks --json flag for structured output."""
    def __init__(self, use_json: bool) -> None:
        self.use_json = use_json

    def output(self, data: Any, text_fn: Optional[Callable[[Any], str]] = None) -> None:
        """Output data as JSON or formatted text."""
        if self.use_json:
            click.echo(json.dumps(data, indent=2, default=str))
        elif text_fn:
            click.echo(text_fn(data))
        else:
            click.echo(data)

    def error(self, message: str, code: int = 1) -> None:
        """Output error and exit."""
        if self.use_json:
            click.echo(json.dumps({"error": message}), err=True)
        else:
            click.echo(f"Error: {message}", err=True)
        sys.exit(code)


from importlib.metadata import version, PackageNotFoundError

try:
    _version = version("microaxial")
except PackageNotFoundError:
    _version = "0.0.0"

BANNER = "\b\n" + fr"""  __  __ _                 _          _       _
 |  \/  (_) ___ _ __ ___  / \   __  _(_) __ _| |
 | |\/| | |/ __| '__/ _ \/ _ \  \ \/ / |/ _` | |
 | |  | | | (__| | | (_) / ___ \  >  <| | (_| | |
 |_|  |_|_|\___|_|  \___/_/   \_\/_/\_\_|\__,_|_|

        Micro-turbojet / UAV axial compressor design toolkit
                              v{_version}
"""

HELP_TEXT = BANNER + """
\b
NOTE FOR AI AGENTS:
  - The --json flag goes HERE on the root command, not on subcommands.
    CORRECT: microaxial --json axial compressor analyze machine
    WRONG:   microaxial axial compressor analyze machine --json
  - Create designs with inline JSON (do NOT write temp files):
    microaxial axial compressor design create NAME --json '{...}'
  - During design iteration keep visualization ON (default) and use
    simple mode (default). Only use --complex for the final build.

\b
Quick start:
  microaxial axial compressor size --mass-flow 0.6 --pressure-ratio 3.2 --rpm 64000
  microaxial axial compressor design init mk1          # bundled micro-turbojet template
  microaxial axial compressor analyze summary          # stage-by-stage table
  microaxial axial compressor analyze checks           # micro-compressor range checks
  microaxial axial compressor analyze map --plot map.html
  microaxial axial compressor design create mark1 --json '{...}'
  microaxial axial compressor design use mark1
  microaxial --json axial compressor analyze stages
  microaxial axial compressor cad assembly
  microaxial axial compressor cad assembly --complex
"""


@click.group(context_settings={"max_content_width": 120}, help=HELP_TEXT)
@click.option("--json", "use_json", is_flag=True, help="Output as JSON (for AI agents)")
@click.pass_context
def cli(ctx: click.Context, use_json: bool) -> None:
    """MicroAxial CLI."""
    ctx.ensure_object(dict)
    ctx.obj["fmt"] = JsonFormatter(use_json)


@cli.group()
@click.pass_context
def axial(ctx: click.Context) -> None:
    """Axial turbomachinery (compressor, turbine, fan)."""
    pass


@axial.group()
@click.pass_context
def compressor(ctx: click.Context) -> None:
    """Axial compressor design, analysis, CAD, and export."""
    pass


compressor.add_command(design)
compressor.add_command(analyze)
compressor.add_command(cad)


@compressor.command("size")
@click.option("--mass-flow", type=float, required=True, help="Inlet mass flow (kg/s)")
@click.option("--pressure-ratio", type=float, required=True, help="Overall total pressure ratio")
@click.option("--rpm", type=float, required=True, help="Shaft speed (rev/min)")
@click.option("--phi", "flow_coefficient", type=float, default=0.55, show_default=True, help="Target mean flow coefficient")
@click.option("--psi", "stage_loading", type=float, default=0.40, show_default=True, help="Target stage loading coefficient")
@click.option("--hub-to-tip", type=float, default=0.68, show_default=True, help="Inlet hub-to-tip ratio")
@click.option("--eta-poly", type=float, default=0.85, show_default=True, help="Assumed polytropic efficiency")
@click.pass_context
def size(ctx: click.Context, mass_flow: float, pressure_ratio: float, rpm: float, flow_coefficient: float,
         stage_loading: float, hub_to_tip: float, eta_poly: float) -> None:
    """Micro-mode first-pass sizing: annulus, tip speed and stage count from engine requirements."""
    from microaxial.micro_mode import micro_sizing
    fmt = ctx.obj["fmt"]
    r = micro_sizing(mass_flow, pressure_ratio, rpm, target_flow_coefficient=flow_coefficient,
                     target_stage_loading=stage_loading, hub_to_tip_ratio=hub_to_tip, polytropic_efficiency=eta_poly)

    def text(d):
        return (f"Tip radius            {d['tip_radius']*1e3:8.1f} mm\n"
                f"Hub radius            {d['hub_radius']*1e3:8.1f} mm\n"
                f"Tip speed             {d['tip_speed']:8.0f} m/s\n"
                f"Mean blade speed      {d['mean_blade_speed']:8.0f} m/s\n"
                f"Axial velocity        {d['axial_velocity']:8.0f} m/s\n"
                f"Overall dTt           {d['total_temperature_rise']:8.1f} K\n"
                f"Stage dTt (target)    {d['stage_temperature_rise']:8.1f} K\n"
                f"Stages (rounded up)   {d['num_stages']:8d}\n"
                f"Tip relative Mach     {d['inlet_tip_relative_mach']:8.2f}\n"
                "\nStarting point only: refine with 'design create' and 'analyze checks'.")
    fmt.output(r.as_dict(), text)

