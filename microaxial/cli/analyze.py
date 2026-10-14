import click

import pandas as pd

from microaxial.cli.state import load_design
from microaxial.cli.utils import design_option


@click.group()
def analyze() -> None:
    """Run mean-line analysis on the active design."""
    pass


def _format_options(fn):
    fn = click.option("--csv", "use_csv", is_flag=True, help="Output as CSV")(fn)
    return fn


def _output_df(ctx: click.Context, df: pd.DataFrame, use_csv: bool) -> None:
    """Output a DataFrame in the appropriate format."""
    fmt = ctx.obj["fmt"]
    if fmt.use_json:
        fmt.output(df.to_dict())
    elif use_csv:
        click.echo(df.to_csv())
    else:
        click.echo(df.to_string())


@analyze.command("machine")
@design_option
@_format_options
@click.pass_context
def analyze_machine(ctx: click.Context, design_name: str | None, use_csv: bool) -> None:
    """Show all turbomachinery-level properties (auto-generated from model fields)."""
    fmt = ctx.obj["fmt"]
    try:
        _, tm = load_design(design_name)
    except ValueError as e:
        fmt.error(str(e))
        return

    from microaxial.exporter import machine_properties_df
    df = machine_properties_df(tm)
    _output_df(ctx, df, use_csv)


@analyze.command("stages")
@design_option
@_format_options
@click.pass_context
def analyze_stages(ctx: click.Context, design_name: str | None, use_csv: bool) -> None:
    """Show all per-stage scalar properties (auto-generated from Stage dataclass)."""
    fmt = ctx.obj["fmt"]
    try:
        _, tm = load_design(design_name)
    except ValueError as e:
        fmt.error(str(e))
        return

    from microaxial.exporter import dataclass_list_to_df
    df = dataclass_list_to_df(tm.stages)
    _output_df(ctx, df, use_csv)


@analyze.command("flow-stations")
@design_option
@_format_options
@click.pass_context
def analyze_flow_stations(ctx: click.Context, design_name: str | None, use_csv: bool) -> None:
    """Show per-stage inlet + mid flow station properties (auto-generated from FlowStation)."""
    fmt = ctx.obj["fmt"]
    try:
        _, tm = load_design(design_name)
    except ValueError as e:
        fmt.error(str(e))
        return

    from microaxial.exporter import stages_flow_stations_df
    df = stages_flow_stations_df(tm)
    _output_df(ctx, df, use_csv)


@analyze.command("blade-rows")
@design_option
@_format_options
@click.pass_context
def analyze_blade_rows(ctx: click.Context, design_name: str | None, use_csv: bool) -> None:
    """Show per-stage rotor + stator blade row properties (auto-generated from BladeRow)."""
    fmt = ctx.obj["fmt"]
    try:
        _, tm = load_design(design_name)
    except ValueError as e:
        fmt.error(str(e))
        return

    from microaxial.exporter import stages_blade_rows_df, stages_blade_rows_streams_df, stages_blade_rows_sub_models_dfs
    df = stages_blade_rows_df(tm)
    df_streams = stages_blade_rows_streams_df(tm)
    sub_models = stages_blade_rows_sub_models_dfs(tm)

    if fmt.use_json:
        combined: dict[str, object] = {
            "blade_rows": df.to_dict(),
            "per_stream": df_streams.to_dict(),
        }
        for section_name, df_sm in sub_models.items():
            combined[section_name] = df_sm.to_dict()
        fmt.output(combined)
    else:
        _output_df(ctx, df, use_csv)
        click.echo("")
        click.echo("--- Per-Stream Properties ---")
        click.echo("")
        _output_df(ctx, df_streams, use_csv)
        for section_name, df_sm in sub_models.items():
            click.echo("")
            click.echo(f"--- {section_name.replace('_', ' ').title()} ---")
            click.echo("")
            _output_df(ctx, df_sm, use_csv)


@analyze.command("summary")
@design_option
@click.pass_context
def analyze_summary(ctx: click.Context, design_name: str | None) -> None:
    """Machine overview and stage-by-stage design summary table."""
    fmt = ctx.obj["fmt"]
    try:
        _, tm = load_design(design_name)
    except ValueError as e:
        fmt.error(str(e))
        return

    from microaxial.summary import format_summary, machine_summary, stage_summary_df
    if fmt.use_json:
        fmt.output({"machine": machine_summary(tm).to_dict(), "stages": stage_summary_df(tm).to_dict(orient="index")})
    else:
        click.echo(format_summary(tm))


@analyze.command("checks")
@design_option
@click.option("--all", "show_all", is_flag=True, help="Show every check, not only warnings and failures")
@_format_options
@click.pass_context
def analyze_checks(ctx: click.Context, design_name: str | None, show_all: bool, use_csv: bool) -> None:
    """Micro gas turbine design mode: check against recommended ranges."""
    fmt = ctx.obj["fmt"]
    try:
        _, tm = load_design(design_name)
    except ValueError as e:
        fmt.error(str(e))
        return

    from microaxial.micro_mode import check_summary, evaluate_design
    df = evaluate_design(tm)
    counts = check_summary(tm)
    if not show_all:
        df = df[df["status"] != "ok"]
    df = df.drop(columns=["note"]) if not fmt.use_json else df
    if fmt.use_json:
        fmt.output({"counts": counts, "checks": df.to_dict(orient="records")})
        return
    click.echo(f"Checks: {counts['ok']} ok, {counts['warn']} warn, {counts['fail']} fail")
    if df.empty:
        click.echo("All checks inside recommended micro-compressor ranges.")
    else:
        _output_df(ctx, df.reset_index(drop=True), use_csv)


@analyze.command("map")
@design_option
@click.option("--plot", "plot_path", type=click.Path(), default=None, help="Write the map to .html or .png")
@click.option("--k-inc", type=float, default=None, help="Incidence loss coefficient (default 5.0)")
@click.option("--df-limit", type=float, default=None, help="Stall-onset diffusion factor (default 0.60)")
@click.option("--running-line-exp", type=float, default=None, help="Running line exponent n in PR-1 ~ (m/m_d)^n (default 2)")
@_format_options
@click.pass_context
def analyze_map(ctx: click.Context, design_name: str | None, plot_path: str | None, k_inc: float | None,
                df_limit: float | None, running_line_exp: float | None, use_csv: bool) -> None:
    """Estimate the off-design compressor map, surge line and surge margin."""
    fmt = ctx.obj["fmt"]
    try:
        _, tm = load_design(design_name)
    except ValueError as e:
        fmt.error(str(e))
        return

    from microaxial.offdesign import OffDesignSettings, estimate_map
    settings = OffDesignSettings()
    if k_inc is not None:
        settings.k_inc = k_inc
    if df_limit is not None:
        settings.df_limit = df_limit
    if running_line_exp is not None:
        settings.running_line_exponent = running_line_exp
    result = estimate_map(tm, settings)
    table = result.surge_margin_table()
    if plot_path:
        result.plot(plot_path)
    if fmt.use_json:
        fmt.output({"design_surge_margin_pct": result.design_surge_margin, "boundaries": table.to_dict(orient="records")})
        return
    _output_df(ctx, table.round(4).reset_index(drop=True), use_csv)
    sm = result.design_surge_margin
    click.echo("")
    click.echo("Design-speed surge margin: " + ("n/a" if sm is None else f"{sm:.1f} %") +
               "   (screening-level estimate; see README for model limits)")
    if plot_path:
        click.echo(f"Map written to {plot_path}")

