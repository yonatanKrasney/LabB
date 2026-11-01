"""
app.py -- the Lab B fitting tool.

Start it by double-clicking run.bat. It opens in your browser; everything runs
on your own laptop and nothing is uploaded anywhere.

All of the statistics live in fitting.py. This file is only the interface.
"""

from __future__ import annotations

import html
import io
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from plots import EXCLUDED_COLOR, FIT_COLOR, INCLUDED_COLOR, report_figure
from fitting import (
    FORMULA_PRESETS,
    FormulaError,
    blocks_to_text,
    column_values,
    derived_quantity,
    find_data_files,
    format_measurement,
    format_percent,
    interpret_chi2,
    interpret_p_value,
    latex_label,
    latex_table,
    list_sheets,
    load_table,
    methods_blocks,
    numeric_columns,
    plain_number,
    round_to_uncertainty,
    run_fit,
    source_block,
    split_header,
)

APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"
STATE_FILE = APP_DIR / ".labfit_session.json"

st.set_page_config(page_title="Lab B fitting", page_icon="chart", layout="wide")


# ==========================================================================
# Remembering what you had open last time
# ==========================================================================

def load_state() -> dict:
    try:
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_state() -> None:
    """Write the current choices to disk, so reopening picks up where you left."""
    payload = {
        "ui": {k: v for k, v in st.session_state.get("ui", {}).items()},
        "exclusions": {
            key: {"rows": sorted(info["rows"]), "reasons": info["reasons"]}
            for key, info in st.session_state.get("exclusions", {}).items()
        },
    }
    try:
        STATE_FILE.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    except Exception:
        pass          # never let a failed save interrupt the analysis


if "ui" not in st.session_state:
    saved = load_state()
    st.session_state.ui = saved.get("ui", {})
    st.session_state.exclusions = {
        key: {"rows": set(info.get("rows", [])), "reasons": info.get("reasons", {})}
        for key, info in saved.get("exclusions", {}).items()
    }
    st.session_state.plot_nonce = 0

ui: dict = st.session_state.ui


def remember(name: str, default):
    """Previously used value for a widget, or a default."""
    return ui.get(name, default)


def pick_index(options: list, name: str, default: int = 0) -> int:
    """Index of the remembered choice inside `options`, if it is still there."""
    previous = ui.get(name)
    if previous in options:
        return options.index(previous)
    return min(default, max(len(options) - 1, 0))


def stable_selectbox(container, label, options, key, remember_name,
                     default_index=0, **kwargs):
    """A selectbox with an explicit key, so it keeps its value when the
    widgets around it appear or disappear -- the Sheet dropdown showing up
    for an Excel file used to be enough to confuse the ones below it.

    If the remembered choice is no longer among the options (you changed
    file, and that column does not exist in the new one) the stale value is
    dropped rather than raising.
    """
    if key in st.session_state and st.session_state[key] not in options:
        del st.session_state[key]
    if key not in st.session_state:
        previous = ui.get(remember_name)
        st.session_state[key] = (
            previous if previous in options
            else options[min(default_index, len(options) - 1)]
        )
    return container.selectbox(label, options, key=key, **kwargs)


# ==========================================================================
# Sidebar: where the data is, and what to fit to it
# ==========================================================================

st.sidebar.title("Lab B fitting")

st.sidebar.subheader("1. Data")

def pick_folder_dialog(start: str) -> str | None:
    """Open the normal Windows folder-chooser.

    Works because the app is running on your own machine: the dialog appears
    on the same computer as the browser. Returns None if the dialog could not
    be opened at all, which is why the text box below it stays.
    """
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception:                         # noqa: BLE001 - tkinter not installed
        return None
    try:
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)     # try to beat the browser to the front
        chosen = filedialog.askdirectory(
            initialdir=start if Path(start).is_dir() else str(APP_DIR),
            title="Choose the folder with your data",
        )
        root.destroy()
        return chosen or ""
    except Exception:                         # noqa: BLE001 - no display, etc.
        return None


def browse_for_folder() -> None:
    """Runs as the Browse button's on_click.

    It has to be a callback: Streamlit refuses to let you assign to a
    widget's session_state key once that widget has been created, and the
    folder box is created further down. Callbacks run at the start of the
    rerun, before any widget exists, so the assignment is allowed there.
    """
    picked = pick_folder_dialog(st.session_state.get("folder_box", str(DATA_DIR)))
    if picked is None:
        st.session_state["_browse_failed"] = True
    elif picked:
        st.session_state["folder_box"] = picked
        st.session_state["_browse_failed"] = False


if "folder_box" not in st.session_state:
    st.session_state.folder_box = remember("folder", str(DATA_DIR))

folder_text = st.sidebar.text_input(
    "Folder to read",
    key="folder_box",
    help="Drop your CSV or Excel files in here. Nothing gets renamed or uploaded.",
)
ui["folder"] = folder_text
folder = Path(folder_text).expanduser()

cols = st.sidebar.columns(2)
cols[0].button("Browse...", on_click=browse_for_folder,
               help="Pick a folder in Windows Explorer")
cols[1].button("Refresh", help="Re-read the folder")

if st.session_state.pop("_browse_failed", False):
    st.sidebar.warning(
        "Could not open the folder dialog on this computer. "
        "Paste the path into the box instead."
    )
else:
    st.sidebar.caption(
        "The dialog can open **behind** this window -- check your taskbar if "
        "nothing seems to happen."
    )

files = find_data_files(folder)
if not files:
    st.sidebar.warning(f"No data files in {folder}")
    st.title("Lab B fitting")
    st.info(
        f"**Put a file in `{folder}` to start.**\n\n"
        "CSV, TSV or Excel. Newest file appears at the top of the list. "
        "The folder path is editable in the sidebar if your data lives somewhere else."
    )
    save_state()
    st.stop()

file_names = [p.name for p in files]
chosen_name = stable_selectbox(
    st.sidebar, "File", file_names, key="file_box", remember_name="file",
    help="Newest first",
)
ui["file"] = chosen_name
path = folder / chosen_name

sheet = None
try:
    sheets = list_sheets(path)
except Exception as exc:                      # noqa: BLE001 - shown to the user
    st.error(f"**Could not open {chosen_name}.**\n\n{exc}")
    save_state()
    st.stop()
if sheets:
    sheet = stable_selectbox(
        st.sidebar, "Sheet", sheets, key=f"sheet_box@{chosen_name}",
        remember_name="sheet",
        help="Read straight out of the file -- no need to type the name",
    )
    ui["sheet"] = sheet

try:
    df = load_table(path, sheet)
except Exception as exc:                      # noqa: BLE001 - shown to the user
    st.error(f"**Could not read {chosen_name}.**\n\n{exc}")
    save_state()
    st.stop()

available = numeric_columns(df)
if len(available) < 4:
    st.error(
        f"**{chosen_name}** has only {len(available)} numeric column(s): "
        f"{', '.join(available) or 'none'}.\n\n"
        "The fit needs four: x, its error, y, and its error."
    )
    st.dataframe(df.head(20))
    st.stop()

st.sidebar.subheader("2. Columns")
st.sidebar.caption("Picked by their real headers, in the order they appear in the file.")

c1, c2 = st.sidebar.columns(2)
column_scope = f"{chosen_name}|{sheet}"
x_col = stable_selectbox(c1, "x", available, f"xc@{column_scope}", "x_col", 0)
dx_col = stable_selectbox(c2, "error in x", available, f"dxc@{column_scope}", "dx_col", 1)
y_col = stable_selectbox(c1, "y", available, f"yc@{column_scope}", "y_col", 2)
dy_col = stable_selectbox(c2, "error in y", available, f"dyc@{column_scope}", "dy_col", 3)
ui.update(x_col=x_col, dx_col=dx_col, y_col=y_col, dy_col=dy_col)

if len({x_col, dx_col, y_col, dy_col}) < 4:
    st.sidebar.error("The same column is selected more than once.")

st.sidebar.subheader("3. Labels")

# Labels are stored PER DATASET and the widgets are keyed on it. Previously
# they were global, so switching file kept the old names and the methods text
# cheerfully described the wrong experiment. Defaults are read out of the
# column headers, which already carry the units.
label_key = f"{chosen_name}|{sheet}|{x_col}|{y_col}"
default_x_name, default_x_units = split_header(x_col)
default_y_name, default_y_units = split_header(y_col)

st.sidebar.caption("Filled in from the column headers. Edit if you want.")
x_name = st.sidebar.text_input(
    "x name", remember(f"x_name@{label_key}", default_x_name), key=f"xn@{label_key}")
x_units = st.sidebar.text_input(
    "x units", remember(f"x_units@{label_key}", default_x_units), key=f"xu@{label_key}")
y_name = st.sidebar.text_input(
    "y name", remember(f"y_name@{label_key}", default_y_name), key=f"yn@{label_key}")
y_units = st.sidebar.text_input(
    "y units", remember(f"y_units@{label_key}", default_y_units), key=f"yu@{label_key}")
title = st.sidebar.text_input(
    "Plot title",
    remember(f"title@{label_key}", f"{y_name} as a function of {x_name}"),
    key=f"ti@{label_key}",
)
ui[f"x_name@{label_key}"] = x_name
ui[f"x_units@{label_key}"] = x_units
ui[f"y_name@{label_key}"] = y_name
ui[f"y_units@{label_key}"] = y_units
ui[f"title@{label_key}"] = title
annotate = st.sidebar.checkbox(
    "Show chi²/p-value box on the figure",
    value=remember("annotate", True),
    help="A small box in the corner of the exported figure. Turn it off if your "
         "report quotes the numbers in the caption instead.",
)
ui["annotate"] = annotate


def axis_label(name: str, units: str) -> str:
    return f"{name} [{units}]" if units.strip() else name


st.sidebar.subheader("4. Model")


def apply_preset() -> None:
    name = st.session_state.preset_choice
    if name in FORMULA_PRESETS:
        st.session_state.formula_text = FORMULA_PRESETS[name]


preset_options = ["(type my own)"] + list(FORMULA_PRESETS)
st.sidebar.selectbox(
    "Start from", preset_options, key="preset_choice", on_change=apply_preset,
    help="Fills the box below. You can edit it afterwards.",
)

if "formula_text" not in st.session_state:
    st.session_state.formula_text = remember("formula", "a0 + a1*x")

formula = st.sidebar.text_area(
    "Formula",
    key="formula_text",
    height=80,
    help=(
        "Write the model in terms of x and parameters a0, a1, a2, ...\n"
        "The number of parameters is read off the formula.\n"
        "Available: + - * / ** (or ^), sin cos tan arctan exp log log10 sqrt abs, pi, e"
    ),
)
ui["formula"] = formula

with st.sidebar.expander("Initial guess (optional)"):
    st.caption(
        "Usually unnecessary -- the solver tries several starting points. "
        "Set one if a stubborn nonlinear fit needs steering."
    )
    guess_text = st.text_input(
        "Comma-separated, one per parameter",
        remember("guess", ""),
        placeholder="e.g. 1, 0.5, 0",
    )
    ui["guess"] = guess_text
    multi_start = st.checkbox(
        "Also try automatic starting points", value=remember("multi_start", True)
    )
    ui["multi_start"] = multi_start

initial_guess = None
if guess_text.strip():
    try:
        initial_guess = [float(v) for v in guess_text.replace(";", ",").split(",") if v.strip()]
    except ValueError:
        st.sidebar.error("The initial guess must be numbers separated by commas.")

st.sidebar.divider()
st.sidebar.caption(f"Reading `{path.name}`" + (f" / sheet `{sheet}`" if sheet else ""))


# ==========================================================================
# The data, and which points are currently dropped
# ==========================================================================

x_all = column_values(df, x_col)
dx_all = column_values(df, dx_col)
y_all = column_values(df, y_col)
dy_all = column_values(df, dy_col)
n_rows = len(x_all)

dataset_key = f"{path.name}|{sheet}|{x_col}|{dx_col}|{y_col}|{dy_col}"
store = st.session_state.exclusions.setdefault(dataset_key, {"rows": set(), "reasons": {}})
store["rows"] = {int(r) for r in store["rows"] if int(r) < n_rows}
excluded: set[int] = store["rows"]
reasons: dict = store["reasons"]

mask = np.zeros(n_rows, dtype=bool)
if excluded:
    mask[sorted(excluded)] = True


def toggle_rows(rows: list[int]) -> None:
    for row in rows:
        if row in excluded:
            excluded.discard(row)
            reasons.pop(str(row), None)
        else:
            excluded.add(row)


# ==========================================================================
# Run the fit (twice, if points are excluded)
# ==========================================================================

fit_error = None
result = None
result_all = None

try:
    result = run_fit(
        x_all, dx_all, y_all, dy_all, formula,
        initial_guess=initial_guess, exclude=mask, try_multiple_starts=multi_start,
    )
    if excluded:
        try:
            result_all = run_fit(
                x_all, dx_all, y_all, dy_all, formula,
                initial_guess=initial_guess, try_multiple_starts=multi_start,
            )
        except Exception:
            result_all = None
except (FormulaError, ValueError, RuntimeError) as exc:
    fit_error = str(exc)


# ==========================================================================
# Main area
# ==========================================================================

st.title(title or "Lab B fitting")

if fit_error:
    st.error(f"**The fit did not run.**\n\n{fit_error}")

if result is not None and result.warnings:
    for warning in result.warnings:
        st.warning(warning)

tab_fit, tab_stats, tab_report = st.tabs(
    ["Fit and points", "Statistics", "Report output"]
)


# -------------------------------------------------------------------- plots

def make_plotly_figure(res) -> tuple[go.Figure, list[np.ndarray]]:
    """Interactive fit plot. Returns the figure and, per trace, the row indices
    its points came from, so a click can be traced back to a table row."""
    fig = go.Figure()
    trace_rows: list[np.ndarray] = []

    kept_rows = np.where(~mask)[0]
    drop_rows = np.where(mask)[0]

    fig.add_trace(go.Scatter(
        x=x_all[kept_rows], y=y_all[kept_rows],
        error_x=dict(type="data", array=dx_all[kept_rows], color="#b0b0b0", thickness=1),
        error_y=dict(type="data", array=dy_all[kept_rows], color="#b0b0b0", thickness=1),
        mode="markers",
        marker=dict(size=9, color=INCLUDED_COLOR),
        name="in the fit",
        # column 0 is the internal row index, column 1 the number you see
        customdata=np.column_stack([kept_rows, kept_rows + 1]),
        hovertemplate="point %{customdata[1]}<br>x=%{x:.6g}<br>y=%{y:.6g}<extra>click to drop</extra>",
    ))
    trace_rows.append(kept_rows)

    if len(drop_rows):
        fig.add_trace(go.Scatter(
            x=x_all[drop_rows], y=y_all[drop_rows],
            error_x=dict(type="data", array=dx_all[drop_rows], color="#d8d8d8", thickness=1),
            error_y=dict(type="data", array=dy_all[drop_rows], color="#d8d8d8", thickness=1),
            mode="markers",
            marker=dict(size=10, color="rgba(0,0,0,0)",
                        line=dict(color=EXCLUDED_COLOR, width=2)),
            name="excluded",
            customdata=np.column_stack([drop_rows, drop_rows + 1]),
            hovertemplate="point %{customdata[1]}<br>x=%{x:.6g}<br>y=%{y:.6g}<extra>click to put back</extra>",
        ))
        trace_rows.append(drop_rows)

    if res is not None:
        xs, ys = res.curve()
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="lines", name="fit",
            line=dict(color=FIT_COLOR, width=2),
            hoverinfo="skip",
        ))
        trace_rows.append(np.array([], dtype=int))

    fig.update_layout(
        height=430,
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis_title=axis_label(x_name, x_units),
        yaxis_title=axis_label(y_name, y_units),
        legend=dict(orientation="h", y=1.1, x=0),
        dragmode="select",
        clickmode="event+select",
        template="simple_white",
    )
    return fig, trace_rows


def make_residual_figure(res) -> go.Figure:
    fig = go.Figure()
    if res is None:
        return fig
    fig.add_trace(go.Scatter(
        x=res.x, y=res.residuals,
        error_y=dict(type="data", array=res.residual_errors, color="#b0b0b0", thickness=1),
        error_x=dict(type="data", array=res.dx, color="#b0b0b0", thickness=1),
        mode="markers", marker=dict(size=8, color=INCLUDED_COLOR),
        name="residual",
        hovertemplate="x=%{x:.6g}<br>residual=%{y:.4g}<extra></extra>",
    ))
    fig.add_hline(y=0, line=dict(color=FIT_COLOR, dash="dash"))
    fig.update_layout(
        height=300,
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis_title=axis_label(x_name, x_units),
        yaxis_title=f"{y_name} - f({x_name})" + (f" [{y_units}]" if y_units.strip() else ""),
        showlegend=False,
        template="simple_white",
    )
    return fig


# A run of Latin letters, digits and the punctuation that clings to them.
# Inside a Hebrew sentence such a run has to be isolated, or the bidi
# algorithm moves its brackets and drags the sentence's full stop along with
# them: 'f(x)·dx,' comes out as ',f(x)·dx'. <bdi> tells the browser to lay the
# run out on its own, and it affects the display only -- the text you copy is
# untouched.
_LATIN_RUN = re.compile(
    r"[A-Za-z][A-Za-z0-9_^'()\[\]./*+· -]*[A-Za-z0-9_)\]]|[A-Za-z]"
)


def _inline_html(text: str) -> str:
    """Turn the small amount of markdown used in the methods text into HTML."""
    out = html.escape(text)
    out = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", out)
    out = re.sub(r"`([^`]+)`", r"<code>\1</code>", out)
    return out


def _inline_html_rtl(text: str) -> str:
    """As above, then isolate every Latin run so it cannot reorder the line."""
    out = _LATIN_RUN.sub(lambda m: f"\u0000{m.group(0)}\u0001", text)
    out = _inline_html(out)
    return out.replace("\u0000", "<bdi>").replace("\u0001", "</bdi>")


def render_blocks(blocks, rtl: bool = False) -> None:
    """Show the methods write-up.

    Hebrew paragraphs go inside a right-to-left container; formulas are
    rendered as code, which keeps them left-to-right and stops the bidi
    algorithm from rearranging their brackets.
    """
    for kind, content in blocks:
        if kind == "formula":
            st.code(content, language=None)
        elif rtl:
            st.markdown(
                '<div dir="rtl" style="text-align:right; line-height:1.7; '
                f'margin-bottom:0.8rem">{_inline_html_rtl(content)}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(content)


def make_report_figure(res):
    """The version that goes into the report. Drawn by plots.py."""
    return report_figure(
        res, x_all, dx_all, y_all, dy_all, mask,
        x_label=axis_label(x_name, x_units),
        y_label=axis_label(y_name, y_units),
        title=title,
        annotate_statistics=annotate,
    )


# --------------------------------------------------------- tab 1: fit/points

with tab_fit:
    left, right = st.columns([3, 2], gap="medium")

    with left:
        figure, trace_rows = make_plotly_figure(result)
        event = st.plotly_chart(
            figure,
            on_select="rerun",
            selection_mode=("points", "box", "lasso"),
            key=f"mainplot_{st.session_state.plot_nonce}",
        )

        def rows_from_event(ev) -> list[int]:
            selection = None
            if ev is not None:
                selection = getattr(ev, "selection", None)
                if selection is None and isinstance(ev, dict):
                    selection = ev.get("selection")
            if not selection:
                return []
            found: list[int] = []
            for point in selection.get("points", []):
                data = point.get("customdata")
                if isinstance(data, (list, tuple)) and data:
                    found.append(int(data[0]))
                    continue
                curve = point.get("curve_number")
                index = point.get("point_index", point.get("point_number"))
                if curve is not None and index is not None and curve < len(trace_rows):
                    rows = trace_rows[curve]
                    if index < len(rows):
                        found.append(int(rows[index]))
            return sorted(set(found))

        clicked = rows_from_event(event)
        if clicked:
            toggle_rows(clicked)
            st.session_state.plot_nonce += 1     # reset the plot's selection
            save_state()
            st.rerun()

        st.caption(
            "**Click a point to drop it from the fit. Click it again to put it back.** "
            "Drag a box or lasso to do several at once. Dropped points stay on the "
            "plot as hollow grey circles -- they are never deleted."
        )

        if excluded:
            if st.button(f"Put back all {len(excluded)} excluded point(s)"):
                excluded.clear()
                reasons.clear()
                st.session_state.plot_nonce += 1
                save_state()
                st.rerun()

        st.plotly_chart(make_residual_figure(result))
        st.caption(
            "Residual error bars are the **combined** error "
            "sqrt(dy² + (f'(x)·dx)²), not dy alone -- an error in x moves the "
            "curve too, and ignoring that makes a good fit look bad."
        )

    with right:
        st.markdown("##### Points")
        st.caption("Untick a row to leave it out. Same effect as clicking the plot.")

        table = pd.DataFrame({
            "use": ~mask,
            "point": np.arange(1, n_rows + 1),      # 1-based, as in the methods text
            x_name or "x": x_all,
            f"±{x_name}" if x_name else "±x": dx_all,
            y_name or "y": y_all,
            f"±{y_name}" if y_name else "±y": dy_all,
        })
        value_columns = [c for c in table.columns if c not in ("use",)]

        edited = st.data_editor(
            table,
            hide_index=True,
            height=360,
            disabled=value_columns,
            column_config={
                "use": st.column_config.CheckboxColumn(
                    "use", help="Tick = included in the fit", width="small"
                ),
                "point": st.column_config.NumberColumn("point", width="small"),
            },
            key=f"editor_{dataset_key}_{len(excluded)}_{hash(tuple(sorted(excluded)))}",
        )

        wanted_excluded = {
            int(point) - 1 for point, use in zip(edited["point"], edited["use"]) if not use
        }
        if wanted_excluded != excluded:
            for row in excluded - wanted_excluded:
                reasons.pop(str(row), None)
            store["rows"] = wanted_excluded
            st.session_state.plot_nonce += 1
            save_state()
            st.rerun()

        if excluded:
            st.markdown("##### Why was each point dropped?")
            st.caption(
                "This goes into the methods text, so you are not reconstructing "
                "it from memory three weeks later."
            )
            for row in sorted(excluded):
                reasons[str(row)] = st.text_input(
                    f"Point {row + 1}  (x={x_all[row]:.4g}, y={y_all[row]:.4g})",
                    value=reasons.get(str(row), ""),
                    key=f"reason_{dataset_key}_{row}",
                    placeholder="e.g. scope triggered early; magnet slipped",
                )


# ------------------------------------------------------- tab 2: statistics

with tab_stats:
    if result is None:
        st.info("No fit to report yet.")
    else:
        st.markdown("##### Fitted parameters")
        with st.expander("What am I looking at?"):
            st.markdown(
                "- **value ± error** — the parameter and its uncertainty, "
                "rounded so the error has two significant digits.\n"
                "- **relative error** — the error as a percentage of the value. "
                "It tells you which parameters the data actually pins down: "
                "2% is well determined, 80% means the data barely constrains it.\n"
                "- **sign** — worth a glance. A slope that comes out negative "
                "when theory says positive usually means two columns are "
                "swapped, or an axis runs the other way."
            )

        unit_columns = st.columns(result.n_params)
        param_units: list[str] = []
        for i in range(result.n_params):
            param_units.append(unit_columns[i].text_input(
                f"units of a{i}",
                value=ui.get(f"punits_{dataset_key}_{i}", ""),
                key=f"punits_{dataset_key}_{i}",
                placeholder="e.g. V s/rad",
            ))
            ui[f"punits_{dataset_key}_{i}"] = param_units[i]

        rows = []
        for i, (value, error) in enumerate(zip(result.params, result.param_errors)):
            rows.append({
                "parameter": f"a{i}",
                "value ± error": format_measurement(value, error),
                "units": param_units[i],
                "relative error": format_percent(value, error),
                "sign": "positive" if value > 0 else ("negative" if value < 0 else "zero"),
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True)

        st.markdown("##### Goodness of fit")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("chi²", f"{result.chi2:#.4g}")
        m2.metric("degrees of freedom", f"{result.dof}")
        m3.metric("reduced chi²", f"{result.chi2_reduced:.3g}")
        # A p-value that underflows to exactly 0 should not read as a bug.
        m4.metric(
            "p-value",
            "< 1e-300" if result.p_value == 0 else f"{result.p_value:.3g}",
        )
        st.write(interpret_chi2(result.chi2_reduced))
        st.write(interpret_p_value(result.p_value))
        with st.expander("What am I looking at?"):
            st.markdown(
                "- **chi²** — total disagreement. For each point: how far it "
                "sits from the curve measured in its own error bars, squared, "
                "all added up. A point one error bar away contributes 1; three "
                "error bars away contributes 9.\n"
                "- **degrees of freedom** — points minus parameters. Two "
                "parameters through two points fit exactly and prove nothing; "
                "this is how much real checking the data did.\n"
                "- **reduced chi²** — chi² divided by degrees of freedom, and "
                "the one to read. If your error bars are honest it lands near "
                "**1**. Far above: the model is wrong or the errors are too "
                "small. Far below: the errors are too generous.\n"
                "- **p-value** — if the model is right, how often would the "
                "scatter be at least this bad? Below 0.05 the fit is in "
                "trouble; above 0.95 the error bars are too big."
            )

        # ---- values formatted for pasting into the report or a slide ----
        st.markdown("##### Copy for the report")
        st.caption(
            "Hover over a box and a copy icon appears in its top-right corner."
        )

        def measurement_line(index: int) -> str:
            unit = param_units[index].strip()
            text = format_measurement(result.params[index], result.param_errors[index])
            return f"a{index} = {text}" + (f" {unit}" if unit else "")

        copy_lines = [measurement_line(i) for i in range(result.n_params)]
        copy_lines += [
            f"χ²_red = {result.chi2_reduced:.3g}",
            f"p = {plain_number(result.p_value, 3)}",
            f"dof = {result.dof}",
            f"N points = {result.n_points}"
            + (f" ({result.n_excluded} excluded)" if result.n_excluded else ""),
        ]

        copy_left, copy_right = st.columns([1, 1])
        with copy_left:
            st.caption("Everything at once")
            st.code("\n".join(copy_lines), language=None)
        with copy_right:
            st.caption("One at a time")
            for line in copy_lines:
                st.code(line, language=None)

        if result_all is not None:
            st.markdown("##### With and without the excluded points")
            st.caption(
                "If the parameters barely move, dropping the point only tidied "
                "the chi-squared and you should say so. If they move a lot, the "
                "exclusion is doing real work and needs a real justification."
            )
            comparison = []
            for i in range(result.n_params):
                kept_v, kept_e = result.params[i], result.param_errors[i]
                all_v, all_e = result_all.params[i], result_all.param_errors[i]
                shift = abs(kept_v - all_v) / max(np.hypot(kept_e, all_e), 1e-300)
                comparison.append({
                    "parameter": f"a{i}",
                    "all points": format_measurement(all_v, all_e),
                    "excluded dropped": format_measurement(kept_v, kept_e),
                    "shift (in sigma)": f"{shift:.2f}",
                })
            comparison.append({
                "parameter": "reduced chi²",
                "all points": f"{result_all.chi2_reduced:.3g}",
                "excluded dropped": f"{result.chi2_reduced:.3g}",
                "shift (in sigma)": "",
            })
            comparison.append({
                "parameter": "p-value",
                "all points": plain_number(result_all.p_value, 3),
                "excluded dropped": plain_number(result.p_value, 3),
                "shift (in sigma)": "",
            })
            st.dataframe(pd.DataFrame(comparison), hide_index=True)

        st.markdown("##### Comparison with theory (N-sigma test)")
        with st.expander("What am I looking at?"):
            st.markdown(
                "Does your measurement agree with the textbook value? You "
                "cannot answer that from the difference alone — it depends "
                "entirely on how precise your measurement is. N is that "
                "difference **measured in your own error bars**:\n\n"
                "```\nN = |measured - theory| / sqrt(sigma_measured² + sigma_theory²)\n```\n"
                "Measure g = 9.70 ± 0.15 against 9.81 and N = 0.73, which "
                "agrees. Measure g = 9.70 ± 0.01 and the same difference gives "
                "N = 11, which does not. Under 2 agrees, 2–3 is marginal, "
                "over 3 disagrees and the discussion has to explain it."
            )
        head = st.columns([1, 2, 2, 3])
        head[1].caption("**expected value** (from theory)")
        head[2].caption("**its uncertainty** (0 if it is exact)")
        head[3].caption("**result**")
        for i in range(result.n_params):
            c = st.columns([1, 2, 2, 3])
            c[0].markdown(f"**a{i}**")
            expected = c[1].text_input(
                f"expected value of a{i}", key=f"theory_{dataset_key}_{i}",
                value=ui.get(f"theory_{dataset_key}_{i}", ""),
                placeholder="leave blank to skip", label_visibility="collapsed",
            )
            expected_error = c[2].text_input(
                f"uncertainty of the expected value of a{i}",
                key=f"theory_err_{dataset_key}_{i}",
                value=ui.get(f"theory_err_{dataset_key}_{i}", "0"),
                placeholder="0", label_visibility="collapsed",
            )
            ui[f"theory_{dataset_key}_{i}"] = expected
            ui[f"theory_err_{dataset_key}_{i}"] = expected_error
            if expected.strip():
                try:
                    n_sigma = result.n_sigma(
                        i, float(expected), float(expected_error or 0)
                    )
                except ValueError:
                    c[3].error("not a number")
                else:
                    verdict = (
                        "agrees" if n_sigma < 2
                        else ("marginal" if n_sigma < 3 else "disagrees")
                    )
                    c[3].markdown(f"N = **{n_sigma:.2f}** — {verdict}")

        st.markdown("##### Derived quantity (indirect error)")
        with st.expander("What am I looking at?"):
            st.markdown(
                "Often the constant you want is not a fit parameter but comes "
                "out of one. Fit a pendulum's T² against L and the slope is "
                "4π²/g — you want g, not the slope. Type `4*pi**2/a1` here and "
                "you get it, with the error propagated:\n\n"
                "```\nsigma_g² = sum_ij (dg/da_i)(dg/da_j) * C_ij\n```\n"
                "This is the standard indirect-error formula your course "
                "teaches, plus the cross terms. Fit parameters are correlated, "
                "so those terms are not optional — leaving them out can be "
                "wrong by a factor of two or more. The checkbox below lets you "
                "see both numbers."
            )
        d1, d2, d3 = st.columns([3, 2, 2])
        derived_formula = d1.text_input(
            "formula in a0, a1, ...", value=remember("derived_formula", ""),
            placeholder="e.g. 1/(2*pi*a1)",
        )
        derived_label = d2.text_input("name", value=remember("derived_label", ""),
                                      placeholder="e.g. epsilon_0")
        derived_units = d3.text_input("units", value=remember("derived_units", ""),
                                      placeholder="e.g. F/m")
        show_both_errors = st.checkbox(
            "Also show the error without the correlation terms",
            value=remember("show_both_errors", True),
            help="The diagonal-only version is what the textbook formula gives "
                 "when it assumes the parameters are independent. Shown for "
                 "comparison; the full one is the correct value.",
        )
        ui.update(derived_formula=derived_formula, derived_label=derived_label,
                  derived_units=derived_units, show_both_errors=show_both_errors)

        derived = None
        if derived_formula.strip():
            try:
                derived = derived_quantity(
                    result, derived_formula, derived_label, derived_units
                )
            except Exception as exc:                  # noqa: BLE001 - shown to user
                st.error(f"Could not evaluate that: {exc}")
            else:
                st.success(f"{derived}   (full covariance — use this one)")
                if show_both_errors:
                    simple = derived_quantity(
                        result, derived_formula, derived_label, derived_units,
                        use_covariance=False,
                    )
                    st.info(f"{simple}   (diagonal only, correlations ignored)")
                    if np.isfinite(simple.error) and np.isfinite(derived.error) \
                            and derived.error > 0:
                        ratio = simple.error / derived.error
                        if abs(ratio - 1) > 0.05:
                            st.caption(
                                f"Ignoring the correlations makes the error "
                                f"**{ratio:.2f}× too {'big' if ratio > 1 else 'small'}** here."
                            )
                        else:
                            st.caption(
                                "The two agree here, so the correlations happen "
                                "not to matter for this particular formula."
                            )

        with st.expander("Correlation between the parameters"):
            st.caption(
                "1 or -1 means the two parameters are interchangeable in this "
                "model and the fit cannot separate them."
            )
            errors = np.sqrt(np.diag(result.cov))
            with np.errstate(divide="ignore", invalid="ignore"):
                correlation = result.cov / np.outer(errors, errors)
            st.dataframe(
                pd.DataFrame(
                    np.round(correlation, 3),
                    index=result.param_names,
                    columns=result.param_names,
                ),
            )


# ----------------------------------------------------- tab 3: report output

with tab_report:
    if result is None:
        st.info("No fit to report yet.")
    else:
        st.markdown("##### Figure for the report")
        report_figure = make_report_figure(result)
        st.pyplot(report_figure)

        # File name for the downloads: letters, digits and underscores only,
        # so it is safe on Windows and inside a LaTeX label.
        stem = re.sub(
            r"[^0-9A-Za-z]+", "_",
            f"{Path(chosen_name).stem}_{y_name or 'y'}_vs_{x_name or 'x'}",
        ).strip("_") or "fit"
        f1, f2, _ = st.columns([1, 1, 3])
        for column, kind in ((f1, "png"), (f2, "pdf")):
            buffer = io.BytesIO()
            report_figure.savefig(buffer, format=kind, dpi=300, bbox_inches="tight",
                                  facecolor="white")
            column.download_button(
                f"Download {kind.upper()}", buffer.getvalue(),
                file_name=f"{stem}.{kind}",
                mime="image/png" if kind == "png" else "application/pdf",
            )
        st.caption("PDF is vector, so it stays sharp at any size in LaTeX.")

        # ---- table of results, in the format you are going to paste into ----
        st.markdown("##### Table of results")
        table_format = st.radio(
            "Format",
            ["Plain text (Word, PowerPoint, Docs)", "LaTeX"],
            horizontal=True,
            index=0 if remember("table_format", "plain") == "plain" else 1,
            label_visibility="collapsed",
        )
        ui["table_format"] = "plain" if table_format.startswith("Plain") else "latex"

        if ui["table_format"] == "plain":
            lines = ["Parameter\tValue\tError\tRelative error"]
            for i, (value, error) in enumerate(zip(result.params, result.param_errors)):
                v, u = round_to_uncertainty(value, error)
                lines.append(f"a{i}\t{v}\t{u}\t{format_percent(value, error)}")
            lines += [
                f"chi2_red\t{result.chi2_reduced:.3g}",
                f"p-value\t{plain_number(result.p_value, 3)}",
                f"dof\t{result.dof}",
            ]
            st.code("\n".join(lines), language=None)
            st.caption(
                "Tab-separated. Pasting this into Word or PowerPoint and choosing "
                "*Convert text to table* gives you a real table."
            )
        else:
            st.code(
                latex_table(
                    result,
                    caption=f"Fit parameters for {title}",
                    label=latex_label(stem),
                    param_labels=[f"$a_{i}$" for i in range(result.n_params)],
                ),
                language="latex",
            )

        # ---- methods -----------------------------------------------------
        st.markdown("##### Methods")
        methods_language = st.radio(
            "Language",
            ["עברית", "English"],
            horizontal=True,
            index=0 if remember("methods_lang", "he") == "he" else 1,
            label_visibility="collapsed",
        )
        lang = "he" if methods_language == "עברית" else "en"
        ui["methods_lang"] = lang

        st.caption(
            "What the program actually did, in sentences. Paste it into the "
            "report, or read it off when you are asked how the analysis was done."
        )

        excluded_notes = [
            (row, float(x_all[row]), float(y_all[row]), reasons.get(str(row), ""))
            for row in sorted(excluded)
        ]
        blocks = methods_blocks(
            result,
            lang=lang,
            x_label=axis_label(x_name, x_units),
            y_label=axis_label(y_name, y_units),
            source=(
                source_block(chosen_name, sheet or "", (x_col, dx_col, y_col, dy_col))
                if lang == "he" else
                f"`{chosen_name}`"
                + (f", sheet `{sheet}`" if sheet else "")
                + f", columns {x_col} / {dx_col} / {y_col} / {dy_col}."
            ),
            excluded_notes=excluded_notes,
        )

        if derived is not None:
            measurement = format_measurement(derived.value, derived.error)
            units_text = f" {derived.units}" if derived.units else ""
            if lang == "he":
                blocks += [
                    ("text", f"**גודל נגזר.** {derived.label or 'התוצאה'} חושב "
                             "מפרמטרי ההתאמה לפי:"),
                    ("formula", f"{derived.label or 'g'} = {derived.formula}\n"
                                f"{derived.label or 'g'} = {measurement}{units_text}"),
                    ("text", "השגיאה חושבה כשגיאה עקיפה דרך מטריצת השונות "
                             "המשותפת המלאה:"),
                    ("formula", "sigma^2 = sum_ij (dg/da_i)(dg/da_j) * C_ij"),
                    ("text", "כלומר כולל איברי המתאם שבין הפרמטרים, ולא רק "
                             "האלכסון."),
                ]
            else:
                blocks += [
                    ("text", f"**Derived quantity.** {derived.label or 'The result'} "
                             f"was computed from the fit parameters as "
                             f"`{derived.formula}`, giving {measurement}{units_text}. "
                             "Its uncertainty was propagated through the full "
                             "covariance matrix,"),
                    ("formula", "sigma^2 = sum_ij (dg/da_i)(dg/da_j) * C_ij"),
                    ("text", "so the correlations between parameters are included."),
                ]

        render_blocks(blocks, rtl=(lang == "he"))

        with st.expander("Copy as plain text"):
            st.caption(
                "No ** markers and no code fences -- pastes cleanly into Word, "
                "PowerPoint or an email."
            )
            st.code(blocks_to_text(blocks, markdown=False), language=None)
        with st.expander("Copy as Markdown"):
            st.code(blocks_to_text(blocks, markdown=True), language="markdown")

save_state()
