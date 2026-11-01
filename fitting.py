"""
fitting.py -- statistics core for the Lab B data analysis tool.

This module does the physics/statistics and nothing else: no user interface.
You can import it from a plain script or a notebook if you ever want to.

The fitting method is Orthogonal Distance Regression (ODR), which accounts for
uncertainty in BOTH x and y -- the same method used last year.

    from fitting import load_table, run_fit
    df = load_table("data/rotating_magnet_emf.csv")
    res = run_fit(x, dx, y, dy, "a0 + a1*x")
    print(res.summary_lines())
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pandas as pd
from scipy import stats

# scipy.odr is deprecated from scipy 1.17 and scheduled for removal in 1.19.
# It still works and is still the standard tool for this, but the warning is
# noise during a lab session. requirements.txt pins scipy below 1.19 so the
# import cannot vanish underneath you mid-semester; revisit if that changes.
import warnings as _warnings
with _warnings.catch_warnings():
    _warnings.simplefilter("ignore", DeprecationWarning)
    from scipy.odr import ODR, Model, RealData

# --------------------------------------------------------------------------
# 1. Loading data files
# --------------------------------------------------------------------------

EXCEL_SUFFIXES = {".xlsx", ".xlsm", ".xls", ".xltx", ".ods"}
TEXT_SUFFIXES = {".csv", ".tsv", ".txt", ".dat"}
DATA_SUFFIXES = EXCEL_SUFFIXES | TEXT_SUFFIXES

#: Which reader each spreadsheet format needs. Naming the engine explicitly
#: keeps pandas from having to guess, which is where it used to trip over a
#: half-imported openpyxl (see the eager import below).
EXCEL_ENGINES = {
    ".xlsx": "openpyxl",
    ".xlsm": "openpyxl",
    ".xltx": "openpyxl",
    ".xls": "xlrd",
    ".ods": "odf",
}

# Import the spreadsheet readers HERE, once, while the program is starting and
# nothing else is running. pandas imports them lazily, in the middle of reading
# a file, and Streamlit reruns the script on a worker thread -- so two reruns
# could start importing openpyxl at the same moment. The loser of that race
# sees a module that is in sys.modules but not finished, and pandas' engine
# detection then fails with
#
#   AttributeError: partially initialized module 'openpyxl' has no attribute
#   'Workbook' (most likely due to a circular import)
#
# A broken module also stays in sys.modules, so every later read fails too.
# Importing up front means the module is complete before pandas ever looks at
# it, and a genuinely broken install is reported now, clearly, instead of as a
# confusing traceback later.
EXCEL_IMPORT_ERRORS: dict[str, str] = {}
for _engine, _module in (("openpyxl", "openpyxl"), ("xlrd", "xlrd"), ("odf", "odf")):
    try:
        __import__(_module)
    except Exception as _exc:                     # noqa: BLE001 - recorded, not raised
        EXCEL_IMPORT_ERRORS[_engine] = f"{type(_exc).__name__}: {_exc}"


def is_excel(path: str | Path) -> bool:
    return Path(path).suffix.lower() in EXCEL_SUFFIXES


def excel_engine(path: str | Path) -> str:
    """The reader for this spreadsheet, or a clear error if it is unusable."""
    suffix = Path(path).suffix.lower()
    engine = EXCEL_ENGINES.get(suffix)
    if engine is None:
        raise ValueError(f"'{suffix}' is not a spreadsheet format this tool reads.")
    if engine in EXCEL_IMPORT_ERRORS:
        raise ImportError(
            f"The reader for {suffix} files ({engine}) will not load on this "
            f"computer.\n\n{EXCEL_IMPORT_ERRORS[engine]}\n\n"
            "Reinstalling it usually fixes this. Close the app, then run:\n"
            f'    "%LOCALAPPDATA%\\labB-fitting\\venv\\Scripts\\python.exe" '
            f"-m pip install --force-reinstall {engine}\n\n"
            "Saving the file as CSV also works -- CSV needs no extra reader."
        )
    return engine


def list_sheets(path: str | Path) -> list[str]:
    """Real sheet names inside an Excel file. Empty list for a CSV."""
    if not is_excel(path):
        return []
    with pd.ExcelFile(path, engine=excel_engine(path)) as xl:
        return list(xl.sheet_names)


def load_table(path: str | Path, sheet: str | int | None = None) -> pd.DataFrame:
    """Read a data file into a DataFrame, picking the reader by file extension.

    This is the fix for last year's bug, where load_columns_from_excel called
    pd.read_excel on a path that was actually a .csv.
    """
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix in EXCEL_SUFFIXES:
        return pd.read_excel(
            path,
            sheet_name=sheet if sheet is not None else 0,
            engine=excel_engine(path),
        )

    if suffix == ".tsv":
        return pd.read_csv(path, sep="\t")

    if suffix in TEXT_SUFFIXES:
        # sep=None + engine="python" sniffs comma / semicolon / whitespace,
        # so a file exported from Excel in a locale that uses ';' still works.
        return pd.read_csv(path, sep=None, engine="python")

    raise ValueError(
        f"Don't know how to read '{path.name}'. "
        f"Supported: {', '.join(sorted(DATA_SUFFIXES))}"
    )


def find_data_files(folder: str | Path) -> list[Path]:
    """Every data file in a folder, newest first, so today's export is on top."""
    folder = Path(folder)
    if not folder.is_dir():
        return []
    files = [
        p for p in folder.iterdir()
        if p.is_file()
        and p.suffix.lower() in DATA_SUFFIXES
        # ~$ is Excel's lock file, . is hidden, and a readme is a note to the
        # reader rather than a measurement -- none of them belong in the list
        and not p.name.startswith(("~$", "."))
        and not p.stem.lower().startswith("readme")
    ]
    return sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)


def numeric_columns(df: pd.DataFrame) -> list[str]:
    """Columns that hold numbers, by their real header text."""
    out = []
    for col in df.columns:
        series = pd.to_numeric(df[col], errors="coerce")
        if series.notna().sum() >= 2:
            out.append(str(col))
    return out


def split_header(header: str) -> tuple[str, str]:
    """Split a column header into a name and its units.

    'w [rad/s]' -> ('w', 'rad/s');  'B (T)' -> ('B', 'T');  'angle' -> ('angle', '')

    Lab files almost always carry the units in the header already, so the
    axis labels can fill themselves in instead of being typed -- and, more
    importantly, instead of being left over from the previous file.
    """
    text = str(header).strip()
    match = re.search(r"^(.*?)\s*[\[(]([^\])]*)[\])]\s*$", text)
    if match and match.group(1).strip():
        return match.group(1).strip(), match.group(2).strip()
    return text, ""


def column_values(df: pd.DataFrame, column: str) -> np.ndarray:
    return pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=float)


# --------------------------------------------------------------------------
# 2. Turning a typed formula into a fit model
# --------------------------------------------------------------------------

# Only these names may appear in a formula. Everything else is rejected, which
# turns a typo into a clear message instead of a confusing crash.
SAFE_NAMES: dict[str, object] = {
    "sin": np.sin, "cos": np.cos, "tan": np.tan,
    "arcsin": np.arcsin, "arccos": np.arccos, "arctan": np.arctan,
    "asin": np.arcsin, "acos": np.arccos, "atan": np.arctan,
    "sinh": np.sinh, "cosh": np.cosh, "tanh": np.tanh,
    "exp": np.exp, "log": np.log, "ln": np.log,
    "log10": np.log10, "log2": np.log2,
    "sqrt": np.sqrt, "abs": np.abs, "absolute": np.abs,
    "sign": np.sign, "floor": np.floor, "ceil": np.ceil,
    "pi": np.pi, "e": np.e,
    "where": np.where, "minimum": np.minimum, "maximum": np.maximum,
}

PARAM_PATTERN = re.compile(r"\ba(\d+)\b")

#: Ready-made formulas, so the common cases are one paste away.
FORMULA_PRESETS: dict[str, str] = {
    "Linear":                      "a0 + a1*x",
    "Line through the origin":     "a0*x",
    "Parabola":                    "a0 + a1*x + a2*x**2",
    "Power law":                   "a0 * x**a1",
    "Exponential decay":           "a0 + a1*exp(-x/a2)",
    "Sine":                        "a0 + a1*sin(a2*x + a3)",
    "Cosine":                      "a0 + a1*cos(x + a2)",
    "Inverse / optics (1/f)":      "a0 + a1*x/(x - a1)",
    "RLC amplitude":               "a0/sqrt(1 + (a1*x - 1/(a2*x))**2) + a3",
    "RLC phase":                   "arctan(a0*x - 1/(a1*x)) + a2",
}


class FormulaError(ValueError):
    """Raised when a typed formula cannot be used as a fit model."""


@dataclass
class ParsedFormula:
    text: str
    n_params: int
    func: Callable[[Sequence[float], np.ndarray], np.ndarray]

    @property
    def param_names(self) -> list[str]:
        return [f"a{i}" for i in range(self.n_params)]


def parse_formula(formula: str) -> ParsedFormula:
    """Compile a formula like 'a0 + a1*x' into a function f(A, x).

    Parameters are written a0, a1, a2, ... and are discovered automatically,
    so the number of fit parameters comes from the formula itself. This is why
    you never need to pass initial_guesses of the wrong length again.
    """
    text = (formula or "").strip()
    if not text:
        raise FormulaError("The formula is empty.")

    # Convenience: accept '^' for powers, the way you'd write it on paper.
    text = text.replace("^", "**")

    indices = sorted({int(m) for m in PARAM_PATTERN.findall(text)})
    if not indices:
        raise FormulaError(
            "No fit parameters found. Name them a0, a1, a2, ... "
            "for example:  a0 + a1*x"
        )
    if indices[0] != 0 or indices != list(range(len(indices))):
        missing = [i for i in range(max(indices) + 1) if i not in indices]
        raise FormulaError(
            "Parameters must start at a0 with no gaps. "
            f"Missing: {', '.join('a' + str(i) for i in missing)}"
        )
    n_params = len(indices)

    try:
        code = compile(text, "<formula>", "eval")
    except SyntaxError as exc:
        raise FormulaError(f"That isn't valid syntax: {exc.msg}") from exc

    allowed = set(SAFE_NAMES) | {"x"} | {f"a{i}" for i in indices}
    unknown = [name for name in code.co_names if name not in allowed]
    if unknown:
        raise FormulaError(
            f"Unknown name(s): {', '.join(sorted(set(unknown)))}. "
            f"Use x, parameters a0..a{n_params - 1}, and functions like "
            "sin, cos, exp, log, sqrt, arctan."
        )

    def func(A: Sequence[float], x: np.ndarray) -> np.ndarray:
        scope = {f"a{i}": A[i] for i in range(n_params)}
        scope["x"] = x
        with np.errstate(all="ignore"):
            value = eval(code, {"__builtins__": {}}, {**SAFE_NAMES, **scope})
        return np.asarray(value, dtype=float) * np.ones_like(np.asarray(x, dtype=float))

    return ParsedFormula(text=text, n_params=n_params, func=func)


# --------------------------------------------------------------------------
# 3. Rounding to the uncertainty
# --------------------------------------------------------------------------

def round_to_uncertainty(value: float, uncertainty: float, sig: int = 2) -> tuple[str, str]:
    """Round a value so it ends at the same decimal place as its uncertainty.

    Reports the uncertainty to `sig` significant digits (2 is the usual lab
    convention) and the value to the matching precision.

        >>> round_to_uncertainty(9.8123, 0.0456)
        ('9.812', '0.046')

    Last year's version broke when rounding pushed the uncertainty into the
    next decade (0.0996 -> 0.10, which is 2 digits at a different place).
    This one re-checks the magnitude after rounding.
    """
    if not np.isfinite(value) or not np.isfinite(uncertainty):
        return (f"{value:g}" if np.isfinite(value) else "nan"), "nan"
    if uncertainty == 0:
        return f"{value:g}", "0"

    magnitude = math.floor(math.log10(abs(uncertainty)))
    decimals = sig - 1 - magnitude
    rounded_unc = round(uncertainty, decimals)

    # Re-check: 0.0996 rounds to 0.10, whose magnitude moved up by one.
    new_magnitude = math.floor(math.log10(abs(rounded_unc)))
    if new_magnitude != magnitude:
        decimals = sig - 1 - new_magnitude
        rounded_unc = round(uncertainty, decimals)

    if decimals > 0:
        return f"{value:.{decimals}f}", f"{rounded_unc:.{decimals}f}"

    # Uncertainty is 1 or larger: no decimal point, and the value is rounded
    # to the same place (e.g. 1234 +/- 56 at decimals=-1 -> 1230 +/- 60).
    return f"{round(value, decimals):.0f}", f"{rounded_unc:.0f}"


#: Outside this range of exponents, a plain decimal becomes unreadable
#: (0.0000000000000000000032) and the value is written as a power of ten.
SCIENTIFIC_LOW, SCIENTIFIC_HIGH = -3, 5


def _exponent(value: float, uncertainty: float) -> int:
    reference = abs(value) if value != 0 and np.isfinite(value) else abs(uncertainty)
    if reference == 0 or not np.isfinite(reference):
        return 0
    return math.floor(math.log10(reference))


def format_measurement(
    value: float,
    uncertainty: float,
    sig: int = 2,
    latex: bool = False,
    force_scientific: bool | None = None,
) -> str:
    """'9.812 +/- 0.046', or '(1.600 +/- 0.032) x 10^-19' when that is clearer.

    Physics constants live at exponents where a plain decimal is unreadable, so
    very large and very small numbers are written with a shared power of ten --
    the way they belong in the report.
    """
    if not np.isfinite(value):
        return "nan"

    exponent = _exponent(value, uncertainty)
    use_scientific = (
        force_scientific
        if force_scientific is not None
        else not (SCIENTIFIC_LOW <= exponent <= SCIENTIFIC_HIGH)
    )

    if not use_scientific:
        v, u = round_to_uncertainty(value, uncertainty, sig)
        return f"${v} \\pm {u}$" if latex else f"{v} ± {u}"

    scale = 10.0 ** exponent
    v, u = round_to_uncertainty(value / scale, uncertainty / scale, sig)
    if latex:
        return f"$({v} \\pm {u}) \\times 10^{{{exponent}}}$"
    return f"({v} ± {u})·10^{exponent}"


def format_percent(value: float, uncertainty: float) -> str:
    """Relative error as something readable: '190%', '2.3%', '0.04%'.

    Plain '%.2g' formatting turns 190 into '1.9e+02', which is nobody's idea
    of a percentage.
    """
    pct = relative_percent(value, uncertainty)
    if not np.isfinite(pct):
        return "--"
    if pct >= 1000:
        return f"{pct:,.0f}%"          # hopeless parameter, but say so plainly
    if pct >= 10:
        return f"{pct:.0f}%"
    if pct >= 1:
        return f"{pct:.1f}%"
    if pct >= 0.01:
        return f"{pct:.2f}%"
    return f"{pct:.1g}%"


def relative_percent(value: float, uncertainty: float) -> float:
    """Uncertainty as a percentage of the value; nan if the value is 0."""
    if value == 0 or not np.isfinite(value) or not np.isfinite(uncertainty):
        return float("nan")
    return abs(100.0 * uncertainty / value)


# --------------------------------------------------------------------------
# 4. The fit itself
# --------------------------------------------------------------------------

@dataclass
class FitResult:
    """Everything that came out of one fit."""

    formula: str
    params: np.ndarray
    param_errors: np.ndarray
    cov: np.ndarray

    # the data actually fitted (excluded points already removed)
    x: np.ndarray
    dx: np.ndarray
    y: np.ndarray
    dy: np.ndarray

    # residuals, three ways -- see the module docstring for why
    residuals: np.ndarray            # y - f(x), the vertical distance
    residual_errors: np.ndarray      # sqrt(dy^2 + (f'(x) dx)^2), the combined error
    odr_residuals: np.ndarray        # ODR's own eps, consistent with x shifted by delta
    x_shifts: np.ndarray             # ODR's delta: how far each x was moved

    chi2: float                      # sum (residual / combined error)^2
    dof: int
    chi2_reduced: float
    p_value: float
    odr_sum_square: float            # ODR's internal cost, for comparison

    initial_guess: np.ndarray
    guess_source: str
    cov_raw: np.ndarray | None = None   # ODR's unscaled cov_beta, kept for reference
    n_excluded: int = 0
    warnings: list[str] = field(default_factory=list)
    _func: Callable | None = None

    # ---- derived quantities -------------------------------------------

    @property
    def n_points(self) -> int:
        return len(self.x)

    @property
    def n_params(self) -> int:
        return len(self.params)

    @property
    def param_names(self) -> list[str]:
        return [f"a{i}" for i in range(self.n_params)]

    def model(self, x: np.ndarray) -> np.ndarray:
        """The fitted curve evaluated anywhere."""
        if self._func is None:
            raise RuntimeError("This FitResult has no model function attached.")
        return self._func(self.params, np.asarray(x, dtype=float))

    def curve(self, n: int = 400) -> tuple[np.ndarray, np.ndarray]:
        """Smooth (x, y) arrays for drawing the fit line."""
        lo, hi = float(np.min(self.x)), float(np.max(self.x))
        pad = 0.02 * (hi - lo) if hi > lo else 1.0
        xs = np.linspace(lo - pad, hi + pad, max(n, 10 * self.n_points))
        return xs, self.model(xs)

    @property
    def normalized_residuals(self) -> np.ndarray:
        """Residual divided by its own error -- should scatter within +/-2."""
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(
                self.residual_errors > 0,
                self.residuals / self.residual_errors,
                np.nan,
            )

    def n_sigma(self, index: int, expected: float, expected_error: float = 0.0) -> float:
        """How many combined standard deviations a parameter sits from theory.

        N = |a - a_theory| / sqrt(sigma_a^2 + sigma_theory^2)

        Below about 2 the measurement agrees with theory; 2-3 is marginal;
        above 3 it disagrees and the report needs to explain why.
        """
        spread = math.hypot(self.param_errors[index], expected_error)
        if spread == 0:
            return float("inf") if self.params[index] != expected else 0.0
        return abs(self.params[index] - expected) / spread

    def summary_lines(self) -> list[str]:
        lines = [
            f"{name}: {format_measurement(v, e)}  ({format_percent(v, e)})"
            for name, v, e in zip(self.param_names, self.params, self.param_errors)
        ]
        lines += [
            f"chi2 = {self.chi2:.4g}   dof = {self.dof}",
            f"chi2_reduced = {self.chi2_reduced:.4g}",
            f"p-value = {self.p_value:.4g}",
        ]
        return lines


def _derivative(func: Callable, params: Sequence[float], x: np.ndarray) -> np.ndarray:
    """df/dx by central difference, used to propagate the x-errors into y."""
    x = np.asarray(x, dtype=float)
    step = 1e-6 * np.maximum(np.abs(x), 1.0)
    with np.errstate(all="ignore"):
        forward = func(params, x + step)
        backward = func(params, x - step)
    slope = (forward - backward) / (2.0 * step)
    return np.nan_to_num(slope, nan=0.0, posinf=0.0, neginf=0.0)


def _clean_errors(err: np.ndarray, reference: np.ndarray, label: str,
                  warnings: list[str]) -> np.ndarray:
    """ODR needs strictly positive errors. Replace zeros, and say so."""
    err = np.abs(np.asarray(err, dtype=float))
    bad = ~np.isfinite(err) | (err <= 0)
    if bad.any():
        scale = np.nanmax(np.abs(reference)) or 1.0
        err = err.copy()
        err[bad] = 1e-9 * scale
        warnings.append(
            f"{bad.sum()} point(s) had a zero or missing {label} error; "
            f"substituted a negligible value ({1e-9 * scale:.2g}) so ODR could run. "
            "A real measurement always has an uncertainty -- check the data file."
        )
    return err


def _candidate_guesses(user_guess, n_params, x, y, warnings) -> list[tuple[np.ndarray, str]]:
    """Starting points to try, best-effort, so a bad guess doesn't kill the fit."""
    candidates: list[tuple[np.ndarray, str]] = []

    if user_guess is not None:
        guess = np.asarray(user_guess, dtype=float).ravel()
        if guess.size == n_params:
            candidates.append((guess, "the initial guess you entered"))
        else:
            warnings.append(
                f"The initial guess had {guess.size} value(s) but the formula has "
                f"{n_params} parameter(s); it was ignored."
            )

    candidates.append((np.ones(n_params), "all ones (automatic)"))

    # A straight-line guess is a surprisingly good start for many models.
    finite = np.isfinite(x) & np.isfinite(y)
    if finite.sum() >= 2 and np.ptp(x[finite]) > 0:
        slope, intercept = np.polyfit(x[finite], y[finite], 1)
        poly = np.ones(n_params)
        poly[0] = intercept
        if n_params > 1:
            poly[1] = slope
        candidates.append((poly, "a straight-line pre-fit (automatic)"))

    scale = float(np.nanmean(np.abs(y[finite]))) if finite.any() else 1.0
    if scale and np.isfinite(scale):
        candidates.append((np.full(n_params, scale), f"the data scale, {scale:.3g} (automatic)"))
    candidates.append((np.full(n_params, 0.1), "all 0.1 (automatic)"))

    return candidates


def run_fit(
    x, dx, y, dy,
    formula: str,
    initial_guess: Sequence[float] | None = None,
    exclude: Sequence[bool] | None = None,
    try_multiple_starts: bool = True,
    max_iterations: int = 2000,
) -> FitResult:
    """Fit `formula` to the data with ODR and compute the usual statistics.

    `exclude` is a boolean array, True for points to leave out of the fit.
    Excluded points are removed here; the caller still has them for plotting.
    """
    warnings: list[str] = []

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    dx = np.asarray(dx, dtype=float)
    dy = np.asarray(dy, dtype=float)

    lengths = {len(x), len(dx), len(y), len(dy)}
    if len(lengths) != 1:
        raise ValueError(f"The four columns have different lengths: {sorted(lengths)}")

    n_total = len(x)
    keep = np.ones(n_total, dtype=bool)
    if exclude is not None:
        keep &= ~np.asarray(exclude, dtype=bool)

    # Rows where any of the four values is missing cannot be fitted.
    complete = np.isfinite(x) & np.isfinite(y) & np.isfinite(dx) & np.isfinite(dy)
    n_incomplete = int((keep & ~complete).sum())
    if n_incomplete:
        warnings.append(
            f"{n_incomplete} row(s) were skipped because a value was blank "
            "or not a number."
        )
    keep &= complete

    x_fit, dx_fit, y_fit, dy_fit = x[keep], dx[keep], y[keep], dy[keep]
    n_excluded = n_total - int(keep.sum())

    parsed = parse_formula(formula)
    if len(x_fit) <= parsed.n_params:
        raise ValueError(
            f"Only {len(x_fit)} usable point(s) for {parsed.n_params} parameter(s). "
            "A fit needs more points than parameters."
        )

    dx_safe = _clean_errors(dx_fit, x_fit, "x", warnings)
    dy_safe = _clean_errors(dy_fit, y_fit, "y", warnings)

    model = Model(parsed.func)
    data = RealData(x_fit, y_fit, sx=dx_safe, sy=dy_safe)

    guesses = _candidate_guesses(initial_guess, parsed.n_params, x_fit, y_fit, warnings)
    if not try_multiple_starts:
        guesses = guesses[:1]

    best = None
    failures = []
    for guess, source in guesses:
        try:
            odr = ODR(data, model, beta0=np.asarray(guess, dtype=float), maxit=max_iterations)
            output = odr.run()
        except Exception as exc:                       # noqa: BLE001 - report, don't crash
            failures.append(f"{source}: {exc}")
            continue
        cost = float(output.sum_square)
        if not np.isfinite(cost) or not np.all(np.isfinite(output.beta)):
            failures.append(f"{source}: the fit did not converge to finite values")
            continue
        if best is None or cost < best[0]:
            best = (cost, output, np.asarray(guess, dtype=float), source)

    if best is None:
        raise RuntimeError(
            "The fit failed from every starting point.\n  "
            + "\n  ".join(failures)
            + "\nUsually this means the formula doesn't match the data, or a "
              "parameter needs a sensible initial guess."
        )

    _, output, guess_used, guess_source = best
    if guess_source.endswith("(automatic)") and initial_guess is not None:
        warnings.append(
            f"Your initial guess didn't give the best fit; the solver started from "
            f"{guess_source} instead."
        )

    params = np.asarray(output.beta, dtype=float)
    param_errors = np.asarray(output.sd_beta, dtype=float)

    # ODR returns cov_beta UNSCALED: sd_beta = sqrt(diag(cov_beta) * res_var),
    # where res_var = sum_square / dof. So cov_beta on its own is inconsistent
    # with the quoted parameter errors, and propagating an error through the
    # raw matrix gives the wrong answer. Rescale it once, here, so that
    # sqrt(diag(cov)) == sd_beta and any derived quantity comes out right.
    cov_raw = np.asarray(output.cov_beta, dtype=float)
    _dof = len(x_fit) - parsed.n_params
    res_var = (float(output.sum_square) / _dof) if _dof > 0 else 1.0
    cov = cov_raw * res_var

    # --- residuals and chi-squared ------------------------------------
    #
    # The residual is the vertical distance y - f(x). Its uncertainty is NOT
    # just dy: an error in x also moves the curve, by roughly f'(x)*dx. The
    # two add in quadrature, and that combined error is what the residual
    # must be compared against. Dividing by dy alone (last year's version)
    # makes a good fit look bad whenever the x-errors matter.
    residuals = y_fit - parsed.func(params, x_fit)
    slope = _derivative(parsed.func, params, x_fit)
    residual_errors = np.sqrt(dy_safe**2 + (slope * dx_safe) ** 2)

    with np.errstate(divide="ignore", invalid="ignore"):
        chi2 = float(np.sum((residuals / residual_errors) ** 2))

    dof = len(x_fit) - parsed.n_params
    chi2_reduced = chi2 / dof if dof > 0 else float("nan")
    p_value = float(stats.chi2.sf(chi2, dof)) if dof > 0 else float("nan")

    # ODR also reports where it moved each x to (delta) and the matching
    # y-residual (eps). Those are the true perpendicular residuals.
    x_shifts = np.asarray(getattr(output, "delta", np.zeros_like(x_fit)), dtype=float)
    odr_residuals = -np.asarray(getattr(output, "eps", -residuals), dtype=float)

    if dof <= 0:
        warnings.append(
            "Zero degrees of freedom: as many parameters as points. "
            "chi-squared and the p-value are meaningless here."
        )

    return FitResult(
        formula=parsed.text,
        params=params,
        param_errors=param_errors,
        cov=cov,
        cov_raw=cov_raw,
        x=x_fit, dx=dx_fit, y=y_fit, dy=dy_fit,
        residuals=residuals,
        residual_errors=residual_errors,
        odr_residuals=odr_residuals,
        x_shifts=x_shifts,
        chi2=chi2,
        dof=dof,
        chi2_reduced=chi2_reduced,
        p_value=p_value,
        odr_sum_square=float(output.sum_square),
        initial_guess=guess_used,
        guess_source=guess_source,
        n_excluded=n_excluded,
        warnings=warnings,
        _func=parsed.func,
    )


# --------------------------------------------------------------------------
# 5. Report output
# --------------------------------------------------------------------------

def _latex_escape(text: str) -> str:
    for old, new in (("\\", r"\textbackslash{}"), ("_", r"\_"), ("%", r"\%"),
                     ("&", r"\&"), ("#", r"\#"), ("$", r"\$"), ("^", r"\^{}"),
                     ("{", r"\{"), ("}", r"\}"), ("~", r"\~{}")):
        text = text.replace(old, new)
    return text


def latex_number(value: float, sig: int = 2) -> str:
    """A number in LaTeX maths, with powers of ten written properly.

    '7.56e-08' is Python's notation, not a report's. This gives
    $7.6 \\times 10^{-8}$.
    """
    if not np.isfinite(value):
        return "--"
    if value == 0:
        return "$0$"
    exponent = math.floor(math.log10(abs(value)))
    if -3 <= exponent <= 4:
        return f"${value:.{sig}g}$"
    mantissa = value / 10.0 ** exponent
    return f"${mantissa:.{sig}g} \\times 10^{{{exponent}}}$"


def plain_number(value: float, sig: int = 2) -> str:
    """Same idea for plain text: '7.6x10^-8', readable on a slide."""
    if not np.isfinite(value):
        return "--"
    if value == 0:
        return "0"
    exponent = math.floor(math.log10(abs(value)))
    if -3 <= exponent <= 4:
        return f"{value:.{sig}g}"
    return f"{value / 10.0 ** exponent:.{sig}g}·10^{exponent}"


def latex_label(text: str, prefix: str = "tab") -> str:
    """A LaTeX \\label that will not break the document.

    Only letters, digits, hyphens and colons survive. Brackets in particular
    have to go: a label like `tab:b_[t]_vs_angle_[rad]` makes LaTeX read the
    bracket as the start of an optional argument.
    """
    cleaned = re.sub(r"[^0-9A-Za-z]+", "-", text.strip().lower()).strip("-")
    return f"{prefix}:{cleaned}" if cleaned else f"{prefix}:fit"


def latex_table(
    result: FitResult,
    caption: str = "Fit parameters",
    label: str = "tab:fit",
    param_labels: Sequence[str] | None = None,
    units: Sequence[str] | None = None,
) -> str:
    """A ready-to-paste LaTeX table of the parameters and fit quality."""
    rows = []
    for i, (value, error) in enumerate(zip(result.params, result.param_errors)):
        name = param_labels[i] if param_labels and i < len(param_labels) else f"$a_{i}$"
        unit = units[i] if units and i < len(units) else ""
        measurement = format_measurement(value, error, latex=True)
        pct_text = _latex_escape(format_percent(value, error))
        rows.append(
            f"    {_latex_escape(name) if not name.startswith('$') else name} & "
            f"{measurement} & {pct_text} & {_latex_escape(unit)} \\\\"
        )

    return "\n".join([
        r"\begin{table}[h]",
        r"  \centering",
        r"  \begin{tabular}{l c c l}",
        r"    \hline",
        r"    Parameter & Value & Relative error & Units \\",
        r"    \hline",
        *rows,
        r"    \hline",
        f"    $\\chi^2_{{\\mathrm{{red}}}}$ & ${result.chi2_reduced:.3g}$ & & \\\\",
        f"    $p$-value & {latex_number(result.p_value)} & & \\\\",
        f"    Degrees of freedom & ${result.dof}$ & & \\\\",
        r"    \hline",
        r"  \end{tabular}",
        f"  \\caption{{{_latex_escape(caption)}}}",
        f"  \\label{{{latex_label(label.split(':', 1)[-1])}}}",
        r"\end{table}",
    ])


@dataclass
class DerivedQuantity:
    """A physical constant worked out from the fit parameters."""
    formula: str
    value: float
    error: float
    label: str = ""
    units: str = ""

    def __str__(self) -> str:
        name = self.label or "result"
        unit = f" {self.units}" if self.units else ""
        return f"{name} = {format_measurement(self.value, self.error)}{unit}"


def derived_quantity(
    result: FitResult,
    formula: str,
    label: str = "",
    units: str = "",
    use_covariance: bool = True,
) -> DerivedQuantity:
    """Compute a quantity from the fit parameters, with the error propagated.

    Write the formula in terms of a0, a1, ... -- for example, if the slope a1
    equals 1/(2*pi*f) then the formula for f is '1/(2*pi*a1)'.

    The uncertainty uses the FULL covariance matrix,

        sigma_g^2 = sum_ij (dg/da_i)(dg/da_j) C_ij

    not just the diagonal. That matters: fitted parameters are correlated, and
    ignoring the off-diagonal terms can be wrong by a large factor in either
    direction.
    """
    text = (formula or "").strip().replace("^", "**")
    if not text:
        raise FormulaError("The derived-quantity formula is empty.")

    try:
        code = compile(text, "<derived>", "eval")
    except SyntaxError as exc:
        raise FormulaError(f"That isn't valid syntax: {exc.msg}") from exc

    allowed = set(SAFE_NAMES) | set(result.param_names)
    unknown = [n for n in code.co_names if n not in allowed]
    if unknown:
        raise FormulaError(
            f"Unknown name(s): {', '.join(sorted(set(unknown)))}. "
            f"Use the fit parameters {', '.join(result.param_names)}, numbers, "
            "and functions like sqrt, log, pi."
        )

    def evaluate(params: np.ndarray) -> float:
        scope = {name: params[i] for i, name in enumerate(result.param_names)}
        with np.errstate(all="ignore"):
            return float(eval(code, {"__builtins__": {}}, {**SAFE_NAMES, **scope}))

    params = np.asarray(result.params, dtype=float)
    value = evaluate(params)

    # Numerical gradient, stepped by each parameter's own uncertainty so the
    # step is meaningful whatever the scale of the parameter.
    gradient = np.zeros(len(params))
    for i in range(len(params)):
        step = result.param_errors[i] if result.param_errors[i] > 0 else 1e-6
        step = max(step * 1e-3, abs(params[i]) * 1e-8, 1e-12)
        up, down = params.copy(), params.copy()
        up[i] += step
        down[i] -= step
        gradient[i] = (evaluate(up) - evaluate(down)) / (2 * step)

    covariance = np.asarray(result.cov, dtype=float)
    if not use_covariance:
        # The textbook indirect-error formula: diagonal only, which assumes the
        # parameters are independent. They are not, so this is here for
        # comparison with what a course may expect -- not because it is right.
        covariance = np.diag(np.diag(covariance))

    variance = float(gradient @ covariance @ gradient)
    error = math.sqrt(variance) if variance > 0 and np.isfinite(variance) else float("nan")

    return DerivedQuantity(text, value, error, label, units)


def interpret_p_value(p: float) -> str:
    """Plain-language reading of the p-value, in lab-report terms."""
    if not np.isfinite(p):
        return "The p-value could not be computed."
    if p < 0.05:
        return (
            f"p = {p:.3g} is below 0.05: the data scatter more than the quoted "
            "errors allow. Either the model is wrong or the uncertainties are "
            "underestimated."
        )
    if p > 0.95:
        return (
            f"p = {p:.3g} is above 0.95: the points sit closer to the curve than "
            "the errors suggest they should, which usually means the "
            "uncertainties are overestimated."
        )
    return f"p = {p:.3g} lies in the reasonable range, so the model is consistent with the data."


def interpret_chi2(chi2_reduced: float) -> str:
    if not np.isfinite(chi2_reduced):
        return "Reduced chi-squared is undefined (no degrees of freedom)."
    if chi2_reduced < 0.5:
        return (
            f"Reduced chi-squared = {chi2_reduced:.3g}, well below 1 -- the error "
            "bars are probably too large."
        )
    if chi2_reduced > 2:
        return (
            f"Reduced chi-squared = {chi2_reduced:.3g}, well above 1 -- the model "
            "does not describe the data, or the errors are too small."
        )
    return f"Reduced chi-squared = {chi2_reduced:.3g}, close to 1, as a good fit should be."


def methods_text(
    result: FitResult,
    x_label: str = "x",
    y_label: str = "y",
    source: str = "",
    excluded_notes: Sequence[tuple[int, float, float, str]] = (),
) -> str:
    """A prose account of exactly what was done, for the report and for
    answering 'what statistical approach did you use?' out loud."""

    parts: list[str] = []

    if source:
        parts.append(f"**Data.** {source}")

    parts.append(
        f"**Model.** {y_label} was fitted against {x_label} with the function "
        f"`{result.formula}`, where a0...a{result.n_params - 1} are the "
        f"{result.n_params} free parameters. "
        f"{result.n_points} points entered the fit"
        + (
            f"; {result.n_excluded} "
            f"{'point was' if result.n_excluded == 1 else 'points were'} excluded."
            if result.n_excluded else "."
        )
    )

    if excluded_notes:
        listed = "; ".join(
            f"point {i + 1} ({x:.4g}, {y:.4g})" + (f" -- {why}" if why else " -- no reason recorded")
            for i, x, y, why in excluded_notes
        )
        parts.append(f"**Excluded points.** {listed}")

    parts.append(
        "**Fitting method.** Orthogonal Distance Regression (`scipy.odr`), which "
        "minimises the perpendicular distance to the curve and therefore accounts "
        "for the uncertainty in both axes, rather than in y only as ordinary "
        f"least squares does. The solver started from "
        f"[{', '.join(f'{g:.4g}' for g in result.initial_guess)}] -- {result.guess_source}."
    )

    parts.append(
        "**Uncertainties on the parameters.** The standard errors reported by the "
        "fit (`output.sd_beta`). Note that ODR returns its covariance matrix "
        "unscaled: sd_beta = sqrt(diag(cov_beta) x res_var) with "
        "res_var = sum_square/dof, so the raw matrix is not consistent with the "
        "quoted errors. It was rescaled by res_var before use, which makes "
        "sqrt(diag(C)) equal to the quoted errors, and the full rescaled matrix "
        "-- off-diagonal correlations included -- is what any derived quantity "
        "is propagated through."
    )

    parts.append(
        "**Residuals.** The residual of each point is the vertical distance "
        "r = y - f(x). Its uncertainty is not dy alone: an error in x also "
        "displaces the curve, by approximately f'(x)dx, so the two combine in "
        "quadrature as sqrt(dy^2 + (f'(x)dx)^2). The derivative f'(x) was "
        "evaluated numerically at the fitted parameters. This combined error is "
        "what the residual plot shows and what chi-squared uses."
    )

    parts.append(
        f"**Goodness of fit.** chi-squared = sum (r_i / sigma_i)^2 = "
        f"{result.chi2:#.4g} over {result.dof} degrees of freedom "
        f"({result.n_points} points minus {result.n_params} parameters), giving a "
        f"reduced chi-squared of {result.chi2_reduced:.4g} and a p-value of "
        f"{result.p_value:.4g} from the chi-squared survival function. "
        f"{interpret_chi2(result.chi2_reduced)} {interpret_p_value(result.p_value)}"
    )

    parts.append(
        "**A note on chi-squared.** This value is computed explicitly from the "
        "residuals and their combined errors, as defined above. It is not the "
        "same number as ODR's own objective function "
        f"(`output.sum_square` = {result.odr_sum_square:#.4g}), which weights the "
        "perpendicular distances and is what the solver actually minimised. The "
        "two are close for a good fit but are not interchangeable, so the value "
        "quoted in the report is the explicit one."
    )

    parts.append(
        "**Agreement with theory.** Each parameter was compared with its expected "
        "value using N = |a - a_theory| / sqrt(sigma_a^2 + sigma_theory^2). "
        "N below 2 counts as agreement, 2 to 3 as marginal, above 3 as "
        "disagreement that the discussion has to account for."
    )

    parts.append(
        "**Rounding.** Every uncertainty is quoted to two significant digits and "
        "each value is rounded to the same decimal place."
    )

    if result.warnings:
        parts.append("**Warnings raised during this fit.** " + " ".join(result.warnings))

    return "\n\n".join(parts)


# --------------------------------------------------------------------------
# 6. The same thing in Hebrew
# --------------------------------------------------------------------------
#
# Hebrew and formulas do not mix well on one line: a right-to-left sentence
# with 'sqrt(dy^2 + ...)' inside it gets reordered by the bidirectional text
# algorithm and comes out scrambled, especially around brackets and the final
# full stop. So the Hebrew version is built as BLOCKS -- prose and formulas
# kept apart -- and every formula gets a line of its own, where it is laid
# out left-to-right on its own terms.

Block = tuple[str, str]       # ("text", markdown) or ("formula", code)


def source_block(
    file_name: str,
    sheet_name: str = "",
    column_names: Sequence[str] = (),
) -> str:
    """Where the data came from, as a neutral left-to-right block.

    Written as a block rather than a sentence because file names, column
    headers and units are all Latin script: dropped into a Hebrew sentence
    they drag the punctuation around with them and the line comes out
    scrambled.
    """
    lines = [file_name]
    if sheet_name:
        lines.append(f"sheet: {sheet_name}")
    if len(column_names) == 4:
        x, dx, y, dy = column_names
        lines.append(f"x  = {x}")
        lines.append(f"dx = {dx}")
        lines.append(f"y  = {y}")
        lines.append(f"dy = {dy}")
    return "\n".join(lines)


def _he_blocks(
    result: FitResult,
    x_label: str,
    y_label: str,
    source: str,
    excluded_notes: Sequence[tuple[int, float, float, str]],
) -> list[Block]:
    blocks: list[Block] = []
    add = blocks.append

    if source:
        add(("text", "**נתונים.** הקובץ והעמודות שנותחו:"))
        add(("formula", source))

    # The axis labels carry units in brackets ('e0 [V]'), and brackets inside
    # a Hebrew sentence get reordered. They go in the formula block instead.
    add(("text", "**מודל.** הותאמה הפונקציה הבאה:"))
    add(("formula",
         f"y = {y_label}\n"
         f"x = {x_label}\n"
         f"f(x) = {result.formula}"))
    excluded_sentence = (
        f" {result.n_excluded} נקודות הוצאו מההתאמה."
        if result.n_excluded > 1 else
        (" נקודה אחת הוצאה מההתאמה." if result.n_excluded == 1 else "")
    )
    add(("text",
         f"הפרמטרים החופשיים הם a0 עד a{result.n_params - 1}, בסך הכול "
         f"{result.n_params}. מספר הנקודות שנכנסו להתאמה: {result.n_points}."
         + excluded_sentence))

    if excluded_notes:
        listed = "; ".join(
            f"נקודה {i + 1} ({x:.4g}, {y:.4g})"
            + (f" — {why}" if why else " — לא נרשמה סיבה")
            for i, x, y, why in excluded_notes
        )
        add(("text", f"**נקודות שהוצאו.** {listed}"))

    add(("text",
         "**שיטת ההתאמה.** רגרסיה אורתוגונלית, בשם ODR, שבוצעה בעזרת הספרייה "
         "`scipy.odr`. השיטה ממזערת את המרחק הניצב מכל נקודה לעקומה, ולכן "
         "מתחשבת באי־הוודאות בשני הצירים, בניגוד לשיטת הריבועים הפחותים "
         "הרגילה שמתחשבת בשגיאה בציר האנכי בלבד. נקודת ההתחלה של הפותר:"))
    add(("formula",
         "beta0 = [" + ", ".join(f"{g:.4g}" for g in result.initial_guess) + "]"))

    add(("text",
         "**שגיאות הפרמטרים.** נלקחו מהשגיאות הסטנדרטיות שההתאמה מחזירה, "
         "כלומר `sd_beta`. חשוב לציין שהשיטה מחזירה את מטריצת השונות "
         "המשותפת ללא נרמול:"))
    add(("formula", "sd_beta = sqrt( diag(cov_beta) * res_var )\n"
                    "res_var = sum_square / dof"))
    add(("text",
         "כלומר המטריצה הגולמית אינה עקבית עם השגיאות המדווחות. היא נורמלה "
         "ב־`res_var` לפני כל שימוש, כך שהשורש של האלכסון שלה שווה לשגיאות "
         "המדווחות, והמטריצה המנורמלת במלואה — כולל איברי המתאם שמחוץ "
         "לאלכסון — היא זו שדרכה מחושבת כל שגיאה עקיפה."))

    add(("text",
         "**שאריות.** השארית של כל נקודה היא המרחק האנכי בינה לבין העקומה:"))
    add(("formula", "r = y - f(x)"))
    add(("text",
         "אי־הוודאות של השארית אינה dy בלבד. שגיאה בציר x מזיזה גם את "
         "העקומה עצמה, בקירוב ב־f'(x)·dx, ולכן שתי התרומות מצטרפות בחיבור "
         "בריבועים:"))
    add(("formula", "sigma = sqrt( dy^2 + ( f'(x) * dx )^2 )"))
    add(("text",
         "הנגזרת f'(x) חושבה נומרית בערכי הפרמטרים שהתקבלו. זוהי השגיאה "
         "המשולבת שמוצגת בגרף השאריות, והיא זו שנכנסת לחישוב חי בריבוע."))

    add(("text", "**טיב ההתאמה.** חי בריבוע חושב ישירות מהשאריות:"))
    add(("formula",
         f"chi^2 = sum ( r_i / sigma_i )^2 = {result.chi2:#.4g}"))
    add(("text",
         f"מספר דרגות החופש הוא {result.dof} ({result.n_points} נקודות פחות "
         f"{result.n_params} פרמטרים), ומכאן:"))
    add(("formula",
         f"chi^2_red = {result.chi2_reduced:.4g}\n"
         f"p = {plain_number(result.p_value, 3)}"))
    add(("text", _he_interpret_chi2(result.chi2_reduced) + " " +
                 _he_interpret_p(result.p_value)))

    add(("text",
         "**הערה על חי בריבוע.** הערך שלמעלה חושב במפורש מהשאריות ומהשגיאות "
         "המשולבות, כפי שהוגדרו. הוא אינו זהה לפונקציית המטרה הפנימית של ODR"))
    add(("formula", f"output.sum_square = {result.odr_sum_square:#.4g}"))
    add(("text",
         "שמשקללת את המרחקים הניצבים והיא זו שהפותר באמת מזער. השניים קרובים "
         "זה לזה בהתאמה טובה אך אינם בני החלפה, ולכן הערך המדווח הוא המפורש."))

    add(("text",
         "**השוואה לתיאוריה.** כל פרמטר הושווה לערך התיאורטי שלו באמצעות:"))
    add(("formula", "N = |a - a_theory| / sqrt( sigma_a^2 + sigma_theory^2 )"))
    add(("text",
         "ערך N מתחת ל־2 נחשב להסכמה עם התיאוריה, בין 2 ל־3 גבולי, ומעל 3 "
         "אי־הסכמה שהדיון חייב להסביר."))

    add(("text",
         "**עיגול.** כל אי־ודאות מצוינת בשתי ספרות משמעותיות, וכל ערך מעוגל "
         "לאותו מקום עשרוני כמו אי־הוודאות שלו."))

    if result.warnings:
        add(("text", "**אזהרות שעלו בהתאמה.** " + " ".join(result.warnings)))

    return blocks


def _he_interpret_chi2(chi2_reduced: float) -> str:
    if not np.isfinite(chi2_reduced):
        return "חי בריבוע מצומצם אינו מוגדר (אין דרגות חופש)."
    if chi2_reduced < 0.5:
        return (f"חי בריבוע מצומצם יצא {chi2_reduced:.3g}, הרבה מתחת ל־1, "
                "מה שמצביע על כך שפסי השגיאה כנראה גדולים מדי.")
    if chi2_reduced > 2:
        return (f"חי בריבוע מצומצם יצא {chi2_reduced:.3g}, הרבה מעל ל־1, "
                "כלומר המודל אינו מתאר היטב את הנתונים או שהשגיאות קטנות מדי.")
    return (f"חי בריבוע מצומצם יצא {chi2_reduced:.3g}, קרוב ל־1, "
            "כמצופה מהתאמה טובה.")


def _he_interpret_p(p: float) -> str:
    if not np.isfinite(p):
        return "לא ניתן היה לחשב את ערך p."
    text = plain_number(p, 3)
    if p < 0.05:
        return (f"ערך p יצא {text}, מתחת ל־0.05: פיזור הנקודות גדול ממה "
                "שהשגיאות המדווחות מאפשרות. או שהמודל שגוי, או שאי־הוודאויות "
                "הוערכו בחסר.")
    if p > 0.95:
        return (f"ערך p יצא {text}, מעל 0.95: הנקודות קרובות לעקומה יותר ממה "
                "שהשגיאות מרמזות, ולרוב פירוש הדבר שאי־הוודאויות הוערכו ביתר.")
    return f"ערך p יצא {text}, בתחום הסביר, כלומר המודל עקבי עם הנתונים."


def methods_blocks(
    result: FitResult,
    lang: str = "he",
    x_label: str = "x",
    y_label: str = "y",
    source: str = "",
    excluded_notes: Sequence[tuple[int, float, float, str]] = (),
) -> list[Block]:
    """The methods write-up as blocks, so prose and formulas render apart."""
    if lang == "he":
        return _he_blocks(result, x_label, y_label, source, excluded_notes)
    english = methods_text(result, x_label, y_label, source, excluded_notes)
    return [("text", paragraph) for paragraph in english.split("\n\n")]


def blocks_to_text(blocks: Sequence[Block], markdown: bool = True) -> str:
    """Flatten blocks into something copyable.

    markdown=True keeps the ** bold markers and fences the formulas.
    markdown=False strips them, which is what you want when pasting into
    Word or a slide.
    """
    out: list[str] = []
    for kind, content in blocks:
        if kind == "formula":
            if markdown:
                out.append(f"```\n{content}\n```")
            else:
                # indent EVERY line, not just the first
                out.append("\n".join("    " + line for line in content.splitlines()))
        else:
            out.append(content if markdown else content.replace("**", "").replace("`", ""))
    return "\n\n".join(out)
