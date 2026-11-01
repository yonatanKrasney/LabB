# Lab B fitting tool

Fits data with **ODR** (orthogonal distance regression -- uses the errors in
both x and y), plots it, and produces the numbers and figures a lab report
needs. Runs entirely on this laptop; nothing is uploaded.

## Starting it

Double-click **`run.bat`**.

The first run sets up a private Python environment and installs what it needs
(a couple of minutes, once). After that it starts in a few seconds and opens
in your browser. Keep the black window open while you work; closing it stops
the app.

If it says Python was not found, install it from
<https://www.python.org/downloads/windows/> and tick *"Add python.exe to PATH"*
during the install.

## Using it

1. Drop your CSV or Excel file into the **`data`** folder next to `run.bat`.
   No renaming, no uploading. Newest file appears at the top of the list.
   **Browse...** opens the normal Windows folder picker if your data lives
   somewhere else -- note the dialog sometimes opens *behind* the browser.
2. In the sidebar pick the file, the sheet (Excel only), and the four columns
   -- by their real headers, not by number. The axis names and units fill
   themselves in from the headers (`w [rad/s]` becomes name `w`, units
   `rad/s`), and they reset when you change file, so the labels can never
   describe the previous experiment.
3. Type the model in the **Formula** box, in terms of `x` and parameters
   `a0, a1, a2, ...`:

   ```
   a0 + a1*x
   a0*cos(x + a1) + a2
   a0/sqrt(1 + (a1*x - 1/(a2*x))**2) + a3
   ```

   The number of parameters is read off the formula, so there is nothing to
   keep in sync. Available: `+ - * / **` (or `^`), `sin cos tan arctan exp log
   log10 sqrt abs`, `pi`, `e`. The **Start from** dropdown has the common
   models ready to edit.
4. **Click a point on the plot to drop it from the fit. Click it again to put
   it back.** Drag a box or lasso for several at once, or untick rows in the
   table. Dropped points stay on the plot as hollow grey circles -- they are
   never deleted from your file -- and you can record *why* each one was
   dropped, which then appears in the methods text.

Your file, columns, formula and dropped points are remembered between
sessions, in `.labfit_session.json`.

## The three tabs

**Fit and points** -- the interactive plot, the residual plot, and the table.

**Statistics** -- parameters with their errors, reduced chi-squared, p-value,
the with/without comparison when points are excluded, the N-sigma test against
a theoretical value, a derived quantity calculator, and the parameter
correlation matrix. Each section has a **What am I looking at?** box
explaining what the numbers mean. The **Copy for the report** block gives every
value formatted to paste straight into a slide or a document -- hover over a
box and a copy icon appears in its corner.

**Report output** -- the figure to download (PNG or vector PDF), a table of
results in plain text (pastes into Word or PowerPoint) or LaTeX, and the
**Methods** panel: prose describing exactly what the analysis did, generated
from the fit you are actually looking at, **in Hebrew or English**. Useful both
for the report and for answering questions about the method out loud.

The Hebrew version keeps every formula on a line of its own. Hebrew runs
right-to-left and formulas run left-to-right, and mixing them in one sentence
makes the brackets and full stops jump around; separating them avoids that
entirely.

## What the statistics actually do

- **Fit:** `scipy.odr`, which minimises the perpendicular distance to the curve
  and so uses the uncertainty in both axes, unlike ordinary least squares.
- **Residual errors:** `sqrt(dy² + (f'(x)·dx)²)`, not `dy` alone. An error in x
  displaces the curve by about `f'(x)·dx`, and the two add in quadrature.
  Using `dy` alone makes a perfectly good fit look bad whenever the x errors
  matter.
- **chi-squared:** computed explicitly as `sum (r/sigma)²` from those residuals.
  This is deliberately *not* ODR's own `sum_square`, which weights
  perpendicular distances; both numbers are reported so the difference is
  visible.
- **Parameter errors:** `sd_beta`. Note ODR returns `cov_beta` *unscaled* --
  `sd_beta = sqrt(diag(cov_beta) × sum_square/dof)` -- so the raw matrix
  disagrees with the quoted errors. It is rescaled once before any error
  propagation; getting this wrong typically changes a derived error by tens of
  percent.
- **Derived quantities (indirect error):** propagated through the full
  rescaled covariance matrix, `sigma² = Σ_ij (dg/da_i)(dg/da_j) C_ij`. This is
  the standard indirect-error formula plus the cross terms. Fit parameters are
  correlated, so those terms are not optional -- dropping them can be wrong by
  a factor of two or more. A checkbox shows the diagonal-only value alongside,
  for comparison with what a course may expect.
- **N-sigma test:** `|a - a_theory| / sqrt(sigma_a² + sigma_theory²)`.
  Under 2 agrees, 2-3 marginal, over 3 disagrees.
- **Rounding:** uncertainties to two significant digits, values to the matching
  decimal place; very large and very small numbers written as a power of ten.

## The files

| file | what it is |
|---|---|
| `run.bat` | the launcher -- this is the only one you need |
| `app.py` | the interface |
| `fitting.py` | the statistics, with no interface -- importable on its own |
| `plots.py` | the matplotlib figure for the report |
| `data/` | your data files go here |
| `examples/` | two sample files to try it on (point the folder box at them) |

## Sharing it

`.gitignore` already keeps out the two things that should not be shared:
`.labfit_session.json`, which stores the paths on your own computer, and
everything in `data/`, which is your measurements. The code itself contains
no personal information.

`fitting.py` works by itself if you ever want to script something:

```python
from fitting import load_table, column_values, run_fit
df = load_table("data/my_measurement.csv")
x, dx, y, dy = (column_values(df, c) for c in df.columns[:4])
result = run_fit(x, dx, y, dy, "a0 + a1*x")
print("\n".join(result.summary_lines()))
```
