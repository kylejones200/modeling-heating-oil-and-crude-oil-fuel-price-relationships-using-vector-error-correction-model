"""Generated from Jupyter notebook: wti_crude_oil_price_fetcher

Magics and shell lines are commented out. Run with a normal Python interpreter."""


# --- code cell ---

"""
Fetch historical WTI Crude Oil prices from Yahoo Finance.

Ticker: CL=F (NYMEX Crude Oil Futures, front month)

Outputs:
  wti_yf.csv with columns Date, Price

Usage:
  python fetch_wti_yf.py --start 2010-01-01 --end 2025-12-31 --out wti_yf.csv
"""

from pathlib import Path

import pandas as pd
import yfinance as yf


def fetch_wti(start: str, end: str) -> pd.DataFrame:
    ticker = yf.Ticker("CL=F")
    hist = ticker.history(start=start, end=end, auto_adjust=True)
    hist = hist.reset_index()
    hist = hist[["Date", "Close"]].rename(columns={"Close": "Price"})
    return hist


def main():

    df = fetch_wti("2010-01-01", "2025-08-10")
    out_path = Path("out.csv")
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} rows to {out_path}")


if __name__ == "__main__":
    main()


# --- code cell ---

"""
WTI Mountain GIF with Threshold Fill

Input
  CSV with columns: Date, Price (case-insensitive). Date parses with pandas.to_datetime.
  Example header: Date,WTI

Output
  wti_mountain.gif in the same folder.

Usage
  python make_wti_gif.py --csv wti.csv --out wti_mountain.gif
  Optional: --title "WTI Crude Oil Price" --fps 24 --dpi 150

Notes
  The area shows green above $70 and red below $70.
  Horizontal lines mark $70 (US shale price) and $50 (Global average).
"""

import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.dates import AutoDateLocator, ConciseDateFormatter


def _setup_matplotlib():
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 1.1,
            "axes.labelsize": 11,
            "axes.titlesize": 13,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "figure.dpi": 120,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.08,
        }
    )


def _bracket_spines(ax):
    # Nudge left and bottom spines outward for a bracket look
    ax.spines["left"].set_position(("outward", 6))
    ax.spines["bottom"].set_position(("outward", 6))


def load_wti(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Normalize column names
    cols = {c: c.lower() for c in df.columns}
    df.columns = [c.lower() for c in df.columns]
    # Try common price column names
    price_col = None
    for cand in ["price", "wti", "close", "value"]:
        if cand in df.columns:
            price_col = cand
            break
    if price_col is None:
        raise ValueError(
            "Could not find a price column. Expected one of: price, wti, close, value."
        )
    if "date" not in df.columns:
        raise ValueError("Could not find a 'Date' column.")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    # Drop duplicates and obvious bad rows
    df = df.dropna(subset=["date", price_col])
    df = df[~df[price_col].astype(str).str.match(r"^\s*$")]
    df[price_col] = pd.to_numeric(df[price_col], errors="coerce")
    df = df.dropna(subset=[price_col])
    # Optional: downsample very dense daily data to weekly to speed GIF
    # Comment out if you prefer daily.
    if len(df) > 4000:
        df = df.set_index("date").resample("W-FRI").last().reset_index()
    return df.rename(columns={price_col: "price"})


def make_frames(ax, dates, prices, y0, color_hi, color_lo, line_color):
    # Prepare masked arrays relative to threshold y0 ($70)
    above = np.maximum(prices, y0)
    below = np.minimum(prices, y0)
    # Draw the static threshold and annotations
    ax.axhline(y0, lw=1.1, color="0.35", alpha=0.9)
    ax.axhline(50, lw=1.0, color="0.55", ls="--", alpha=0.8)
    ax.text(
        dates[0],
        y0 + 1.2,
        "$70  US shale price",
        va="bottom",
        ha="left",
        fontsize=9,
        color="0.2",
    )
    ax.text(
        dates[0],
        50 + 1.2,
        "$50  Global average",
        va="bottom",
        ha="left",
        fontsize=9,
        color="0.3",
    )
    # Dynamic artists
    fill_hi = ax.fill_between([], [], [], where=[], interpolate=True, alpha=0.55)
    fill_lo = ax.fill_between([], [], [], where=[], interpolate=True, alpha=0.55)
    (line,) = ax.plot([], [], lw=1.2, color=line_color, alpha=0.95)
    return fill_hi, fill_lo, line


def animate_wti(
    df,
    out_path: Path,
    title: str = "WTI Crude Oil Price",
    fps: int = 24,
    dpi: int = 150,
):
    _setup_matplotlib()
    dates = pd.to_datetime(df["date"]).to_numpy()
    prices = df["price"].to_numpy().astype(float)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    _bracket_spines(ax)

    # Axis and labels
    locator = AutoDateLocator()
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(ConciseDateFormatter(locator))
    ax.set_ylabel("USD per barrel")
    ax.set_title(title)

    # Y-limits with a little headroom
    y_min = math.floor(min(prices.min(), 45) / 5) * 5
    y_max = math.ceil(max(prices.max(), 85) / 5) * 5
    ax.set_ylim(y_min, y_max)

    # Colors
    color_hi = "#2ca02c"  # green
    color_lo = "#d62728"  # red
    line_color = "black"

    # Prepare artists
    y0 = 70.0
    fill_hi, fill_lo, line = make_frames(
        ax, dates, prices, y0, color_hi, color_lo, line_color
    )

    # We will update fills each frame. To avoid artist recreation, keep references.
    def init():
        line.set_data([], [])
        return (line,)

    def update(frame):
        # Frame is inclusive end index
        x = dates[: frame + 1]
        y = prices[: frame + 1]

        # Remove previous collections if any
        for coll in list(ax.collections):
            coll.remove()

        # Recreate fills for current window
        ax.fill_between(
            x, y0, y, where=y >= y0, interpolate=True, alpha=0.55, color=color_hi
        )
        ax.fill_between(
            x, y, y0, where=y < y0, interpolate=True, alpha=0.55, color=color_lo
        )

        # Redraw static lines on top
        ax.axhline(y0, lw=1.1, color="0.35", alpha=0.9, zorder=3)
        ax.axhline(50, lw=1.0, color="0.55", ls="--", alpha=0.8, zorder=3)

        # Labels pinned near left edge
        ax.text(
            dates[0],
            y0 + 1.2,
            "$70  US shale price",
            va="bottom",
            ha="left",
            fontsize=9,
            color="0.2",
        )
        ax.text(
            dates[0],
            50 + 1.2,
            "$50  Global average",
            va="bottom",
            ha="left",
            fontsize=9,
            color="0.3",
        )

        line.set_data(x, y)
        return (line,)

    n = len(dates)
    interval_ms = max(12, int(1000 / fps))
    anim = FuncAnimation(
        fig, update, init_func=init, frames=n, interval=interval_ms, blit=False
    )

    writer = PillowWriter(fps=fps)
    anim.save(str(out_path), writer=writer, dpi=dpi)

    # Also save the final static frame as PNG
    plt.savefig(
        str(out_path.with_suffix(".png")), dpi=dpi, bbox_inches="tight", pad_inches=0.08
    )
    plt.close(fig)


def main():

    csv_path = "out.csv"
    out_path = "wti.gig"

    df = load_wti(csv_path)

    animate_wti(df, out_path, title="WTI Crude Oil Price", fps=24, dpi=250)


if __name__ == "__main__":
    main()


# --- code cell ---

# WTI Mountain GIF in a notebook (no CLI)

import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.dates import AutoDateLocator, ConciseDateFormatter

# ---- Config ----
use_yfinance = True  # Set False to load from CSV instead
csv_path = "out.csv"  # Used only if use_yfinance = False
start_date = "2010-01-01"  # yfinance start
end_date = None  # yfinance end; None means up to today
output_gif = "wti_mountain.gif"
output_png = "wti_mountain.png"
title = "WTI Crude Oil Price"
fps = 40
dpi = 180
threshold_hi = 70.0  # US shale price
threshold_lo = 50.0  # Global average


# ---- Style ----
def _setup_matplotlib():
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 1.1,
            "axes.labelsize": 11,
            "axes.titlesize": 13,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "figure.dpi": 120,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.08,
        }
    )


def _bracket_spines(ax):
    ax.spines["left"].set_position(("outward", 6))
    ax.spines["bottom"].set_position(("outward", 6))


# ---- Data ----
def fetch_wti_yf(start: str, end: str | None) -> pd.DataFrame:
    import yfinance as yf

    t = yf.Ticker("CL=F")
    hist = t.history(start=start, end=end, auto_adjust=True)
    hist = hist.reset_index()
    df = hist[["Date", "Close"]].rename(columns={"Date": "date", "Close": "price"})
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna().sort_values("date").reset_index(drop=True)
    return df


def load_wti_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.lower() for c in df.columns]
    if "date" not in df.columns:
        raise ValueError("Need a Date column.")
    price_col = None
    for cand in ("price", "wti", "close", "value"):
        if cand in df.columns:
            price_col = cand
            break
    if price_col is None:
        raise ValueError("Need a price column: price|wti|close|value")
    df["date"] = pd.to_datetime(df["date"])
    df[price_col] = pd.to_numeric(df[price_col], errors="coerce")
    df = df.dropna(subset=["date", price_col]).sort_values("date")
    if len(df) > 4000:
        df = df.set_index("date").resample("W-FRI").last().reset_index()
    return df.rename(columns={price_col: "price"}).reset_index(drop=True)


# ---- Plot + GIF ----
def animate_wti(
    df: pd.DataFrame,
    out_gif: str | Path,
    out_png: str | Path,
    title: str,
    y0: float,
    y_ref: float,
    fps: int,
    dpi: int,
):
    _setup_matplotlib()
    dates = pd.to_datetime(df["date"]).to_numpy()
    prices = df["price"].to_numpy().astype(float)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    _bracket_spines(ax)

    locator = AutoDateLocator()
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(ConciseDateFormatter(locator))
    ax.set_ylabel("USD per barrel")
    ax.set_title(title)

    y_min = math.floor(min(prices.min(), y_ref) / 5) * 5
    y_max = math.ceil(max(prices.max(), y0) / 5) * 5
    ax.set_ylim(y_min, y_max)

    color_hi = "#2ca02c"  # green
    color_lo = "#d62728"  # red
    line_color = "black"

    n = len(dates)

    def init():
        ax.plot([], [], lw=1.2, color=line_color, alpha=0.95)
        return []

    def update(frame):
        ax.clear()
        _bracket_spines(ax)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(ConciseDateFormatter(locator))
        ax.set_ylabel("USD per barrel")
        ax.set_title(title)
        ax.set_ylim(y_min, y_max)

        x = dates[: frame + 1]
        y = prices[: frame + 1]

        ax.fill_between(
            x, y0, y, where=y >= y0, interpolate=True, alpha=0.55, color=color_hi
        )
        ax.fill_between(
            x, y, y0, where=y < y0, interpolate=True, alpha=0.55, color=color_lo
        )

        ax.axhline(y0, lw=1.1, color="0.35", alpha=0.9, zorder=3)
        ax.axhline(y_ref, lw=1.0, color="0.55", ls="--", alpha=0.8, zorder=3)

        ax.text(
            x[0] if len(x) else dates[0],
            y0 + 1.2,
            "$70  US shale price",
            va="bottom",
            ha="left",
            fontsize=9,
            color="0.2",
        )
        ax.text(
            x[0] if len(x) else dates[0],
            y_ref + 1.2,
            "$50  Global average",
            va="bottom",
            ha="left",
            fontsize=9,
            color="0.3",
        )

        ax.plot(x, y, lw=1.2, color=line_color, alpha=0.95)
        return []

    interval_ms = max(12, int(1000 / fps))
    anim = FuncAnimation(
        fig, update, init_func=init, frames=n, interval=interval_ms, blit=False
    )

    writer = PillowWriter(fps=fps)
    anim.save(str(out_gif), writer=writer, dpi=dpi)
    plt.savefig(str(out_png), dpi=dpi)  # save figure file per your preference
    plt.show()
    plt.close(fig)


# ---- Run ----
if use_yfinance:
    df_wti = fetch_wti_yf(start_date, end_date)
else:
    df_wti = load_wti_csv(csv_path)

animate_wti(df_wti, output_gif, output_png, title, threshold_hi, threshold_lo, fps, dpi)
print(f"Wrote {output_gif} and {output_png}")


# --- code cell ---

# Memory-safe WTI mountain GIF in a notebook

import math
from pathlib import Path

# Optional: pip install yfinance imageio if missing
# %pip install yfinance imageio --quiet
import imageio.v2 as imageio
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.dates import AutoDateLocator, ConciseDateFormatter

# ---- Config ----
use_yfinance = True  # Set False to read from CSV
csv_path = "wti.csv"  # Used if use_yfinance == False
start_date = "2010-01-01"
end_date = None  # None = today
out_gif = "wti_mountain.gif"
out_png = "wti_mountain.png"
title = "WTI Crude Oil Price"
fps = 48
dpi = 160
y_shale = 70.0  # US shale price
y_global = 50.0  # Global average
max_frames = 600  # Cap frames to protect RAM/CPU

# ---- Style ----
mpl.rcParams.update(
    {
        "font.family": "serif",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 1.1,
        "axes.labelsize": 11,
        "axes.titlesize": 13,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.dpi": 120,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.08,
    }
)


def _bracket(ax):
    ax.spines["left"].set_position(("outward", 6))
    ax.spines["bottom"].set_position(("outward", 6))


# ---- Data ----
def fetch_wti_yf(start, end):
    import yfinance as yf

    t = yf.Ticker("CL=F")
    df = t.history(start=start, end=end, auto_adjust=True).reset_index()
    df = df[["Date", "Close"]].rename(columns={"Date": "date", "Close": "price"})
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna().sort_values("date").reset_index(drop=True)
    return df


def load_wti_csv(path):
    df = pd.read_csv(path)
    df.columns = [c.lower() for c in df.columns]
    if "date" not in df.columns:
        raise ValueError("Need Date column")
    pcol = next(
        (c for c in ("price", "wti", "close", "value") if c in df.columns), None
    )
    if pcol is None:
        raise ValueError("Need price column: price|wti|close|value")
    df["date"] = pd.to_datetime(df["date"])
    df[pcol] = pd.to_numeric(df[pcol], errors="coerce")
    df = df.dropna(subset=["date", pcol]).sort_values("date")
    return df.rename(columns={pcol: "price"}).reset_index(drop=True)


df = fetch_wti_yf(start_date, end_date) if use_yfinance else load_wti_csv(csv_path)

# Downsample to weekly if very long
if len(df) > 4000:
    df = df.set_index("date").resample("W-FRI").last().dropna().reset_index()

# Frame stride
n = len(df)
stride = max(1, int(math.ceil(n / max_frames)))
df_anim = df.iloc[::stride].copy()
dates = pd.to_datetime(df_anim["date"]).to_numpy()
prices = df_anim["price"].to_numpy(dtype=float)
n_frames = len(df_anim)

# ---- Animate with streaming writer (no frame cache) ----
locator = AutoDateLocator()

fig, ax = plt.subplots(figsize=(10, 4.5))
_b = _bracket


def draw_frame(k):
    ax.clear()
    _b(ax)
    ax.set_title(title)
    ax.set_ylabel("USD per barrel")
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(ConciseDateFormatter(locator))

    x = dates[: k + 1]
    y = prices[: k + 1]

    y_min = math.floor(min(prices.min(), y_global) / 5) * 5
    y_max = math.ceil(max(prices.max(), y_shale) / 5) * 5
    ax.set_ylim(y_min, y_max)

    ax.fill_between(
        x, y_shale, y, where=y >= y_shale, interpolate=True, alpha=0.55, color="#2ca02c"
    )
    ax.fill_between(
        x, y, y_shale, where=y < y_shale, interpolate=True, alpha=0.55, color="#d62728"
    )

    ax.axhline(y_shale, lw=1.1, color="0.35", alpha=0.9, zorder=3)
    ax.axhline(y_global, lw=1.0, color="0.55", ls="--", alpha=0.8, zorder=3)

    if len(x):
        ax.text(
            x[0],
            y_shale + 1.2,
            "$70  US shale price",
            va="bottom",
            ha="left",
            fontsize=9,
            color="0.2",
        )
        ax.text(
            x[0],
            y_global + 1.2,
            "$50  Global average",
            va="bottom",
            ha="left",
            fontsize=9,
            color="0.3",
        )

    ax.plot(x, y, lw=1.2, color="black", alpha=0.95)


# Final PNG first
draw_frame(n_frames - 1)
plt.savefig(out_png, dpi=dpi)
plt.show()

# GIF with streaming writer
with imageio.get_writer(out_gif, mode="I", fps=fps) as w:
    for k in range(n_frames):
        draw_frame(k)
        fig.canvas.draw()
        frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        w.append_data(frame)

plt.close(fig)
print(f"Wrote {out_png} and {out_gif}")


# --- code cell ---

# Memory-safe WTI mountain GIF in a notebook (buffer-based frames)

import io
import math

import imageio.v2 as imageio
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.dates import AutoDateLocator, ConciseDateFormatter

# %pip install yfinance imageio --quiet

use_yfinance = True
csv_path = "wti.csv"
start_date = "2022-01-01"
end_date = None
out_gif = "wti_mountain.gif"
out_png = "wti_mountain.png"
title = "WTI Crude Oil Price"
fps = 24
dpi = 110
y_shale = 70.0
y_global = 50.0
max_frames = 200


mpl.rcParams.update(
    {
        "font.family": "serif",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 1.1,
        "axes.labelsize": 11,
        "axes.titlesize": 13,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.dpi": 120,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.08,
    }
)


def _bracket(ax):
    ax.spines["left"].set_position(("outward", 6))
    ax.spines["bottom"].set_position(("outward", 6))


def fetch_wti_yf(start, end):
    import yfinance as yf

    t = yf.Ticker("CL=F")
    df = t.history(start=start, end=end, auto_adjust=True).reset_index()
    df = df[["Date", "Close"]].rename(columns={"Date": "date", "Close": "price"})
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna().sort_values("date").reset_index(drop=True)
    return df


def load_wti_csv(path):
    df = pd.read_csv(path)
    df.columns = [c.lower() for c in df.columns]
    if "date" not in df.columns:
        raise ValueError("Need Date column")
    pcol = next(
        (c for c in ("price", "wti", "close", "value") if c in df.columns), None
    )
    if pcol is None:
        raise ValueError("Need price column: price|wti|close|value")
    df["date"] = pd.to_datetime(df["date"])
    df[pcol] = pd.to_numeric(df[pcol], errors="coerce")
    df = df.dropna(subset=["date", pcol]).sort_values("date")
    return df.rename(columns={pcol: "price"}).reset_index(drop=True)


df = fetch_wti_yf(start_date, end_date) if use_yfinance else load_wti_csv(csv_path)

if len(df) > 4000:
    df = df.set_index("date").resample("W-FRI").last().dropna().reset_index()

n = len(df)
stride = max(1, int(math.ceil(n / max_frames)))
df_anim = df.iloc[::stride].copy()
dates = pd.to_datetime(df_anim["date"]).to_numpy()
prices = df_anim["price"].to_numpy(float)
n_frames = len(df_anim)

locator = AutoDateLocator()
y_min = math.floor(min(prices.min(), y_global) / 5) * 5
y_max = math.ceil(max(prices.max(), y_shale) / 5) * 5

fig, ax = plt.subplots(figsize=(10, 4.5))


def draw_frame(k):
    ax.clear()
    _bracket(ax)
    ax.set_title(title)
    ax.set_ylabel("USD per barrel")
    ax.set_ylim(y_min, y_max)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(ConciseDateFormatter(locator))

    x = dates[: k + 1]
    y = prices[: k + 1]

    ax.fill_between(
        x, y_shale, y, where=y >= y_shale, interpolate=True, alpha=0.55, color="#2ca02c"
    )
    ax.fill_between(
        x, y, y_shale, where=y < y_shale, interpolate=True, alpha=0.55, color="#d62728"
    )
    ax.axhline(y_shale, lw=1.1, color="0.35", alpha=0.9, zorder=3)
    ax.axhline(y_global, lw=1.0, color="0.55", ls="--", alpha=0.8, zorder=3)

    if len(x):
        ax.text(
            x[0],
            y_shale + 1.2,
            "$70  US shale price",
            va="bottom",
            ha="left",
            fontsize=9,
            color="0.2",
        )
        ax.text(
            x[0],
            y_global + 1.2,
            "$50  Global average",
            va="bottom",
            ha="left",
            fontsize=9,
            color="0.3",
        )

    ax.plot(x, y, lw=1.2, color="black", alpha=0.95)


# Final PNG
draw_frame(n_frames - 1)
plt.savefig(out_png, dpi=dpi)
plt.show()

# GIF with PNG buffer frames
with imageio.get_writer(out_gif, mode="I", fps=fps) as w:
    for k in range(n_frames):
        draw_frame(k)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi)
        buf.seek(0)
        w.append_data(imageio.imread(buf))
        buf.close()

plt.close(fig)
print(f"Wrote {out_png} and {out_gif}")
