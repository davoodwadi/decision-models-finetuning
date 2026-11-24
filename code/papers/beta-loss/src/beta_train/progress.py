"""HTML progress bars for notebooks, redrawn in place through a display handle (no growing output). Used by the trainer
(steps, micro-chunk passes) and the validators (GPU batches)."""

from __future__ import annotations

import html
import time

from IPython.display import HTML, display

# Shortest time between two redraws of a progress bar (each redraw sends one HTML update to the notebook front end)
PROGRESS_INTERVAL_SEC = 0.5


# Progress bar colors: semi-transparent, so the bar reads in light and dark notebook themes
PROGRESS_TRACK_STYLE = "background-color: rgba(128, 128, 128, 0.25)"
PROGRESS_FILL_STYLE = "background-color: rgba(0, 160, 0, 0.65)"


def format_duration(seconds: float) -> str:
    # 75 -> "01:15", 3725 -> "1:02:05"
    whole_seconds = int(round(seconds))
    hours = whole_seconds // 3600
    minutes = (whole_seconds % 3600) // 60
    secs = whole_seconds % 60
    if hours > 0:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


class HtmlProgress:
    """A progress bar drawn as one HTML output and redrawn in place through a display handle, like the validation table
    and the figure (a tqdm widget can stop being drawn in some notebook front ends while those keep updating).
    The part of tqdm's interface the trainer uses: update, reset, set_description, set_postfix, close.
    Redraws at most every `min_interval_sec` (each redraw is one message to the front end); reset, set_postfix, close
    and reaching the total always redraw."""

    def __init__(self, total: int, initial: int, description: str, unit: str, min_interval_sec: float):
        for name, value in (("total", total), ("initial", initial)):
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer, got {value!r}")
        if initial > total:
            raise ValueError(f"initial ({initial}) is above total ({total})")
        if min_interval_sec < 0:
            raise ValueError(f"min_interval_sec must be >= 0, got {min_interval_sec!r}")
        self.total = total
        self.n = initial
        self.description = description
        self.unit = unit
        self.min_interval_sec = min_interval_sec
        self.postfix = ""
        self.closed = False
        self.start_n = initial                 # the rate counts only the updates since the (re)start
        self.start_time = time.perf_counter()
        self.last_draw_time = 0.0
        self.handle = display(HTML(self.render_html()), display_id=True)
        self.last_draw_time = time.perf_counter()

    def render_html(self) -> str:
        # Description, bar, n/total and percent, elapsed < time left, rate, postfix
        elapsed = time.perf_counter() - self.start_time
        done_since_start = self.n - self.start_n
        if self.total > 0:
            fraction = self.n / self.total
        else:
            fraction = 0.0
        fraction = min(max(fraction, 0.0), 1.0)
        if done_since_start > 0 and elapsed > 0:
            rate = done_since_start / elapsed
            remaining = format_duration((self.total - self.n) / rate)
            if rate >= 1:
                rate_text = f"{rate:.2f} {self.unit}/s"
            else:
                rate_text = f"{1 / rate:.2f} s/{self.unit}"
        else:
            remaining = "?"
            rate_text = f"? {self.unit}/s"
        status = f"{self.n}/{self.total} ({100 * fraction:.0f}%) [{format_duration(elapsed)}<{remaining}, {rate_text}]"
        if self.postfix:
            status = f"{status} {self.postfix}"
        return (
            "<div style='font-family: monospace; font-size: 12px; display: flex; align-items: center; gap: 8px; margin: 2px 0'>"
            f"<span>{html.escape(self.description)}</span>"
            f"<div style='flex: 0 0 240px; height: 10px; border-radius: 3px; overflow: hidden; {PROGRESS_TRACK_STYLE}'>"
            f"<div style='width: {100 * fraction:.2f}%; height: 100%; {PROGRESS_FILL_STYLE}'></div></div>"
            f"<span>{html.escape(status)}</span>"
            "</div>"
        )

    def draw(self, force: bool) -> None:
        if self.closed:
            raise RuntimeError(f"progress bar {self.description!r} is closed")
        now = time.perf_counter()
        if not force and now - self.last_draw_time < self.min_interval_sec:
            return
        self.handle.update(HTML(self.render_html()))
        self.last_draw_time = now

    def update(self, n: int = 1) -> None:
        self.n += n
        if self.n > self.total:
            raise ValueError(f"progress bar {self.description!r} went past its total: {self.n} > {self.total}")
        self.draw(force=self.n == self.total)

    def reset(self, total: int) -> None:
        # Back to 0 with a new total (the chunk bar, once per step); the rate restarts
        if not isinstance(total, int) or isinstance(total, bool) or total < 0:
            raise ValueError(f"total must be a non-negative integer, got {total!r}")
        self.total = total
        self.n = 0
        self.start_n = 0
        self.start_time = time.perf_counter()
        self.draw(force=True)

    def set_description(self, description: str) -> None:
        self.description = description
        self.draw(force=False)

    def set_postfix(self, **values: str) -> None:
        parts = []
        for name, value in values.items():
            parts.append(f"{name}={value}")
        self.postfix = ", ".join(parts)
        self.draw(force=True)

    def close(self) -> None:
        # Final redraw (shows the last state); a closed bar cannot be drawn again. Closing twice is a no-op
        if self.closed:
            return
        self.draw(force=True)
        self.closed = True
