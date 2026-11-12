"""A small QPainter chart: line, bar, hist, scatter and heatmap.
"""
import math

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QFontMetricsF, QLinearGradient, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QWidget

from .. import theme


def nice_ticks(lo, hi, count):
    if hi - lo < 1e-12:
        hi = lo + (abs(lo) * 0.1 or 1.0)
    raw = (hi - lo) / max(count - 1, 1)
    mag = 10 ** math.floor(math.log10(raw))
    step = next(m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw * (1 - 1e-9))
    first = math.floor(lo / step + 1e-9)
    last = math.ceil(hi / step - 1e-9)
    ticks = [round(i * step, 12) for i in range(first, last + 1)]
    return ticks if len(ticks) > 1 else [ticks[0], ticks[0] + step]


def fmt_tick(v, fmt):
    if fmt == "pct":
        return f"{round(v * 100, 4):g}%"
    if fmt == "sci":
        if v == 0:
            return "0"
        mant, exp = f"{v:.1e}".split("e")
        return f"{float(mant):g}e{int(exp)}"
    if abs(v) >= 1000:
        return f"{v / 1000:g}k"
    return f"{round(v, 6):g}"


def fmt_value(v, fmt):
    if fmt == "pct":
        return f"{v * 100:.1f}%"
    if fmt == "int":
        return f"{v:,.0f}"
    if fmt == "sci":
        return f"{v:.2e}"
    return f"{v:,.0f}" if abs(v) >= 1000 else f"{v:.4g}"


def series_color(series, index=0):
    return QColor(theme.SERIES[series.get("slot", index) % len(theme.SERIES)])


def _mix(a, b, t):
    a, b = QColor(a), QColor(b)
    return QColor(*(round(x + (y - x) * t) for x, y in
                    ((a.red(), b.red()), (a.green(), b.green()), (a.blue(), b.blue()))))


class ChartWidget(QWidget):
    def __init__(self, spec=None, compact=True, parent=None):
        super().__init__(parent)
        self.spec = spec
        self.compact = compact
        self._mouse = None
        self.setMinimumHeight(80 if compact else 220)
        if compact:
            self.setAttribute(Qt.WA_TransparentForMouseEvents)
        else:
            self.setMouseTracking(True)
        self.tick_font = QFont()
        self.tick_font.setPixelSize(10 if compact else 11)
        self.fm = QFontMetricsF(self.tick_font)

    def set_spec(self, spec):
        self.spec = spec
        self.update()

    def mouseMoveEvent(self, e):
        self._mouse = e.position()
        self.update()

    def leaveEvent(self, e):
        self._mouse = None
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setFont(self.tick_font)
        spec = self.spec
        if spec and self._has_data(spec):
            getattr(self, "_paint_" + spec["kind"])(p, spec)
        elif spec:
            p.setPen(QColor(theme.MUTED))
            p.drawText(self.rect(), Qt.AlignCenter, spec.get("empty", "No data"))
        p.end()

    @staticmethod
    def _has_data(spec):
        kind = spec["kind"]
        if kind == "line":
            return any(s["values"] for s in spec["series"])
        if kind == "scatter":
            return any(s["points"] for s in spec["series"])
        if kind == "heatmap":
            return bool(spec["matrix"])
        if kind == "hist":
            return bool(spec["values"])
        return bool(spec["categories"])

    # ---- shared pieces -------------------------------------------------

    def _plot_rect(self, y_labels, legend=False):
        fm = self.fm
        titles = not self.compact
        left = max((fm.horizontalAdvance(s) for s in y_labels), default=0) + 8 + (20 if titles else 0)
        top = fm.height() / 2 + 2 + (fm.height() + 8 if legend else 0)
        bottom = fm.height() + 6 + (20 if titles else 0)
        right = 14 if self.compact else 18
        return QRectF(left, top, max(1.0, self.width() - left - right), max(1.0, self.height() - top - bottom))

    def _y_axis(self, p, rect, ticks, ymap, fmt):
        for t in ticks:
            y = round(ymap(t)) + 0.5
            p.setPen(QPen(QColor(theme.BORDER), 1))
            p.drawLine(QPointF(rect.left(), y), QPointF(rect.right(), y))
            p.setPen(QColor(theme.MUTED))
            p.drawText(QRectF(0, y - 8, rect.left() - 8, 16), Qt.AlignRight | Qt.AlignVCenter, fmt_tick(t, fmt))

    def _x_label(self, p, rect, x, text):
        w = self.fm.horizontalAdvance(text) + 4
        p.setPen(QColor(theme.MUTED))
        p.drawText(QRectF(x - w / 2, rect.bottom() + 5, w, self.fm.height()), Qt.AlignHCenter | Qt.AlignTop, text)

    def _axis_titles(self, p, rect, spec, y_title_x=9):
        if self.compact:
            return
        p.setPen(QColor(theme.TEXT_2))
        if spec.get("x_label"):
            p.drawText(QRectF(rect.left(), self.height() - 18, rect.width(), 16), Qt.AlignCenter, spec["x_label"])
        if spec.get("y_label"):
            p.save()
            p.translate(y_title_x, rect.center().y())
            p.rotate(-90)
            p.drawText(QRectF(-rect.height() / 2, -8, rect.height(), 16), Qt.AlignCenter, spec["y_label"])
            p.restore()

    def _legend(self, p, rect, series, dots=False):
        fm = self.fm
        x, mid = rect.left(), 4 + fm.height() / 2
        needed = sum(30 + fm.horizontalAdvance(s["name"]) for s in series)
        if needed > self.width() - x:
            x = 4.0
        room = (self.width() - x) / len(series) - 30
        for i, s in enumerate(series):
            name = fm.elidedText(s["name"], Qt.ElideRight, room)
            p.setPen(Qt.NoPen)
            p.setBrush(series_color(s, i))
            if dots:
                p.drawEllipse(QPointF(x + 4, mid), 3.5, 3.5)
            else:
                p.drawRoundedRect(QRectF(x, mid - 1.5, 12, 3), 1.5, 1.5)
            x += 16
            p.setPen(QColor(theme.TEXT_2))
            p.drawText(QRectF(x, 4, 400, fm.height()), Qt.AlignLeft | Qt.AlignVCenter, name)
            x += fm.horizontalAdvance(name) + 14

    def _dot(self, p, pt, color, r, ring=2.0):
        p.setPen(QPen(QColor(theme.SURFACE), ring))
        p.setBrush(color)
        p.drawEllipse(pt, r, r)

    def _bar(self, p, rect, radius, color):
        """Bar with rounded top corners, square at the baseline."""
        if rect.height() <= radius:
            p.fillRect(rect, color)
            return
        path = QPainterPath()
        path.addRoundedRect(rect, radius, radius)
        base = QPainterPath()
        base.addRect(QRectF(rect.left(), rect.bottom() - radius, rect.width(), radius))
        p.fillPath(path.united(base), color)

    def _tooltip(self, p, anchor, title, rows):
        font = QFont()
        font.setPixelSize(12)
        bold = QFont(font)
        bold.setWeight(QFont.DemiBold)
        fm, bfm = QFontMetricsF(font), QFontMetricsF(bold)
        pad, lh, swatch = 9, fm.height() + 4, 14
        w = max([fm.horizontalAdvance(title)] +
                [swatch + fm.horizontalAdvance(lbl) + 20 + bfm.horizontalAdvance(val) for _, lbl, val in rows]) + pad * 2
        h = pad * 2 + lh * (1 + len(rows))
        x = anchor.x() + 14
        if x + w > self.width() - 2:
            x = anchor.x() - 14 - w
        y = max(2.0, min(anchor.y() - h / 2, self.height() - h - 2))

        p.setPen(QPen(QColor(theme.BORDER_STRONG), 1))
        p.setBrush(QColor(theme.RAISED))
        p.drawRoundedRect(QRectF(x, y, w, h), 6, 6)
        p.setFont(font)
        p.setPen(QColor(theme.TEXT_2))
        ty = y + pad
        p.drawText(QRectF(x + pad, ty, w, lh), Qt.AlignLeft | Qt.AlignVCenter, title)
        for color, label, value in rows:
            ty += lh
            if color is not None:
                p.setPen(Qt.NoPen)
                p.setBrush(color)
                p.drawRoundedRect(QRectF(x + pad, ty + lh / 2 - 4, 8, 8), 2, 2)
            p.setFont(font)
            p.setPen(QColor(theme.TEXT_2))
            p.drawText(QRectF(x + pad + swatch, ty, w, lh), Qt.AlignLeft | Qt.AlignVCenter, label)
            p.setFont(bold)
            p.setPen(QColor(theme.TEXT))
            p.drawText(QRectF(x + pad, ty, w - 2 * pad, lh), Qt.AlignRight | Qt.AlignVCenter, value)
        p.setFont(self.tick_font)

    # ---- chart kinds -----------------------------------------------------

    def _paint_line(self, p, spec):
        series, xs = spec["series"], spec["x"]
        fmt = spec.get("y_format", "float")
        values = [v for s in series for v in s["values"]]
        lo, hi = min(values), max(values)
        if spec.get("y_zero"):
            lo = min(lo, 0.0)
        ticks = nice_ticks(lo, hi, 4 if self.compact else 6)
        y0, y1 = ticks[0], ticks[-1]
        multi = len(series) > 1
        rect = self._plot_rect([fmt_tick(t, fmt) for t in ticks], legend=multi)

        x0, x1 = xs[0], spec.get("x_max", xs[-1])
        if x1 == x0:
            x1 = x0 + 1

        def xmap(x):
            return rect.left() + (x - x0) / (x1 - x0) * rect.width()

        def ymap(y):
            return rect.bottom() - (y - y0) / (y1 - y0) * rect.height()

        self._y_axis(p, rect, ticks, ymap, fmt)
        for t in nice_ticks(x0, x1, 4 if self.compact else 8):
            if x0 <= t <= x1 and float(t).is_integer():
                self._x_label(p, rect, xmap(t), f"{t:g}")
        if multi:
            self._legend(p, rect, series)
        self._axis_titles(p, rect, spec)

        for i, s in enumerate(series):
            color = series_color(s, i)
            pts = [QPointF(xmap(x), ymap(v)) for x, v in zip(xs, s["values"])]
            if len(pts) == 1:
                self._dot(p, pts[0], color, 3)
                continue
            path = QPainterPath(pts[0])
            for pt in pts[1:]:
                path.lineTo(pt)
            if not multi:
                area = QPainterPath(path)
                area.lineTo(pts[-1].x(), rect.bottom())
                area.lineTo(pts[0].x(), rect.bottom())
                area.closeSubpath()
                fade = QLinearGradient(0, rect.top(), 0, rect.bottom())
                top_c, bottom_c = QColor(color), QColor(color)
                top_c.setAlpha(46)
                bottom_c.setAlpha(0)
                fade.setColorAt(0, top_c)
                fade.setColorAt(1, bottom_c)
                p.fillPath(area, fade)
            p.setPen(QPen(color, 1.6 if self.compact else 2, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
            p.setBrush(Qt.NoBrush)
            p.drawPath(path)

        m = self._mouse
        if m is None or not rect.adjusted(-8, 0, 8, 0).contains(m):
            return
        n = max(len(s["values"]) for s in series)
        i = min(range(n), key=lambda k: abs(xmap(xs[k]) - m.x()))
        x = xmap(xs[i])
        p.setPen(QPen(QColor(theme.BORDER_STRONG), 1))
        p.drawLine(QPointF(x, rect.top()), QPointF(x, rect.bottom()))
        rows = []
        for k, s in enumerate(series):
            if i < len(s["values"]):
                color = series_color(s, k)
                self._dot(p, QPointF(x, ymap(s["values"][i])), color, 4.5)
                rows.append((color, s["name"], fmt_value(s["values"][i], fmt)))
        self._tooltip(p, QPointF(x, m.y()), f"{spec.get('x_label', 'x')} {xs[i]:g}", rows)

    def _paint_bar(self, p, spec):
        cats = spec["categories"]
        s = spec["series"][0]
        vals, fmt, color = s["values"], spec.get("y_format", "float"), series_color(s)
        ticks = nice_ticks(min(0.0, min(vals)), max(0.0, max(vals)), 4 if self.compact else 6)
        y0, y1 = ticks[0], ticks[-1]
        rect = self._plot_rect([fmt_tick(t, fmt) for t in ticks])

        def ymap(y):
            return rect.bottom() - (y - y0) / (y1 - y0) * rect.height()

        self._y_axis(p, rect, ticks, ymap, fmt)
        self._axis_titles(p, rect, spec)

        band = rect.width() / len(cats)
        bw = min(band * 0.64, 56.0) if band > 6 else band - 1
        radius = min(3 if self.compact else 4, bw / 2)
        m = self._mouse
        hover = None
        if m is not None and rect.contains(m):
            hover = min(len(cats) - 1, int((m.x() - rect.left()) / band))

        base = ymap(0)
        for i, v in enumerate(vals):
            x = rect.left() + band * i + (band - bw) / 2
            top = ymap(v)
            self._bar(p, QRectF(x, top, bw, base - top), radius, color.lighter(125) if i == hover else color)

        widest = max(self.fm.horizontalAdvance(c) for c in cats) + 6
        step = max(1, math.ceil(widest / band))
        for i in range(0, len(cats), step):
            self._x_label(p, rect, rect.left() + band * (i + 0.5), cats[i])

        if hover is not None:
            title = f"{spec.get('x_label', '')} {cats[hover]}".strip()
            self._tooltip(p, m, title, [(color, s["name"], fmt_value(vals[hover], fmt))])

    def _paint_hist(self, p, spec):
        edges, vals = spec["edges"], spec["values"]
        fmt, xfmt = spec.get("y_format", "int"), spec.get("x_format", "float")
        color = QColor(theme.SERIES[spec.get("slot", 0)])
        ticks = nice_ticks(0, max(vals), 4 if self.compact else 6)
        y1 = ticks[-1]
        rect = self._plot_rect([fmt_tick(t, fmt) for t in ticks])
        x0, x1 = edges[0], edges[-1]

        def xmap(x):
            return rect.left() + (x - x0) / (x1 - x0) * rect.width()

        def ymap(y):
            return rect.bottom() - y / y1 * rect.height()

        self._y_axis(p, rect, ticks, ymap, fmt)
        self._axis_titles(p, rect, spec)
        for t in nice_ticks(x0, x1, 4 if self.compact else 7):
            if x0 <= t <= x1:
                self._x_label(p, rect, xmap(t), fmt_tick(t, xfmt))

        m = self._mouse
        hover = None
        if m is not None and rect.contains(m):
            xv = x0 + (m.x() - rect.left()) / rect.width() * (x1 - x0)
            hover = next((i for i in range(len(vals)) if edges[i] <= xv < edges[i + 1]), None)

        for i, v in enumerate(vals):
            left, right = xmap(edges[i]), xmap(edges[i + 1])
            gap = 2 if right - left > 6 else 1
            top = ymap(v)
            self._bar(p, QRectF(left + gap / 2, top, right - left - gap, rect.bottom() - top),
                      min(3, (right - left) / 3), color.lighter(125) if i == hover else color)

        if hover is not None:
            title = f"{fmt_value(edges[hover], xfmt)} to {fmt_value(edges[hover + 1], xfmt)}"
            self._tooltip(p, m, title, [(color, spec.get("y_label", "Count"), fmt_value(vals[hover], fmt))])

    def _paint_scatter(self, p, spec):
        series = spec["series"]
        xfmt, yfmt = spec.get("x_format", "float"), spec.get("y_format", "float")
        xs = [pt[0] for s in series for pt in s["points"]]
        ys = [pt[1] for s in series for pt in s["points"]]
        n = 4 if self.compact else 6
        if spec.get("diagonal"):
            lo, hi = min(xs + ys), max(xs + ys)
            xt = yt = nice_ticks(lo, hi, n)
        else:
            xt, yt = nice_ticks(min(xs), max(xs), n), nice_ticks(min(ys), max(ys), n)
        multi = len(series) > 1
        rect = self._plot_rect([fmt_tick(t, yfmt) for t in yt], legend=multi)

        def xmap(x):
            return rect.left() + (x - xt[0]) / (xt[-1] - xt[0]) * rect.width()

        def ymap(y):
            return rect.bottom() - (y - yt[0]) / (yt[-1] - yt[0]) * rect.height()

        self._y_axis(p, rect, yt, ymap, yfmt)
        for t in xt:
            self._x_label(p, rect, xmap(t), fmt_tick(t, xfmt))
        if multi:
            self._legend(p, rect, series, dots=True)
        self._axis_titles(p, rect, spec)

        if spec.get("diagonal"):
            d0, d1 = max(xt[0], yt[0]), min(xt[-1], yt[-1])
            p.setPen(QPen(QColor(theme.MUTED), 1, Qt.DashLine))
            p.drawLine(QPointF(xmap(d0), ymap(d0)), QPointF(xmap(d1), ymap(d1)))

        r = 2.6 if self.compact else 4.0
        ring = 1.0 if self.compact else 1.5
        nearest, best = None, 14.0 ** 2
        m = self._mouse
        for i, s in enumerate(series):
            color = series_color(s, i)
            for x, y in s["points"]:
                pt = QPointF(xmap(x), ymap(y))
                self._dot(p, pt, color, r, ring)
                if m is not None:
                    d = (pt.x() - m.x()) ** 2 + (pt.y() - m.y()) ** 2
                    if d < best:
                        nearest, best = (pt, color, s["name"], x, y), d

        if nearest:
            pt, color, name, x, y = nearest
            self._dot(p, pt, color, r + 2, 2)
            self._tooltip(p, pt, name, [(None, spec.get("x_label", "x"), fmt_value(x, xfmt)),
                                        (None, spec.get("y_label", "y"), fmt_value(y, yfmt))])

    def _paint_heatmap(self, p, spec):
        labels, matrix = spec["labels"], spec["matrix"]
        n = len(labels)
        top_val = max(max(row) for row in matrix) or 1
        fm = self.fm
        titles = not self.compact
        left = max(fm.horizontalAdvance(lbl) for lbl in labels) + 8 + (20 if titles else 0)
        bottom = fm.height() + 6 + (20 if titles else 0)
        avail_w, avail_h = self.width() - left - 8, self.height() - 4 - bottom
        cell = max(4.0, min(avail_w / n, avail_h / n))
        gx = left + max(0.0, (avail_w - cell * n) / 2)
        grid = QRectF(gx, 4, cell * n, cell * n)

        m = self._mouse
        hover = None
        if m is not None and grid.contains(m):
            hover = (min(n - 1, int((m.y() - grid.top()) / cell)), min(n - 1, int((m.x() - gx) / cell)))

        value_font = QFont(self.tick_font)
        show_values = cell >= (26 if self.compact else 22)
        for r, row in enumerate(matrix):
            for c, v in enumerate(row):
                t = (v / top_val) ** 0.5
                box = QRectF(gx + c * cell + 1, grid.top() + r * cell + 1, cell - 2, cell - 2)
                p.setPen(Qt.NoPen)
                p.setBrush(_mix(theme.SEQ_LOW, theme.SEQ_HIGH, t))
                p.drawRoundedRect(box, 2, 2)
                if show_values:
                    p.setFont(value_font)
                    p.setPen(QColor("#0e141d" if t > 0.6 else theme.TEXT_2))
                    p.drawText(box, Qt.AlignCenter, f"{v}")

        p.setPen(QColor(theme.MUTED))
        for i, lbl in enumerate(labels):
            p.drawText(QRectF(0, grid.top() + i * cell, gx - 8, cell), Qt.AlignRight | Qt.AlignVCenter, lbl)
        step = max(1, math.ceil((max(fm.horizontalAdvance(lbl) for lbl in labels) + 4) / cell))
        for i in range(0, n, step):
            self._x_label(p, grid, gx + cell * (i + 0.5), labels[i])
        label_w = max(fm.horizontalAdvance(lbl) for lbl in labels)
        self._axis_titles(p, grid, spec, y_title_x=gx - label_w - 20)

        if hover:
            r, c = hover
            p.setPen(QPen(QColor(theme.TEXT), 1.5))
            p.setBrush(Qt.NoBrush)
            p.drawRoundedRect(QRectF(gx + c * cell + 1, grid.top() + r * cell + 1, cell - 2, cell - 2), 2, 2)
            self._tooltip(p, m, f"True {labels[r]} → predicted {labels[c]}",
                          [(None, "Samples", f"{matrix[r][c]:,}")])
