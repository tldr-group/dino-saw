from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias, Optional, Iterable

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib.axes import Axes
from matplotlib.transforms import Bbox
import numpy as np


Edge = Literal["top", "bottom", "left", "right"]
Connection: TypeAlias = tuple[Edge, float]  # e.g. ("left", 0.3)


# -------------------------
# Layout specification
# -------------------------


@dataclass(slots=True)
class FigureSpec:
    # coordinates in figure-relative space (0–1)
    x: float
    y: float
    width: float
    height: float


@dataclass(slots=True)
class RelativeSpec:
    """
    Position this patch relative to another patch.
    `anchor_patch` is the reference patch.
    `offset` is (dx, dy) in figure-relative coordinates (0-1).
    """

    anchor_patch: CustomPatch
    offset: tuple[float, float]
    width: float
    height: float


LayoutSpec: TypeAlias = FigureSpec | RelativeSpec


# -------------------------
# Base wrapper
# -------------------------


class CustomPatch:
    def __init__(self, layout: LayoutSpec):
        self.layout = layout
        self._bbox: Optional[Bbox] = None
        self._resolve_bbox()

    def _resolve_bbox(self) -> None:
        """
        Eagerly resolve the bounding box at instantiation, since layout is now static.
        """
        if isinstance(self.layout, FigureSpec):
            x = self.layout.x
            y = self.layout.y
            w = self.layout.width
            h = self.layout.height
        elif isinstance(self.layout, RelativeSpec):
            anchor_bb = self.layout.anchor_patch.bbox
            x = anchor_bb.x0 + self.layout.offset[0]
            y = anchor_bb.y0 + self.layout.offset[1]
            w = self.layout.width
            h = self.layout.height
        else:
            raise TypeError("Unknown layout type.")
        self._bbox = Bbox.from_bounds(x, y, w, h)

    # ---- layout resolution ----

    def resolve_layout(self) -> tuple[float, float, float, float]:
        """
        Deprecated: kept for compatibility, but does nothing now.
        """
        # No-op, as bbox is now resolved at instantiation
        bb = self.bbox
        return (bb.x0, bb.y0, bb.width, bb.height)

    @property
    def bbox(self) -> Bbox:
        if self._bbox is None:
            raise RuntimeError("BBox not resolved.")
        return self._bbox

    @property
    def mid_x(self) -> float:
        """Return the x coordinate of the center of the bbox."""
        bb = self.bbox
        return bb.x0 + bb.width / 2

    @property
    def mid_y(self) -> float:
        """Return the y coordinate of the center of the bbox."""
        bb = self.bbox
        return bb.y0 + bb.height / 2

    @property
    def x0(self) -> float:
        return self.bbox.x0

    @property
    def y0(self) -> float:
        return self.bbox.y0

    @property
    def w(self) -> float:
        return self.bbox.width

    @property
    def h(self) -> float:
        return self.bbox.height

    # ---- geometry helpers ----

    def connection_point(self, conn: Connection) -> tuple[float, float]:
        edge, frac = conn
        bb = self.bbox

        if edge == "left":
            return bb.x0, bb.y0 + frac * bb.height
        if edge == "right":
            return bb.x1, bb.y0 + frac * bb.height
        if edge == "top":
            return bb.x0 + frac * bb.width, bb.y1
        if edge == "bottom":
            return bb.x0 + frac * bb.width, bb.y0

        raise ValueError(f"Unknown edge: {edge}")

    # ---- drawing ----

    def draw(self, ax: Axes) -> None:
        raise NotImplementedError


# -------------------------
# Concrete patches
# -------------------------


class BoxPatch(CustomPatch):
    def __init__(
        self,
        layout: LayoutSpec,
        *,
        facecolor: str = "lightgray",
        edgecolor: str = "black",
        fancy: bool = False,
        rounding: float = 0.05,
    ):
        super().__init__(layout)
        self.facecolor = facecolor
        self.edgecolor = edgecolor
        self.fancy = fancy
        self.rounding = rounding

    def draw(self, ax: Axes) -> None:
        bb = self.bbox

        patch: mpatches.FancyBboxPatch | mpatches.Rectangle
        if self.fancy:
            patch = mpatches.FancyBboxPatch(
                (bb.x0, bb.y0),
                bb.width,
                bb.height,
                boxstyle=f"round,pad=0.02,rounding_size={self.rounding}",
                facecolor=self.facecolor,
                edgecolor=self.edgecolor,
                transform=ax.transAxes,
            )
        else:
            patch = mpatches.Rectangle(
                (bb.x0, bb.y0),
                bb.width,
                bb.height,
                facecolor=self.facecolor,
                edgecolor=self.edgecolor,
                transform=ax.transAxes,
            )

        ax.add_patch(patch)


class ImagePatch(CustomPatch):
    def __init__(self, layout: LayoutSpec, image: np.ndarray):
        super().__init__(layout)
        self.image = image

    def draw(self, ax: Axes) -> None:
        bb = self.bbox
        ax.imshow(
            self.image,
            extent=(bb.x0, bb.x1, bb.y0, bb.y1),
            transform=ax.transAxes,
            aspect="auto",
            zorder=0,
        )


class TextPatch(CustomPatch):
    def __init__(
        self,
        layout: LayoutSpec,
        text: str,
        fontsize: int = 12,
        color: str = "black",
        ha: str = "center",
        va: str = "center",
        bold: bool = False,
    ):
        super().__init__(layout)
        self.text = text
        self.fontsize = fontsize
        self.color = color
        self.ha = ha  # horizontal alignment
        self.va = va  # vertical alignment
        self.bold = bold

    def draw(self, ax: Axes) -> None:
        bb = self.bbox
        x = bb.x0 + bb.width / 2
        y = bb.y0 + bb.height / 2
        weight = 700 if self.bold else 500
        ax.text(
            x,
            y,
            self.text,
            fontsize=self.fontsize,
            color=self.color,
            ha=self.ha,
            va=self.va,
            transform=ax.transAxes,
            weight=weight,
        )


# -------------------------
# Arrow connector
# -------------------------


class ArrowConnector(CustomPatch):
    def __init__(
        self,
        start_obj: CustomPatch | LayoutSpec,
        end_obj: CustomPatch | LayoutSpec | None,
        start: Connection | None = None,
        end: Connection | None = None,
        dx: float | None = None,
        dy: float | None = None,
        arrowstyle: Literal["->", "-", "<->"] = "->",
        shrinkA: float = 0,
        shrinkB: float = 0,
        linestyle: str = "solid",
        color: str = "black",
    ):
        # arrows do not own layout; they derive it dynamically
        super().__init__(layout=FigureSpec(0, 0, 0, 0))
        self.start_obj = start_obj
        self.end_obj = end_obj
        self.start_conn = start
        self.end_conn = end
        self.dx = dx
        self.dy = dy

        self.shrinkA = shrinkA
        self.shrinkB = shrinkB
        self.arrowstyle = arrowstyle
        self.linestyle = linestyle
        self.color = color

    @property
    def bbox(self) -> Bbox:
        # dynamic bounding box covering arrow endpoints
        x0, y0, x1, y1 = self.get_extent()
        xmin, xmax = sorted([x0, x1])
        ymin, ymax = sorted([y0, y1])
        return Bbox.from_extents(xmin, ymin, xmax, ymax)

    @property
    def mid_x(self) -> float:
        x0, y0, x1, y1 = self.get_extent()
        return (x0 + x1) / 2

    @property
    def mid_y(self) -> float:
        x0, y0, x1, y1 = self.get_extent()
        return (y0 + y1) / 2

    def get_loc(self, obj: CustomPatch | LayoutSpec, conn: Connection | None) -> tuple[float, float]:
        if isinstance(obj, CustomPatch):
            assert conn is not None, "Connection must be provided for CustomPatch"
            return obj.connection_point(conn)
        elif isinstance(obj, LayoutSpec):
            if isinstance(obj, FigureSpec):
                x = obj.x
                y = obj.y
            elif isinstance(obj, RelativeSpec):
                anchor_bb = obj.anchor_patch.bbox
                x = anchor_bb.x0 + obj.offset[0]
                y = anchor_bb.y0 + obj.offset[1]
            else:
                raise TypeError("Unknown layout type.")
            return (x, y)

    def get_extent(self) -> tuple[float, float, float, float]:
        x0, y0 = self.get_loc(self.start_obj, self.start_conn)
        if self.dx is not None and self.dy is not None:
            x1 = x0 + self.dx
            y1 = y0 + self.dy
        else:
            assert self.end_obj is not None
            x1, y1 = self.get_loc(self.end_obj, self.end_conn)

        return (x0, y0, x1, y1)

    def resolve_layout(self) -> tuple[float, float, float, float]:
        x0, y0, x1, y1 = self.get_extent()
        w, h = abs(x1 - x0), abs(y1 - y0)
        self._bbox = Bbox.from_bounds(min(x0, x1), min(y0, y1), w, h)
        return (x0, y0, w, h)

    def apply_shrink(self, x0: float, y0: float, x1: float, y1: float) -> tuple[float, float, float, float]:
        direction = "h" if abs(x1 - x0) > abs(y1 - y0) else "v"
        if direction == "h":
            x0 = x0 + self.shrinkA if x1 > x0 else x0 - self.shrinkA
            x1 = x1 - self.shrinkB if x1 > x0 else x1 + self.shrinkB
        elif direction == "v":
            y0 = y0 + self.shrinkA if y1 > y0 else y0 - self.shrinkA
            y1 = y1 - self.shrinkB if y1 > y0 else y1 + self.shrinkB
        return x0, y0, x1, y1

    def draw(self, ax: Axes) -> None:
        x0, y0, x1, y1 = self.get_extent()

        x0, y0, x1, y1 = self.apply_shrink(x0, y0, x1, y1)

        arrow = mpatches.FancyArrowPatch(
            (x0, y0),
            (x1, y1),
            arrowstyle=self.arrowstyle,
            mutation_scale=12,
            linewidth=1.5,
            color=self.color,
            linestyle=self.linestyle,
            transform=ax.transAxes,
        )
        ax.add_patch(arrow)


def _adjust_color(color: str, factor: float) -> tuple[float, float, float]:
    """
    Darken/lighten an RGB color by multiplying channels by factor.
    factor < 1 -> darker
    factor > 1 -> lighter
    """
    r, g, b = mcolors.to_rgb(color)
    r = min(max(r * factor, 0), 1)
    g = min(max(g * factor, 0), 1)
    b = min(max(b * factor, 0), 1)
    return (r, g, b)


class CubePatch(CustomPatch):
    """
    Draws a 2D projected cuboid.

    - Front-bottom-left is anchored at (bbox.x0, bbox.y0)
    - bbox.width  -> front face width
    - bbox.height -> front face height
    - depth_ratio controls projection depth independently
    """

    def __init__(
        self,
        layout: LayoutSpec,
        color: str = "steelblue",
        edgecolor: str = "black",
        depth_ratio: float = 0.25,
    ):
        super().__init__(layout)
        if depth_ratio <= 0:
            raise ValueError("depth_ratio must be positive")
        self.color = color
        self.edgecolor = edgecolor
        self.depth_ratio = depth_ratio

    def draw(self, ax: Axes) -> None:
        bb = self.bbox

        w = bb.width
        h = bb.height

        # Depth projected proportionally to width/height
        dx = w * self.depth_ratio
        dy = h * self.depth_ratio

        x0 = bb.x0
        y0 = bb.y0

        # Front face
        front = [
            (x0, y0),
            (x0 + w, y0),
            (x0 + w, y0 + h),
            (x0, y0 + h),
        ]

        # Top face
        top = [
            (x0, y0 + h),
            (x0 + dx, y0 + h + dy),
            (x0 + w + dx, y0 + h + dy),
            (x0 + w, y0 + h),
        ]

        # Right face
        side = [
            (x0 + w, y0),
            (x0 + w + dx, y0 + dy),
            (x0 + w + dx, y0 + h + dy),
            (x0 + w, y0 + h),
        ]

        front_color = _adjust_color(self.color, 1.0)
        top_color = _adjust_color(self.color, 1.15)
        side_color = _adjust_color(self.color, 0.85)

        for verts, fc in (
            (front, front_color),
            (top, top_color),
            (side, side_color),
        ):
            poly = mpatches.Polygon(
                verts,
                closed=True,
                facecolor=fc,
                edgecolor=self.edgecolor,
                transform=ax.transAxes,
            )
            ax.add_patch(poly)


# -------------------------
# Rendering orchestration
# -------------------------


def render_objects(
    ax: Axes,
    objs: list[CustomPatch],
    *,
    grid_shape: tuple[int, int] | None = None,
) -> None:
    """
    Resolves layout for all non-arrow objects, then draws everything.
    Arrows may connect to arrows because connection_point is unified.
    """

    # first pass: (no longer needed, bbox is resolved at instantiation)

    # second pass: draw all
    for obj in objs:
        obj.draw(ax)

    ax.set_axis_off()
