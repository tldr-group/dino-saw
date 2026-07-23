from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, TypeAlias, Optional

import numpy as np
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.artist import Artist
from matplotlib.transforms import Bbox
from matplotlib.text import Text

# TODO: Allow scene layout to be relative - that sets relative to thing be plotted
#


Edge = Literal["top", "bottom", "left", "right"]
Connection: TypeAlias = tuple[Edge, float]

Corner = Literal["top_left", "top_right", "bottom_left", "bottom_right"]

RelativeTo: TypeAlias = Edge | Literal["center"] | Corner


def _point_on_bbox(bb: Bbox, relative_to: Optional[RelativeTo]) -> tuple[float, float]:
    if relative_to is None:
        return (bb.x0, bb.y0)

    w, h = bb.width, bb.height

    match relative_to:
        case "center":
            return (bb.x0 + w / 2, bb.y0 + h / 2)
        case "top_left":
            return (bb.x0, bb.y1)
        case "top_right":
            return (bb.x1, bb.y1)
        case "bottom_left":
            return (bb.x0, bb.y0)
        case "bottom_right":
            return (bb.x1, bb.y0)
        case "top":
            return (bb.x0 + w / 2, bb.y1)
        case "bottom":
            return (bb.x0 + w / 2, bb.y0)
        case "left":
            return (bb.x0, bb.y0 + h / 2)
        case "right":
            return (bb.x1, bb.y0 + h / 2)
        case _:
            raise ValueError(f"Unknown RelativeTo value: {relative_to}")


@dataclass(slots=True)
class SceneLayout:
    # x, y, widht, height in relative to a parent Axes
    parent: Axes
    x: float
    y: float
    width: float
    height: float


@dataclass(slots=True)
class NodeLayout:
    """
    Position this patch relative to another patch.
    `anchor_patch` is the reference patch.
    `offset` is (dx, dy) in figure-relative coordinates (0-1).
    """

    anchor_patch: Node
    offset: tuple[float, float]
    width: float
    height: float
    # where on the anchor patch the offset is applied. Defaults to bottom-left
    relative_to: RelativeTo = "bottom_left"


Layout: TypeAlias = SceneLayout | NodeLayout


class Renderable:
    def __init__(self) -> None:
        self._bbox: Bbox = Bbox([[0, 0], [0, 0]])

    def draw(self) -> None:
        raise NotImplementedError()

    def _resolve_bbox(self) -> Bbox:
        raise NotImplementedError()

    @property
    def bbox(self) -> Bbox:
        if self._bbox is None:
            self._bbox = self._resolve_bbox()
        return self._bbox

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


class Node(Renderable):
    def __init__(self, layout: Layout, clip_on: bool = True):
        super().__init__()
        self.layout = layout
        self.parent = self.get_parent()
        self.clip_on = clip_on
        # eagerly resolve bbox (keeps previous behaviour)
        self._bbox = self._resolve_bbox()

    def get_parent(self) -> Axes:
        if isinstance(self.layout, SceneLayout):
            return self.layout.parent
        elif isinstance(self.layout, NodeLayout):
            # recursive parent grabbing
            return self.layout.anchor_patch.get_parent()
        else:
            raise TypeError("Unknown layout type.")

    def get_transform(self):
        assert isinstance(self.parent, Axes)
        return self.parent.transAxes

    def add_to_parent(self, artist: Artist) -> None:
        if isinstance(self.parent, Axes):
            self.parent.add_artist(artist)
        else:
            raise TypeError("Unknown parent type.")

    def _resolve_bbox(self) -> Bbox:
        if isinstance(self.layout, SceneLayout):
            x, y, w, h = self.layout.x, self.layout.y, self.layout.width, self.layout.height
        elif isinstance(self.layout, NodeLayout):
            anchor_bb = self.layout.anchor_patch.bbox
            anchor_x, anchor_y = _point_on_bbox(anchor_bb, self.layout.relative_to)

            if self.layout.relative_to in ("middle", "center"):
                x = anchor_x + self.layout.offset[0] - (self.layout.width / 2)
                y = anchor_y + self.layout.offset[1] - (self.layout.height / 2)
            else:
                x = anchor_x + self.layout.offset[0]
                y = anchor_y + self.layout.offset[1]
            w, h = self.layout.width, self.layout.height
        else:
            raise TypeError("Unknown layout type.")
        return Bbox.from_bounds(x, y, w, h)

    @property
    def bbox(self) -> Bbox:
        return self._bbox


class BoxNode(Node):
    def __init__(self, layout: Layout, *, facecolor="lightgray", edgecolor="black", fancy=False, rounding=0.05):
        super().__init__(layout)
        self.facecolor = facecolor
        self.edgecolor = edgecolor
        self.fancy = fancy
        self.rounding = rounding

    def draw(self) -> None:
        transform = self.get_transform()
        bb = self.bbox
        if self.fancy:
            patch = mpatches.FancyBboxPatch(
                (bb.x0, bb.y0),
                bb.width,
                bb.height,
                boxstyle=f"round,pad=0.02,rounding_size={self.rounding}",
                facecolor=self.facecolor,
                edgecolor=self.edgecolor,
                transform=transform,
            )
        else:
            patch = mpatches.Rectangle(
                (bb.x0, bb.y0),
                bb.width,
                bb.height,
                facecolor=self.facecolor,
                edgecolor=self.edgecolor,
                transform=transform,
            )

        self.add_to_parent(patch)


class ImageNode(Node):
    def __init__(
        self,
        layout: Layout,
        image: np.ndarray,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        cmap: Optional[str] = None,
    ):
        super().__init__(layout)
        self.image = image

        self.vmin = vmin if vmin is not None else np.min(image)
        self.vmax = vmax if vmax is not None else np.max(image)
        self.cmap = cmap

    def draw(self) -> None:
        transform = self.get_transform()
        bb = self.bbox
        self.parent.imshow(
            self.image,
            extent=(bb.x0, bb.x1, bb.y0, bb.y1),
            transform=transform,
            aspect="auto",
            zorder=100,
            vmin=self.vmin,
            vmax=self.vmax,
            cmap=self.cmap,
        )


class TextNode(Node):
    def __init__(
        self,
        layout: Layout,
        text: str,
        fontsize: int = 12,
        color: str = "black",
        ha: str = "center",
        va: str = "center",
        bold: bool = False,
        clip_on: bool = True,
    ):
        super().__init__(layout, clip_on)
        self.text = text
        self.fontsize = fontsize
        self.color = color
        self.ha = ha
        self.va = va
        self.bold = bold

    def draw(self) -> None:
        transform = self.get_transform()
        bb = self.bbox
        x = bb.x0 + bb.width / 2
        y = bb.y0 + bb.height / 2
        weight = 700 if self.bold else 500
        text_artist = Text(
            x,
            y,
            self.text,
            color=self.color,
            ha=self.ha,
            va=self.va,
            weight=weight,
            transform=transform,
            clip_on=self.clip_on,
            fontsize=self.fontsize,
        )
        self.add_to_parent(text_artist)


class ArrowConnector(Node):
    def __init__(
        self,
        start_obj: Renderable | Layout,
        end_obj: Optional[Renderable | Layout],
        start: Optional[Connection] = None,
        end: Optional[Connection] = None,
        dx: Optional[float] = None,
        dy: Optional[float] = None,
        arrowstyle: Literal["->", "-", "<->"] = "->",
        shrinkA: float = 0,
        shrinkB: float = 0,
        linestyle: str = "solid",
        color: str = "black",
        lw: float = 1,
        scale: float = 12.0,
        clip_on_draw: bool = True,
    ):
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
        self.lw = lw
        self.scale = scale
        self.clip_on_draw = clip_on_draw

        self._bbox = self._resolve_bbox()

    def get_parent(self) -> Axes:
        if isinstance(self.start_obj, SceneLayout):
            return self.start_obj.parent
        elif isinstance(self.start_obj, NodeLayout):
            # recursive parent grabbing
            return self.start_obj.anchor_patch.get_parent()
        elif isinstance(self.start_obj, Node):
            return self.start_obj.get_parent()
        elif isinstance(self.end_obj, SceneLayout):
            # recursive parent grabbing
            return self.end_obj.parent
        elif isinstance(self.end_obj, NodeLayout):
            # recursive parent grabbing
            return self.end_obj.anchor_patch.get_parent()
        elif isinstance(self.end_obj, Node):
            return self.end_obj.get_parent()

        else:
            raise TypeError("Unknown layout type.")

    # helpers to compute bboxes for raw layouts
    def _bbox_from_layout(self, layout: Layout) -> Bbox:
        if isinstance(layout, SceneLayout):
            x, y, w, h = layout.x, layout.y, layout.width, layout.height
        elif isinstance(layout, NodeLayout):
            anchor_bb = layout.anchor_patch.bbox
            anchor_x, anchor_y = _point_on_bbox(anchor_bb, layout.relative_to)
            if layout.relative_to in ("middle", "center"):
                x = anchor_x + layout.offset[0] - (layout.width / 2)
                y = anchor_y + layout.offset[1] - (layout.height / 2)
            else:
                x = anchor_x + layout.offset[0]
                y = anchor_y + layout.offset[1]
            w, h = layout.width, layout.height
        else:
            raise TypeError("Unknown layout type for bbox computation")
        return Bbox.from_bounds(x, y, w, h)

    def get_loc(self, obj: Renderable | Layout, conn: Optional[Connection]) -> tuple[float, float]:
        if isinstance(obj, Node):
            assert conn is not None, "Connection must be provided for Node"
            return obj.connection_point(conn)
        else:
            assert isinstance(obj, Layout), "Object must be either Node or Layout"
            # raw layout
            bb = self._bbox_from_layout(obj)
            return _point_on_bbox(bb, conn[0] if conn is not None else None) if conn is not None else (bb.x0, bb.y0)

    def get_extent(self) -> tuple[float, float, float, float]:
        x0, y0 = self.get_loc(self.start_obj, self.start_conn)
        if self.dx is not None and self.dy is not None:
            x1 = x0 + self.dx
            y1 = y0 + self.dy
        else:
            assert self.end_obj is not None
            x1, y1 = self.get_loc(self.end_obj, self.end_conn)
        return (x0, y0, x1, y1)

    def _resolve_bbox(self) -> Bbox:
        x0, y0, x1, y1 = self.get_extent()
        xmin, xmax = sorted([x0, x1])
        ymin, ymax = sorted([y0, y1])
        return Bbox.from_extents(xmin, ymin, xmax, ymax)

    def apply_shrink(self, x0: float, y0: float, x1: float, y1: float) -> tuple[float, float, float, float]:
        direction = "h" if abs(x1 - x0) > abs(y1 - y0) else "v"
        if direction == "h":
            if x1 > x0:
                x0 = x0 + self.shrinkA
                x1 = x1 - self.shrinkB
            else:
                x0 = x0 - self.shrinkA
                x1 = x1 + self.shrinkB
        else:
            if y1 > y0:
                y0 = y0 + self.shrinkA
                y1 = y1 - self.shrinkB
            else:
                y0 = y0 - self.shrinkA
                y1 = y1 + self.shrinkB
        return x0, y0, x1, y1

    def _determine_parent(self) -> Axes:
        # Prefer Node parents if available
        if isinstance(self.start_obj, Node):
            return self.start_obj.get_parent()
        if isinstance(self.end_obj, Node):
            return self.end_obj.get_parent()
        if isinstance(self.start_obj, NodeLayout):
            return self.start_obj.anchor_patch.get_parent()
        if isinstance(self.end_obj, NodeLayout):
            return self.end_obj.anchor_patch.get_parent()
        # otherwise layouts embed parent
        if isinstance(self.start_obj, SceneLayout):
            return self.start_obj.parent
        if isinstance(self.end_obj, SceneLayout):
            return self.end_obj.parent
        # fallback: raise
        raise RuntimeError("Cannot determine parent for ArrowConnector")

    def draw(self) -> None:
        x0, y0, x1, y1 = self.get_extent()
        x0, y0, x1, y1 = self.apply_shrink(x0, y0, x1, y1)

        parent = self._determine_parent()
        transform = parent.transAxes

        arrow = mpatches.FancyArrowPatch(
            (x0, y0),
            (x1, y1),
            arrowstyle=self.arrowstyle,
            mutation_scale=self.scale,
            linewidth=self.lw,
            color=self.color,
            linestyle=self.linestyle,
            transform=transform,
            clip_on=self.clip_on_draw,
        )

        if isinstance(parent, Axes):
            parent.add_patch(arrow)
        else:
            parent.patches.append(arrow)


class CubeNode(Node):
    def __init__(
        self,
        layout: Layout,
        color: str = "steelblue",
        edgecolor: str = "black",
        depth_ratio: float = 0.25,
        lw: float = 0.5,
    ):
        super().__init__(layout)
        if depth_ratio <= 0:
            raise ValueError("depth_ratio must be positive")
        self.color = color
        self.edgecolor = edgecolor
        self.depth_ratio = depth_ratio
        self.lw = lw

    def draw(self) -> None:
        transform = self.get_transform()
        bb = self.bbox

        w = bb.width
        h = bb.height

        dx = w * self.depth_ratio
        dy = h * self.depth_ratio

        x0 = bb.x0
        y0 = bb.y0

        front = [
            (x0, y0),
            (x0 + w, y0),
            (x0 + w, y0 + h),
            (x0, y0 + h),
        ]

        top = [
            (x0, y0 + h),
            (x0 + dx, y0 + h + dy),
            (x0 + w + dx, y0 + h + dy),
            (x0 + w, y0 + h),
        ]

        side = [
            (x0 + w, y0),
            (x0 + w + dx, y0 + dy),
            (x0 + w + dx, y0 + h + dy),
            (x0 + w, y0 + h),
        ]

        def _adjust_color(color: str, factor: float) -> tuple[float, float, float]:
            r, g, b = mcolors.to_rgb(color)
            r = min(max(r * factor, 0), 1)
            g = min(max(g * factor, 0), 1)
            b = min(max(b * factor, 0), 1)
            return (r, g, b)

        front_color = _adjust_color(self.color, 1.0)
        top_color = _adjust_color(self.color, 1.15)
        side_color = _adjust_color(self.color, 0.85)

        for verts, fc in ((front, front_color), (top, top_color), (side, side_color)):
            poly = mpatches.Polygon(
                verts, closed=True, facecolor=fc, edgecolor=self.edgecolor, transform=transform, linewidth=self.lw
            )
            self.add_to_parent(poly)


class PlotNode(Node):
    def __init__(self, layout: Layout, fig: Figure):
        super().__init__(layout)
        self.fig = fig

        x0, y0, w, h = self.bbox.x0, self.bbox.y0, self.bbox.width, self.bbox.height
        self.ax = self.fig.add_axes((x0, y0, w, h))

    def draw(self):
        # self.add_to_parent(self.parent)
        pass


def render_objects(objs: list[Renderable]) -> None:
    """
    Draw all objects. Non-arrow objects (Nodes) are drawn first, then ArrowConnector instances.
    """
    # draw nodes/patches first
    # sorted_objs: list[Renderable] = sorted(objs, key=lambda obj: isinstance(obj, ArrowConnector))
    sorted_objs: list[Renderable] = objs

    for obj in sorted_objs:
        obj.draw()
