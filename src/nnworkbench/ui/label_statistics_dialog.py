from __future__ import annotations

from collections.abc import Sequence
from math import ceil, floor, prod

from qtpy.QtCore import QAbstractTableModel, QEvent, QModelIndex, QPointF, QRect, QSize, Qt
from qtpy.QtGui import QColor, QFontMetrics, QPainter, QPen, QPolygonF
from qtpy.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QScrollArea,
    QTableView,
    QTabWidget,
    QToolTip,
    QVBoxLayout,
    QWidget,
)

from . import style
from .widgets import badge

UNDEFINED = "undefined · not in dataset.json"


def undefined_ids(analysis: dict) -> set[int]:
    """Label IDs found in the label files that dataset.json does not define (shown apart from its classes)."""
    found = analysis.get("undefined_label_ids", {})
    return {int(value) for values in found.values() for value in values} if isinstance(found, dict) else set()


def _class_name(item: dict, undefined: set[int]) -> str:
    return UNDEFINED if int(item.get("id", -1)) in undefined else str(item.get("name", ""))


class _ClassChart(QWidget):
    """Horizontal bars, one row per class (ID, name, bar, value), so hundreds of classes stay readable; the widget
    is as tall as its rows and sits in a scroll area. Only the rows in view are painted; hovering a row shows it in
    full."""

    ROW = 24
    TOP = 30

    def __init__(self, classes: Sequence[dict], undefined: set[int] | None = None, images: int = 0,
                 parent=None) -> None:
        super().__init__(parent)
        self._all_classes = classes
        self._undefined = undefined or set()
        self._images = images
        self._metric = "voxels"
        self._ignore_background = False
        self._classes: list[dict] = []
        self._refresh()

    def set_metric(self, metric: str) -> None:
        self._metric = metric
        self._refresh()

    def set_ignore_background(self, ignore: bool) -> None:
        self._ignore_background = ignore
        self._refresh()

    def _refresh(self) -> None:
        self._classes = [item for item in self._all_classes
                         if not self._ignore_background or int(item.get("id", -1)) != 0]
        self.setMinimumHeight(self.TOP + max(1, len(self._classes)) * self.ROW + 10)
        self.update()

    def sizeHint(self) -> QSize:
        return QSize(480, self.minimumHeight())

    def _row_at(self, y: int) -> int:
        row = (y - self.TOP) // self.ROW if y >= self.TOP else -1
        return row if 0 <= row < len(self._classes) else -1

    def event(self, event) -> bool:
        if event.type() == QEvent.Type.ToolTip:
            row = self._row_at(event.pos().y())
            if row < 0:
                QToolTip.hideText()
                event.ignore()
                return True
            item = self._classes[row]
            class_id = int(item.get("id", -1))
            name = str(item.get("name", ""))
            note = "\nThis ID is not defined in dataset.json." if class_id in self._undefined else ""
            QToolTip.showText(event.globalPos(), f"{class_id}: {name}\n{int(item.get('voxels', 0)):,} voxels · in "
                              f"{int(item.get('images', 0)):,} of {self._images:,} images{note}", self)
            return True
        return super().event(event)

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.fillRect(event.rect(), QColor("#f8fafc"))
        metrics = QFontMetrics(painter.font())
        if not self._classes:
            painter.setPen(QColor(style.MUTED))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "No class data")
            return
        maximum = max(int(item.get(self._metric, 0)) for item in self._classes) or 1
        width = self.width()
        id_width = metrics.horizontalAdvance(str(max(int(item.get("id", 0)) for item in self._classes))) + 16
        name_width = max(90, min(280, int(width * 0.32)))
        value_width = metrics.horizontalAdvance(f"{maximum:,}") + 18
        left = 12 + id_width + name_width
        plot = max(40, width - left - value_width - 12)
        painter.setPen(QColor(style.MUTED))
        caption = "Voxels per class" if self._metric == "voxels" else "Images containing the class"
        painter.drawText(QRect(12, 6, width - 24, 18), Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
                         metrics.elidedText(f"{caption} · {len(self._classes):,} classes · hover a row for details",
                                            Qt.TextElideMode.ElideRight, width - 24))
        first = max(0, (event.rect().top() - self.TOP) // self.ROW)
        last = min(len(self._classes), (event.rect().bottom() - self.TOP) // self.ROW + 1)
        for row in range(first, last):
            item = self._classes[row]
            class_id = int(item.get("id", -1))
            value = int(item.get(self._metric, 0))
            y = self.TOP + row * self.ROW
            undefined = class_id in self._undefined
            if row % 2:
                painter.fillRect(0, y, width, self.ROW, QColor("#f1f4f8"))
            text_rect = QRect(12, y, id_width - 8, self.ROW)
            painter.setPen(QColor(style.MUTED))
            painter.drawText(text_rect, Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight, str(class_id))
            painter.setPen(QColor(style.ORANGE if undefined else "#334155"))
            name = _class_name(item, self._undefined)
            painter.drawText(QRect(12 + id_width, y, name_width - 8, self.ROW),
                             Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
                             metrics.elidedText(name, Qt.TextElideMode.ElideRight, name_width - 8))
            length = int(plot * value / maximum)
            if value > 0:
                painter.fillRect(left, y + 6, max(2, length), self.ROW - 12,
                                 QColor(style.ORANGE if undefined else "#2f6fca"))
            painter.setPen(QColor(style.MUTED))
            painter.drawText(QRect(left + plot + 6, y, value_width, self.ROW),
                             Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, f"{value:,}")


def _chart_area(chart: QWidget) -> QScrollArea:
    area = QScrollArea()
    area.setWidgetResizable(True)
    area.setFrameShape(QFrame.Shape.NoFrame)
    area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    area.setWidget(chart)
    return area


class _AxisScatterChart(QWidget):
    """Show every label image's spatial shape or spacing along X, Y, and Z."""

    _AXIS_NAMES = ("X", "Y", "Z")

    def __init__(self, per_image: Sequence[dict], metric: str, parent=None) -> None:
        super().__init__(parent)
        self._per_image = per_image
        self._metric = metric
        self.setMinimumHeight(250)

    def paintEvent(self, event) -> None:
        painter = QPainter(self); painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor("#f8fafc"))
        values = [item.get(self._metric) for item in self._per_image]
        values = [item for item in values if isinstance(item, list) and 2 <= len(item) <= 3]
        if not values:
            painter.setPen(QColor("#64748b"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter,
                             "Spatial data is unavailable. Run Compute statistics again.")
            return
        axis_count = max(len(item) for item in values)
        numbers = [float(value) for item in values for value in item]
        minimum, maximum = min(numbers), max(numbers)
        if minimum == maximum:
            minimum_span = 1.0 if self._metric == "shape" else 0.01
            span = max(abs(minimum) * 0.1, minimum_span)
            minimum, maximum = minimum - span, maximum + span
        else:
            padding = (maximum - minimum) * 0.05
            minimum, maximum = max(0.0, minimum - padding), maximum + padding
        if self._metric == "shape":
            minimum, maximum = floor(minimum), ceil(maximum)
        margin_left, margin_right, margin_top, margin_bottom = 80, 24, 24, 66
        plot = self.rect().adjusted(margin_left, margin_top, -margin_right, -margin_bottom)
        painter.setPen(QColor("#cbd5e1")); painter.drawRect(plot)
        metrics = QFontMetrics(painter.font())
        painter.save()
        painter.translate(14, plot.center().y()); painter.rotate(-90)
        painter.setPen(QColor("#64748b"))
        painter.drawText(-plot.height() // 2, -8, plot.height(), 16,
                         Qt.AlignmentFlag.AlignCenter, "Dimension")
        painter.restore()
        for axis in range(axis_count):
            y = plot.top() + (plot.height() * axis // max(1, axis_count - 1))
            painter.setPen(QColor("#e2e8f0")); painter.drawLine(plot.left(), y, plot.right(), y)
            painter.setPen(QColor("#334155"))
            painter.drawText(26, y - metrics.height() // 2, 46, metrics.height(),
                             Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                             self._AXIS_NAMES[axis])
        image_points: list[list[QPointF]] = []
        for image_index, item in enumerate(self._per_image):
            item_values = item.get(self._metric)
            if not isinstance(item_values, list):
                continue
            # A tiny deterministic offset keeps coincident image values visible.
            jitter = ((image_index % 5) - 2) * 2
            points = []
            for axis, value in enumerate(item_values[:axis_count]):
                if not isinstance(value, (int, float)):
                    continue
                y = plot.top() + (plot.height() * axis / max(1, axis_count - 1))
                x = plot.left() + (float(value) - minimum) * plot.width() / (maximum - minimum)
                points.append(QPointF(x, y + jitter))
            image_points.append(points)
        # Connect each image's X, Y and Z values, so the values belonging together stay visible.
        line_colour = QColor("#2f6fca"); line_colour.setAlpha(70)
        painter.setPen(QPen(line_colour, 1.5)); painter.setBrush(Qt.BrushStyle.NoBrush)
        for points in image_points:
            if len(points) > 1:
                painter.drawPolyline(QPolygonF(points))
        point_colour = QColor("#2f6fca"); point_colour.setAlpha(145)
        painter.setBrush(point_colour); painter.setPen(Qt.PenStyle.NoPen)
        for points in image_points:
            for point in points:
                painter.drawEllipse(point, 6, 6)
        painter.setPen(QColor("#64748b"))
        if self._metric == "shape":
            step = max(1, ceil((maximum - minimum) / 4))
            tick_values = list(range(int(minimum), int(maximum) + 1, step))
            if tick_values[-1] != maximum:
                tick_values.append(maximum)
        else:
            tick_values = [minimum + tick * (maximum - minimum) / 4 for tick in range(5)]
        for value in tick_values:
            x = int(plot.left() + (value - minimum) * plot.width() / (maximum - minimum))
            label_text = str(value) if self._metric == "shape" else f"{value:.4g}"
            painter.drawLine(x, plot.bottom(), x, plot.bottom() + 5)
            painter.drawText(x - metrics.horizontalAdvance(label_text) // 2, plot.bottom() + 22, label_text)
        label = "Voxel count" if self._metric == "shape" else "Spacing (mm)"
        painter.drawText(plot.left(), plot.bottom() + 36, plot.width(), 18,
                         Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter, label)


_TABLE_STYLE = f"""
QTableView {{
    background: {style.PANEL}; border: 0; gridline-color: transparent; outline: 0;
    selection-background-color: {style.BLUE_SOFT}; selection-color: {style.TEXT};
}}
QTableView::item {{ border-bottom: 1px solid #f0f3f7; padding: 0 8px; }}
QTableView::item:selected {{ background: {style.BLUE_SOFT}; color: {style.TEXT}; }}
"""


def _table_view(model: QAbstractTableModel) -> QTableView:
    """A read-only view like the other tables (left-aligned headers, no grid, row selection) with fixed row
    heights and column widths: nothing is measured per cell, so thousands of rows and columns open at once."""
    table = QTableView()
    table.setModel(model)
    model.setParent(table)
    table.setStyleSheet(_TABLE_STYLE)
    header = table.horizontalHeader()
    header.setDefaultAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
    header.setHighlightSections(False)
    header.setTextElideMode(Qt.TextElideMode.ElideRight)
    header.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
    vertical = table.verticalHeader()
    vertical.hide()
    vertical.setSectionResizeMode(QHeaderView.ResizeMode.Fixed)
    vertical.setDefaultSectionSize(32)
    table.setShowGrid(False)
    table.setWordWrap(False)
    table.setTextElideMode(Qt.TextElideMode.ElideRight)
    table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
    table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
    table.setHorizontalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
    return table


class _ClassModel(QAbstractTableModel):
    """Class summary: ID, name, images containing the class and total voxels."""

    HEADERS = ("ID", "Class", "Images", "Voxels")
    TIPS = ("Label ID in dataset.json", "Class name in dataset.json", "Images containing the class",
            "Total voxels of the class in all label files")

    def __init__(self, classes: Sequence[dict], undefined: set[int]) -> None:
        super().__init__()
        self._classes, self._undefined = classes, undefined

    def rowCount(self, parent: QModelIndex | None = None) -> int:
        return 0 if parent is not None and parent.isValid() else len(self._classes)

    def columnCount(self, parent: QModelIndex | None = None) -> int:
        return 0 if parent is not None and parent.isValid() else len(self.HEADERS)

    def headerData(self, section: int, orientation, role=Qt.ItemDataRole.DisplayRole):
        if orientation == Qt.Orientation.Horizontal and role == Qt.ItemDataRole.DisplayRole:
            return self.HEADERS[section]
        if orientation == Qt.Orientation.Horizontal and role == Qt.ItemDataRole.ToolTipRole:
            return self.TIPS[section]
        if orientation == Qt.Orientation.Horizontal and role == Qt.ItemDataRole.TextAlignmentRole and section != 1:
            return int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)  # above the numbers
        return None

    def data(self, index: QModelIndex, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        item = self._classes[index.row()]
        class_id, column = int(item.get("id", -1)), index.column()
        if role in (Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.ToolTipRole):
            if column == 0:
                return str(class_id)
            if column == 1:
                name = _class_name(item, self._undefined)
                if role == Qt.ItemDataRole.ToolTipRole and class_id in self._undefined:
                    return f"Label ID {class_id} occurs in the label files but is not defined in dataset.json."
                return name
            return f"{int(item.get('images' if column == 2 else 'voxels', 0)):,}"
        if role == Qt.ItemDataRole.ForegroundRole and class_id in self._undefined:
            return QColor(style.ORANGE)
        if role == Qt.ItemDataRole.TextAlignmentRole and column != 1:
            return int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        return None


class _ImageModel(QAbstractTableModel):
    """Per-image details: case, validity, dimensions, spacing, (labeled share,) then the voxels of every class.
    Cells are formatted when they are painted, so a 5000 x 500 table costs nothing up front."""

    def __init__(self, per_image: Sequence[dict], classes: Sequence[dict], headers: list[str],
                 labeled_data: bool, undefined: set[int]) -> None:
        super().__init__()
        self._per_image, self._labeled_data = per_image, labeled_data
        self._class_ids = [str(item.get("id", "")) for item in classes]
        self._headers = headers
        self._class_headers = [f"{item.get('id')}: {_class_name(item, undefined)}" for item in classes]
        self._first_class = len(headers)

    def rowCount(self, parent: QModelIndex | None = None) -> int:
        return 0 if parent is not None and parent.isValid() else len(self._per_image)

    def columnCount(self, parent: QModelIndex | None = None) -> int:
        return 0 if parent is not None and parent.isValid() else self._first_class + len(self._class_ids)

    def headerData(self, section: int, orientation, role=Qt.ItemDataRole.DisplayRole):
        if (orientation == Qt.Orientation.Horizontal and role == Qt.ItemDataRole.TextAlignmentRole
                and section >= self._first_class):
            return int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        if orientation != Qt.Orientation.Horizontal or role not in (Qt.ItemDataRole.DisplayRole,
                                                                     Qt.ItemDataRole.ToolTipRole):
            return None
        if section < self._first_class:
            return self._headers[section]
        return self._class_headers[section - self._first_class]

    def _text(self, image: dict, column: int) -> str:
        if column == 0:
            return image["image"]
        if column == 1:
            findings = image["validity"]
            return "; ".join(str(finding) for finding in findings) if findings else "Valid"
        if column == 2:
            return ", ".join(str(value) for value in image["shape"])
        if column == 3:
            return ", ".join(f"{float(value):.4f}" for value in image["spacing"])
        if column == 4 and self._labeled_data:
            percentage = _labeled_data_percentage(image["shape"], image["voxels"])
            return f"{percentage:.2f}%" if percentage is not None else "—"
        return f"{int(image['voxels'].get(self._class_ids[column - self._first_class], 0)):,}"

    def data(self, index: QModelIndex, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        image, column = self._per_image[index.row()], index.column()
        if role == Qt.ItemDataRole.DisplayRole:
            return self._text(image, column)
        if role == Qt.ItemDataRole.ToolTipRole and column < 2:
            return ("\n".join(str(finding) for finding in image["validity"]) or "Valid") if column else image["image"]
        if role == Qt.ItemDataRole.ForegroundRole and column == 1:
            return QColor("#a74400" if image["validity"] else "#315f1f")
        if role == Qt.ItemDataRole.TextAlignmentRole and column >= self._first_class:
            return int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        return None


class LabelStatisticsView(QWidget):
    """Inline label analysis with class, per-image, and geometry views."""

    def __init__(self, analysis: dict, sparse_labels: bool = False, parent=None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        classes = analysis["classes"]
        self._classes = classes
        per_image = analysis["per_image"]
        self._show_labeled_data = sparse_labels
        undefined = undefined_ids(analysis)
        axis_count = max((len(item[key]) for item in per_image for key in ("shape", "spacing")), default=0)
        axes = ", ".join(_AxisScatterChart._AXIS_NAMES[:axis_count])
        dimensions_header = f"Dimensions ({axes})" if axes else "Dimensions"
        spacing_header = f"Voxel spacing ({axes}; mm)" if axes else "Voxel spacing (mm)"
        # dataset.json's classes only; IDs found in the labels but not defined there are flagged next to it
        defined = sum(int(item.get("id", -1)) not in undefined for item in classes)
        summary_parts = [f"{len(per_image)} label files", f"{defined} classes", f"{axis_count}D"]
        summary = QLabel(" · ".join(summary_parts))
        summary.setObjectName("analysisSummary")
        summary.setProperty("muted", True)
        summary.setWordWrap(True)  # wraps instead of being cut beside the badges in narrow windows
        controls = QHBoxLayout(); controls.setContentsMargins(16, 8, 16, 4); controls.setSpacing(10)
        controls.addWidget(summary, 1)
        self.header_row = controls  # pages can add status badges next to the summary
        if undefined:
            listed = ", ".join(str(value) for value in sorted(undefined))
            plural = "s" if len(undefined) != 1 else ""
            self.undefined_badge = badge(
                f"! undefined ID{plural}: {listed}" if len(undefined) <= 3 else f"! {len(undefined)} undefined IDs",
                "orange",
                f"The label files contain the values {listed}, which dataset.json does not define. nnU-Net "
                "fails on them: relabel those voxels or add the classes to dataset.json (the Per image tab "
                "lists the files).")
            controls.addWidget(self.undefined_badge)
        background = QCheckBox("Hide background")
        controls.addWidget(background); layout.addLayout(controls)
        views = QTabWidget(); layout.addWidget(views, 1)
        # in a narrow window the tabs shorten their titles (full title on hover) instead of scrolling
        views.tabBar().setUsesScrollButtons(False)
        views.tabBar().setElideMode(Qt.TextElideMode.ElideRight)
        views.tabBar().setStyleSheet("QTabBar::tab { padding: 7px 9px; }")
        table = _table_view(_ClassModel(classes, undefined)); self.class_table = table
        header = table.horizontalHeader()
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        for column, width in ((0, 56), (2, 90), (3, 120)):
            header.resizeSection(column, width)
        tabs = [(table, "Classes", "Class summary: images and voxels per class")]
        image_headers = ["Case", "Label validity", dimensions_header, spacing_header]
        if self._show_labeled_data:
            image_headers.append("Labeled data")
        image_table = _table_view(_ImageModel(per_image, classes, image_headers, self._show_labeled_data,
                                              undefined))
        self.image_table = image_table
        header = image_table.horizontalHeader()
        header.setDefaultSectionSize(120)
        for column, width in enumerate((200, 200, 150, 190, 110)[:len(image_headers)]):
            header.resizeSection(column, width)
        tabs.append((image_table, "Per image", "Per-image details: validity, dimensions, spacing and class voxels"))
        self.voxel_chart = _ClassChart(classes, undefined, len(per_image)); self.voxel_chart.set_metric("voxels")
        self.image_chart = _ClassChart(classes, undefined, len(per_image)); self.image_chart.set_metric("images")
        tabs += [(_chart_area(self.voxel_chart), "Voxels", "Class voxel count: total voxels per class"),
                 (_chart_area(self.image_chart), "Coverage", "Class coverage: number of images containing each class"),
                 (_AxisScatterChart(per_image, "shape"), "Shape", "Image shape along X, Y and Z of every label file"),
                 (_AxisScatterChart(per_image, "spacing"), "Spacing", "Voxel spacing along X, Y and Z")]
        for widget, title, tooltip in tabs:  # short titles: all tabs fit next to the dataset.json panel
            views.setTabToolTip(views.addTab(widget, title), tooltip)
        background.toggled.connect(self._set_ignore_background)

    def _set_ignore_background(self, ignore: bool) -> None:
        self.voxel_chart.set_ignore_background(ignore)
        self.image_chart.set_ignore_background(ignore)
        for row, item in enumerate(self._classes):
            self.class_table.setRowHidden(row, ignore and int(item.get("id", -1)) == 0)
        first_class_column = 5 if self._show_labeled_data else 4
        for column, item in enumerate(self._classes, start=first_class_column):
            self.image_table.setColumnHidden(column, ignore and int(item.get("id", -1)) == 0)


def _labeled_data_percentage(shape: object, counts: object) -> float | None:
    """Derive non-ignore voxel coverage from the saved image shape and class counts."""
    if not isinstance(shape, list) or not shape or not all(isinstance(value, int) and value > 0 for value in shape):
        return None
    if not isinstance(counts, dict) or not all(isinstance(value, int) and value >= 0 for value in counts.values()):
        return None
    return 100 * sum(counts.values()) / prod(shape)
