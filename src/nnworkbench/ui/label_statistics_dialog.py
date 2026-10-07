from __future__ import annotations

from collections.abc import Sequence
from math import ceil, floor, prod

from qtpy.QtCore import QPointF, Qt
from qtpy.QtGui import QColor, QFontMetrics, QPainter, QPen, QPolygonF
from qtpy.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)


class _ClassChart(QWidget):
    """Small dependency-free bar chart for either cached class metric."""

    def __init__(self, classes: Sequence[dict], parent=None) -> None:
        super().__init__(parent)
        self._all_classes = classes
        self._metric = "voxels"
        self._ignore_background = False
        self.setMinimumHeight(210)

    def set_metric(self, metric: str) -> None:
        self._metric = metric
        self.update()

    def set_ignore_background(self, ignore: bool) -> None:
        self._ignore_background = ignore
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self); painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor("#f8fafc"))
        classes = [item for item in self._all_classes
                   if not self._ignore_background or int(item.get("id", -1)) != 0]
        if not classes:
            painter.setPen(QColor("#64748b")); painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "No class data")
            return
        margin_left, margin_right, margin_top, margin_bottom = 64, 18, 24, 42
        plot = self.rect().adjusted(margin_left, margin_top, -margin_right, -margin_bottom)
        maximum = max(int(item.get(self._metric, 0)) for item in classes) or 1
        painter.setPen(QColor("#cbd5e1")); painter.drawLine(plot.bottomLeft(), plot.bottomRight())
        painter.drawLine(plot.bottomLeft(), plot.topLeft())
        painter.save()
        painter.translate(14, plot.center().y()); painter.rotate(-90)
        axis_label = "Voxels" if self._metric == "voxels" else "Number of images"
        painter.setPen(QColor("#64748b"))
        painter.drawText(-plot.height() // 2, -8, plot.height(), 16,
                         Qt.AlignmentFlag.AlignCenter, axis_label)
        painter.restore()
        bar_width = max(8, (plot.width() - 8 * (len(classes) - 1)) // max(1, len(classes)))
        step = plot.width() / max(1, len(classes))
        metrics = QFontMetrics(painter.font())
        for index, item in enumerate(classes):
            value = int(item.get(self._metric, 0)); height = int(plot.height() * value / maximum)
            x = int(plot.left() + index * step + (step - bar_width) / 2)
            bar = plot.adjusted(x - plot.left(), plot.height() - height, x + bar_width - plot.right(), 0)
            painter.fillRect(bar, QColor("#2f6fca"))
            painter.setPen(QColor("#334155"))
            label = f"{item.get('id')}: {item.get('name')}"
            painter.drawText(x, plot.bottom() + 17, int(step), 18, Qt.AlignmentFlag.AlignLeft, metrics.elidedText(label, Qt.TextElideMode.ElideRight, int(step)))
        painter.setPen(QColor("#64748b")); painter.drawText(4, plot.top() + 5, f"{maximum:,}")


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
                             "Spatial data is unavailable. Run Analyze Labels again.")
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


class LabelStatisticsView(QWidget):
    """Inline label analysis with class, per-image, and geometry views."""

    def __init__(self, analysis: dict, annotation_workflow_enabled: bool = False, parent=None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        classes = analysis["classes"]
        self._classes = classes
        per_image = analysis["per_image"]
        self._show_labeled_data = annotation_workflow_enabled
        shapes = [tuple(item["shape"]) for item in per_image]
        spacings = [tuple(round(float(value), 8) for value in item["spacing"]) for item in per_image]
        axis_count = max((len(value) for value in (*shapes, *spacings)), default=0)
        axes = ", ".join(_AxisScatterChart._AXIS_NAMES[:axis_count])
        dimensions_header = f"Dimensions ({axes})" if axes else "Dimensions"
        spacing_header = f"Voxel spacing ({axes}; mm)" if axes else "Voxel spacing (mm)"
        summary_parts = [f"{len(per_image)} label files", f"{len(classes)} classes"]
        summary_parts.append(f"{axis_count}D")
        summary = QLabel("Label analysis · " + " · ".join(summary_parts))
        summary.setObjectName("analysisSummary")
        summary.setProperty("muted", True)
        controls = QHBoxLayout(); controls.setContentsMargins(16, 8, 16, 4); controls.setSpacing(10)
        controls.addWidget(summary)
        self.header_row = controls  # pages can add status badges next to the summary
        controls.addStretch()
        background = QCheckBox("Hide background")
        controls.addWidget(background); layout.addLayout(controls)
        views = QTabWidget(); layout.addWidget(views, 1)
        table = QTableWidget(len(classes), 3); self.class_table = table
        table.setHorizontalHeaderLabels(("Class", "Images containing class", "Total voxels"))
        table.verticalHeader().hide(); table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        for row, item in enumerate(classes):
            table.setItem(row, 0, QTableWidgetItem(f"{item.get('id')}: {item.get('name')}"))
            table.setItem(row, 1, QTableWidgetItem(f"{int(item.get('images', 0)):,}"))
            table.setItem(row, 2, QTableWidgetItem(f"{int(item.get('voxels', 0)):,}"))
        views.addTab(table, "Class summary")
        class_column = 5 if self._show_labeled_data else 4
        image_table = QTableWidget(len(per_image), len(classes) + class_column)
        self.image_table = image_table
        image_table.setTextElideMode(Qt.TextElideMode.ElideNone); image_table.setWordWrap(True)
        image_headers = ["Image", "Label validity", dimensions_header, spacing_header]
        if self._show_labeled_data:
            image_headers.append("Labeled data")
        image_table.setHorizontalHeaderLabels((
            *image_headers,
            *(f"{item.get('id')}: {item.get('name')}" for item in classes),
        ))
        image_table.verticalHeader().hide(); image_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        image_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        for column in range(1, len(classes) + class_column):
            image_table.horizontalHeader().setSectionResizeMode(column, QHeaderView.ResizeMode.ResizeToContents)
        for row, image in enumerate(per_image):
            counts = image["voxels"]
            filename = image["image"]
            image_item = QTableWidgetItem(filename); image_item.setToolTip(filename)
            image_table.setItem(row, 0, image_item)
            findings = image["validity"]
            validity, colour = (
                ("\n".join(str(finding) for finding in findings), QColor("#a74400"))
                if findings else ("Valid", QColor("#315f1f"))
            )
            validity_item = QTableWidgetItem(validity)
            validity_item.setForeground(colour)
            image_table.setItem(row, 1, validity_item)
            shape = image["shape"]
            spacing = image["spacing"]
            image_table.setItem(row, 2, QTableWidgetItem(", ".join(str(value) for value in shape)))
            image_table.setItem(row, 3, QTableWidgetItem(", ".join(f"{float(value):.4f}" for value in spacing)))
            if self._show_labeled_data:
                percentage = _labeled_data_percentage(shape, counts)
                percentage_text = f"{percentage:.2f}%" if percentage is not None else "—"
                image_table.setItem(row, 4, QTableWidgetItem(percentage_text))
            for column, class_item in enumerate(classes, start=class_column):
                class_id = str(class_item.get("id", ""))
                image_table.setItem(row, column, QTableWidgetItem(f"{int(counts.get(class_id, 0)):,}"))
        image_table.resizeRowsToContents()
        views.addTab(image_table, "Per-image details")
        self.voxel_chart = _ClassChart(classes); self.voxel_chart.set_metric("voxels")
        self.image_chart = _ClassChart(classes); self.image_chart.set_metric("images")
        views.addTab(self.voxel_chart, "Class voxel count")
        views.addTab(self.image_chart, "Class coverage")
        views.addTab(_AxisScatterChart(per_image, "shape"), "Image shape")
        views.addTab(_AxisScatterChart(per_image, "spacing"), "Image spacing")
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
    if not isinstance(shape, list) or not all(isinstance(value, int) and value > 0 for value in shape):
        return None
    if not isinstance(counts, dict) or not all(isinstance(value, int) and value >= 0 for value in counts.values()):
        return None
    return 100 * sum(counts.values()) / prod(shape)
