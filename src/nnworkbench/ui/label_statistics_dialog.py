from __future__ import annotations

from collections.abc import Sequence

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFontMetrics, QPainter
from PySide6.QtWidgets import (
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
        margin_left, margin_right, margin_top, margin_bottom = 46, 18, 24, 42
        plot = self.rect().adjusted(margin_left, margin_top, -margin_right, -margin_bottom)
        maximum = max(int(item.get(self._metric, 0)) for item in classes) or 1
        painter.setPen(QColor("#cbd5e1")); painter.drawLine(plot.bottomLeft(), plot.bottomRight())
        painter.drawLine(plot.bottomLeft(), plot.topLeft())
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


class LabelStatisticsView(QWidget):
    """Inline cached label statistics with table and dedicated plot tabs."""

    def __init__(self, analysis: dict, parent=None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        classes = [item for item in analysis.get("classes", []) if isinstance(item, dict)]
        self._classes = classes
        controls = QHBoxLayout(); controls.addStretch()
        background = QCheckBox("Ignore background")
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
        views.addTab(table, "Table")
        per_image = [item for item in analysis.get("per_image", []) if isinstance(item, dict)]
        if per_image:
            image_table = QTableWidget(len(per_image), len(classes) + 1)
            self.image_table: QTableWidget | None = image_table
            image_table.setHorizontalHeaderLabels(("Image", *(f"{item.get('id')}: {item.get('name')}" for item in classes)))
            image_table.verticalHeader().hide(); image_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
            image_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
            for column in range(1, len(classes) + 1):
                image_table.horizontalHeader().setSectionResizeMode(column, QHeaderView.ResizeMode.ResizeToContents)
            for row, image in enumerate(per_image):
                counts = image.get("voxels", {})
                image_table.setItem(row, 0, QTableWidgetItem(str(image.get("image", ""))))
                for column, class_item in enumerate(classes, start=1):
                    class_id = str(class_item.get("id", ""))
                    value = counts.get(class_id, 0) if isinstance(counts, dict) else 0
                    image_table.setItem(row, column, QTableWidgetItem(f"{int(value):,}"))
            views.addTab(image_table, "Images")
        else:
            self.image_table = None
            unavailable = QWidget(); unavailable_layout = QVBoxLayout(unavailable)
            unavailable_layout.addWidget(QLabel("Per-image counts are unavailable. Run Analyze Labels again to create them."))
            unavailable_layout.addStretch(); views.addTab(unavailable, "Images")
        self.voxel_chart = _ClassChart(classes); self.voxel_chart.set_metric("voxels")
        self.image_chart = _ClassChart(classes); self.image_chart.set_metric("images")
        views.addTab(self.voxel_chart, "Voxel count")
        views.addTab(self.image_chart, "Images containing class")
        background.toggled.connect(self._set_ignore_background)

    def _set_ignore_background(self, ignore: bool) -> None:
        self.voxel_chart.set_ignore_background(ignore)
        self.image_chart.set_ignore_background(ignore)
        for row, item in enumerate(self._classes):
            self.class_table.setRowHidden(row, ignore and int(item.get("id", -1)) == 0)
        if self.image_table is not None:
            for column, item in enumerate(self._classes, start=1):
                self.image_table.setColumnHidden(column, ignore and int(item.get("id", -1)) == 0)
