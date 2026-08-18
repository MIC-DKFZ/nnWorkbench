"""Editable combo box with a consistently visible dropdown affordance."""

from __future__ import annotations

from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QColor, QPainter, QPolygon
from PySide6.QtWidgets import QComboBox, QStyle, QStyleOptionComboBox


class EditableDropdownComboBox(QComboBox):
    """Keep text editable while drawing a clear arrow over custom Qt stylesheets."""

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt API spelling
        super().paintEvent(event)
        option = QStyleOptionComboBox(); self.initStyleOption(option)
        arrow = self.style().subControlRect(QStyle.ComplexControl.CC_ComboBox, option,
                                            QStyle.SubControl.SC_ComboBoxArrow, self)
        center = arrow.center()
        painter = QPainter(self); painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor("#6b7280") if self.isEnabled() else QColor("#a0a0a0"))
        painter.drawPolygon(QPolygon((QPoint(center.x() - 5, center.y() - 2),
                                      QPoint(center.x() + 5, center.y() - 2),
                                      QPoint(center.x(), center.y() + 4))))
        painter.end()
