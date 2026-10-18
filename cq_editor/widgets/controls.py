"""Small shared controls for the chat and provider settings UI."""

from PyQt5.QtCore import Qt, QPointF, QSize
from PyQt5.QtGui import QColor, QPainter, QPainterPath, QPen
from PyQt5.QtWidgets import QCheckBox, QStyle, QStyleOptionButton, QLabel, QSizePolicy


class ElidedLabel(QLabel):
    """Single-line secondary text that shrinks without forcing wider docks."""

    def __init__(self, text="", parent=None):
        super().__init__(text, parent)
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        self.setMinimumWidth(0)

    def setText(self, text):
        super().setText(text)
        self.setToolTip(text)

    def sizeHint(self):
        return QSize(0, self.fontMetrics().height() + 4)

    def minimumSizeHint(self):
        return self.sizeHint()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setPen(self.palette().windowText().color())
        text = self.fontMetrics().elidedText(
            self.text(), Qt.ElideMiddle, self.contentsRect().width()
        )
        painter.drawText(self.contentsRect(), self.alignment(), text)


class CheckBox(QCheckBox):
    """Keep native checkbox behavior and draw a crisp mark over its styled box.

    A stylesheet background replaces Qt's native checkmark, so a checked box
    otherwise appears as a solid square. Paint the mark in the actual indicator
    rectangle so it follows font scaling and right-to-left layouts.
    """

    def __init__(self, text="", parent=None):
        super().__init__(text, parent)
        self.setCursor(Qt.PointingHandCursor)

    def paintEvent(self, event):
        super().paintEvent(event)
        if self.checkState() == Qt.Unchecked:
            return
        option = QStyleOptionButton()
        self.initStyleOption(option)
        rect = self.style().subElementRect(QStyle.SE_CheckBoxIndicator, option, self)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(
            QPen(QColor("#ffffff"), 1.8, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
        )

        def point(x, y):
            return QPointF(rect.x() + rect.width() * x, rect.y() + rect.height() * y)

        path = QPainterPath()
        if self.checkState() == Qt.PartiallyChecked:
            path.moveTo(point(0.25, 0.5))
            path.lineTo(point(0.75, 0.5))
        else:
            path.moveTo(point(0.24, 0.50))
            path.lineTo(point(0.43, 0.69))
            path.lineTo(point(0.77, 0.31))
        painter.drawPath(path)
