"""Source uncertainty bands on the shared review transport."""
from PySide6.QtCore import QRectF
from PySide6.QtGui import QColor, QPainter
from PySide6.QtWidgets import QSlider


class ReviewTimeline(QSlider):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.intervals = []

    def set_intervals(self, intervals):
        self.intervals = list(intervals)
        self.setToolTip("Orange bands mark weak source observations. Use Review next uncertain interval to inspect them."
                        if self.intervals else "Shared video and motion timeline")
        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        span = self.maximum()-self.minimum()
        if not self.intervals or span <= 0:
            return
        painter = QPainter(self)
        painter.setPen(QColor("#e6a23c"))
        painter.setBrush(QColor("#e6a23c"))
        for start, end in self.intervals:
            x0 = 7+(start*1000-self.minimum())/span*max(1, self.width()-14)
            x1 = 7+(end*1000-self.minimum())/span*max(1, self.width()-14)
            painter.drawRoundedRect(QRectF(x0, self.height()-4, max(2, x1-x0), 3), 1, 1)
        painter.end()
