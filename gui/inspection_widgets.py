"""Interactive image inspection and temporary enlargement of existing plots."""
from PyQt5.QtCore import Qt, QPointF, QRectF, pyqtSignal
from PyQt5.QtGui import QPainter, QFont, QFontMetrics, QColor, QPen
from PyQt5.QtWidgets import QDialog, QVBoxLayout, QSizePolicy
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT
from gui.widgets import ResizablePixmapLabel


def attach_spectrum_cursor(canvas, axes, wavelength_axis=True):
    """Snap to the closest visible spectral curve and its measured band."""
    import numpy as np
    previous = getattr(canvas, '_spectral_cursor_connections', ())
    for connection in previous:
        canvas.mpl_disconnect(connection)
    marker, = axes.plot([],[], 'o', color='#ffd166', markersize=6, zorder=10, label='_nolegend_')
    annotation = axes.annotate('',xy=(0,0),xytext=(12,12),textcoords='offset points',
        fontsize=9,color='#f4f4f4',zorder=11,
        bbox=dict(boxstyle='round,pad=.4',facecolor='#252525',edgecolor='#ffd166',alpha=.95))
    annotation.set_visible(False)
    canvas._spectral_cursor_annotation = annotation
    def hide(_event=None):
        if annotation.get_visible():
            annotation.set_visible(False);marker.set_data([],[]);canvas.draw_idle()
    def motion(event):
        if event.inaxes is not axes or event.xdata is None:
            hide();return
        candidates=[]
        for line in axes.lines:
            if line is marker or not line.get_visible():
                continue
            x,y=np.asarray(line.get_xdata(),float),np.asarray(line.get_ydata(),float)
            valid=np.flatnonzero(np.isfinite(x)&np.isfinite(y))
            if not len(valid):continue
            index=valid[np.argmin(np.abs(x[valid]-event.xdata))]
            point=(float(x[index]),float(y[index]))
            screen=axes.transData.transform(point)
            distance=float((screen[0]-event.x)**2+(screen[1]-event.y)**2)
            candidates.append((distance,point,line.get_label()))
        if not candidates:hide();return
        _,(x,y),name=min(candidates,key=lambda value:value[0])
        marker.set_data([x],[y]);annotation.xy=(x,y)
        label='' if str(name).startswith('_') else str(name)+'\n'
        annotation.set_text(f'{label}{x:g} {"nm" if wavelength_axis else "(band)"}\nIntensity: {y:.5g}')
        xlo,xhi=axes.get_xlim();ylo,yhi=axes.get_ylim()
        left=x>xlo+.65*(xhi-xlo);below=y>ylo+.7*(yhi-ylo)
        annotation.set_position((-12 if left else 12,-12 if below else 12))
        annotation.set_ha('right' if left else 'left');annotation.set_va('top' if below else 'bottom')
        annotation.set_visible(True);canvas.draw_idle()
    canvas._spectral_cursor_connections = [canvas.mpl_connect('motion_notify_event',motion),
        canvas.mpl_connect('axes_leave_event',hide),canvas.mpl_connect('figure_leave_event',hide)]


def enlarge_plot(canvas, parent=None):
    original_parent = canvas.parentWidget()
    def containing(layout):
        if layout.indexOf(canvas) >= 0:
            return layout
        for index in range(layout.count()):
            child = layout.itemAt(index).layout()
            if child is not None:
                found = containing(child)
                if found is not None:
                    return found
        return None
    original_layout = containing(original_parent.layout())
    if original_layout is None:
        return
    index = original_layout.indexOf(canvas)
    original_minimum, original_maximum = canvas.minimumSize(), canvas.maximumSize()
    canvas.setMinimumSize(0, 0)
    canvas.setMaximumSize(16777215, 16777215)
    dialog = QDialog(parent)
    dialog.setWindowTitle('ROI spectrum — enlarged view')
    dialog.setWindowFlags(dialog.windowFlags() | Qt.WindowMaximizeButtonHint)
    dialog.resize(1150, 780)
    layout = QVBoxLayout(dialog)
    original_toolbar = getattr(canvas, 'toolbar', None)
    toolbar = NavigationToolbar2QT(canvas, dialog)
    toolbar.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
    toolbar.setStyleSheet('QToolBar { background:#d6d6d6; color:#222; border:0; } QToolButton { color:#222; }')
    layout.addWidget(toolbar)
    layout.addWidget(canvas, 1)
    try:
        dialog.exec_()
    finally:
        for name in ('_id_press','_id_release','_id_drag'):
            callback = getattr(toolbar, name, None)
            if callback is not None:
                canvas.mpl_disconnect(callback)
        canvas.toolbar = original_toolbar
        layout.removeWidget(canvas)
        canvas.setParent(original_parent)
        canvas.setMinimumSize(original_minimum)
        canvas.setMaximumSize(original_maximum)
        original_layout.insertWidget(index, canvas, 1)
        canvas.show()
        canvas.draw_idle()
        dialog.deleteLater()


class ZoomablePixmapLabel(ResizablePixmapLabel):
    viewChanged = pyqtSignal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.zoom = 1.
        self.pan = QPointF()
        self._drag = None
        self.roi_annotations = []
        self.setToolTip('Wheel: zoom at pointer. Right or middle drag: pan. Fit image: reset. Left click: hold spectrum.')

    def setPixmap(self, pixmap):
        changed = self._orig_pix is None or self._orig_pix.size() != pixmap.size()
        super().setPixmap(pixmap)
        if changed:
            self.reset_view()

    def clear(self):
        super().clear()
        self.reset_view()

    def image_rect(self):
        if self._orig_pix is None or self._orig_pix.isNull():
            return QRectF()
        scale = min(self.width()/self._orig_pix.width(), self.height()/self._orig_pix.height()) * self.zoom
        width, height = self._orig_pix.width()*scale, self._orig_pix.height()*scale
        left = (self.width()-width)/2 + self.pan.x()
        top = (self.height()-height)/2 + self.pan.y()
        left = (self.width()-width)/2 if width <= self.width() else max(self.width()-width, min(0., left))
        top = (self.height()-height)/2 if height <= self.height() else max(self.height()-height, min(0., top))
        self.pan = QPointF(left-(self.width()-width)/2, top-(self.height()-height)/2)
        return QRectF(left, top, width, height)

    def image_coordinates(self, position):
        rect = self.image_rect()
        if rect.isEmpty() or not rect.contains(QPointF(position)):
            return None
        x = int((position.x()-rect.left())/rect.width()*self._orig_pix.width())
        y = int((position.y()-rect.top())/rect.height()*self._orig_pix.height())
        return min(self._orig_pix.width()-1,x), min(self._orig_pix.height()-1,y)

    def reset_view(self):
        self.zoom, self.pan = 1., QPointF()
        self.update()
        self.viewChanged.emit(self.zoom)

    def zoom_by(self, factor, position=None):
        position = QPointF(position) if position is not None else QPointF(self.rect().center())
        old = self.image_rect()
        if old.isEmpty():
            return
        previous = self.zoom
        self.zoom = max(1., min(32., self.zoom*factor))
        ratio = self.zoom/previous
        width, height = old.width()*ratio, old.height()*ratio
        left = position.x() - (position.x()-old.left())*ratio
        top = position.y() - (position.y()-old.top())*ratio
        self.pan = QPointF(left-(self.width()-width)/2, top-(self.height()-height)/2)
        self.image_rect()
        self.update()
        self.viewChanged.emit(self.zoom)

    def paintEvent(self, event):
        if self._orig_pix is None or self._orig_pix.isNull():
            return super().paintEvent(event)
        painter = QPainter(self)
        painter.fillRect(self.rect(), self.palette().window())
        painter.setRenderHint(QPainter.SmoothPixmapTransform, self.zoom <= 1.)
        painter.drawPixmap(self.image_rect(), self._orig_pix, QRectF(self._orig_pix.rect()))
        # Text is drawn after scaling the image, so zoom never enlarges labels.
        from gui.roi_labels import place_roi_labels, roi_leader_endpoints
        image = self.image_rect()
        sx, sy = image.width()/self._orig_pix.width(), image.height()/self._orig_pix.height()
        bounds = [QRectF(image.left()+x0*sx, image.top()+y0*sy,
                         (x1-x0+1)*sx, (y1-y0+1)*sy)
                  for _, (x0, y0, x1, y1), _ in self.roi_annotations]
        font = QFont('Arial')
        font.setPixelSize(12)
        painter.setFont(font)
        metrics = QFontMetrics(font)
        sizes = [(metrics.horizontalAdvance(name)+8, metrics.height()+4) for name, _, _ in self.roi_annotations]
        placements = place_roi_labels(bounds, sizes, QRectF(self.rect()).adjusted(3, 3, -3, -3))
        painter.setRenderHint(QPainter.Antialiasing)
        viewport = image.intersected(QRectF(self.rect()))
        for (name, _, outlines), placement in zip(self.roi_annotations, placements):
            if placement is None:
                continue
            rect, start, end = placement
            paths = [[(image.left()+(x+.5)*sx, image.top()+(y+.5)*sy) for x, y in points]
                     for points in outlines]
            endpoints = roi_leader_endpoints(paths, rect, viewport)
            if endpoints is None:
                continue
            start, end = endpoints
            painter.setPen(QPen(QColor('#ffdc50'), 1))
            painter.drawLine(start, end)
            painter.fillRect(rect, QColor(25, 25, 25, 210))
            painter.drawText(rect, Qt.AlignCenter, name)

    def wheelEvent(self, event):
        self.zoom_by(1.25 ** (event.angleDelta().y()/120.), event.pos())
        event.accept()

    def mousePressEvent(self, event):
        if event.button() in (Qt.RightButton, Qt.MiddleButton):
            self._drag = QPointF(event.pos())
            self.setCursor(Qt.ClosedHandCursor)
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._drag is not None:
            point = QPointF(event.pos())
            self.pan += point-self._drag
            self._drag = point
            self.image_rect()
            self.update()
            event.accept()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self._drag is not None:
            self._drag = None
            self.unsetCursor()
            event.accept()
        else:
            super().mouseReleaseEvent(event)
