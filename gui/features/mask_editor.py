"""Interactive editor for native-resolution sample object labels."""
import cv2
import numpy as np
from PyQt5.QtCore import Qt, QTimer, QLineF
from PyQt5.QtGui import QImage, QPixmap, QPen
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QGraphicsScene,
    QGraphicsView, QLabel, QPushButton, QDoubleSpinBox, QSpinBox, QSlider,
    QDialogButtonBox)


class MaskView(QGraphicsView):
    def __init__(self, editor):
        super().__init__(editor)
        self.editor = editor
        self.setScene(QGraphicsScene(self))
        self.image = self.scene().addPixmap(QPixmap())
        self.line = None

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if not self.image.pixmap().isNull():
            self.fitInView(self.image, Qt.KeepAspectRatio)

    def mousePressEvent(self, event):
        point = self.mapToScene(event.pos())
        if event.button() != Qt.LeftButton or not self.image.contains(point):
            return
        if self.editor.cut.isChecked():
            self.start = point
            self.line = self.scene().addLine(QLineF(point, point), QPen(Qt.yellow, 2))
        else:
            self.editor.toggle(int(point.x()), int(point.y()))

    def mouseMoveEvent(self, event):
        if self.line is not None:
            self.line.setLine(QLineF(self.start, self.mapToScene(event.pos())))

    def mouseReleaseEvent(self, event):
        if self.line is not None:
            end = self.mapToScene(event.pos())
            self.scene().removeItem(self.line)
            self.line = None
            self.editor.cuts.append(((int(self.start.x()), int(self.start.y())),
                                     (int(end.x()), int(end.y()))))
            self.editor.rebuild()


class MaskEditor(QDialog):
    def __init__(self, parent, band, threshold, erosion, previous=None, *, band_nm, otsu_mask=None):
        super().__init__(parent)
        self.setWindowTitle(f"Edit sample objects · {band_nm:g} nm")
        self.resize(900, 780)
        self.band = band
        valid = band[np.isfinite(band)]
        if not valid.size:
            raise ValueError("No finite intensities on this band.")
        lo, hi = map(float, np.percentile(valid, [1, 99.9]))
        self.base = np.nan_to_num(np.clip((band-lo)/max(hi-lo, 1e-12), 0, 1)*255).astype('uint8')
        self.cuts, self.selected_points = ([], []) if previous is None else (list(previous[0]), list(previous[1]))
        layout = QVBoxLayout(self)
        row = QHBoxLayout()
        layout.addLayout(row)
        self.threshold = QDoubleSpinBox()
        self.threshold.setDecimals(12)
        self.threshold.setRange(min(float(valid.min()), threshold), max(float(valid.max()), threshold))
        self.threshold.setValue(threshold)
        self.initial_threshold = self.threshold.value()
        self.otsu_mask = otsu_mask
        self.threshold_label = QLabel()
        row.addWidget(self.threshold_label)
        row.addWidget(self.threshold)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 1000)
        self.low, self.high = min(lo,threshold), max(hi,threshold)
        row.addWidget(self.slider, 1)
        self.area = QSpinBox()
        self.area.setRange(1, band.size)
        self.area.setValue(10 if previous is None else previous[2])
        row.addWidget(QLabel("Min area (px)"))
        row.addWidget(self.area)
        self.erosion = QSpinBox()
        self.erosion.setRange(0,100)
        self.erosion.setValue(erosion)
        row.addWidget(QLabel("Erosion (px)"))
        row.addWidget(self.erosion)
        self.view = MaskView(self)
        layout.addWidget(self.view, 1)
        layout.addWidget(QLabel("Click the objects to keep. Click again to deselect. Use Draw separation to split an object."))
        row = QHBoxLayout()
        layout.addLayout(row)
        self.cut = QPushButton("Draw separation")
        self.cut.setCheckable(True)
        row.addWidget(self.cut)
        undo = QPushButton("Undo last cut")
        undo.clicked.connect(self.undo)
        row.addWidget(undo)
        self.status = QLabel()
        row.addWidget(self.status, 1)
        buttons = QDialogButtonBox(QDialogButtonBox.Apply | QDialogButtonBox.Cancel)
        self.apply = buttons.button(QDialogButtonBox.Apply)
        self.apply.clicked.connect(self.finish)
        buttons.rejected.connect(self.reject)
        row.addWidget(buttons)
        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.setInterval(65)
        self.timer.timeout.connect(self.rebuild)
        self.threshold.valueChanged.connect(self.sync)
        self.slider.valueChanged.connect(lambda v: self.threshold.setValue(self.low+(self.high-self.low)*v/1000))
        for control in (self.area,self.erosion):
            control.valueChanged.connect(lambda _: self.timer.start())
        self.sync()
        self.rebuild()

    def sync(self):
        self.threshold_label.setText("Threshold (Otsu)" if self.uses_otsu() else "Threshold")
        self.slider.blockSignals(True)
        self.slider.setValue(round(1000*(self.threshold.value()-self.low)/max(self.high-self.low,1e-12)))
        self.slider.blockSignals(False)
        self.timer.start()

    def uses_otsu(self):
        return self.otsu_mask is not None and self.threshold.value() == self.initial_threshold

    def undo(self):
        if self.cuts:
            self.cuts.pop()
            self.rebuild()

    def rebuild(self):
        self.timer.stop()
        if self.uses_otsu():
            mask = self.otsu_mask.astype(np.uint8)
        else:
            mask = (np.isfinite(self.band) & (self.band > self.threshold.value())).astype(np.uint8)
        for start,end in self.cuts:
            cv2.line(mask,start,end,0,2)
        radius = self.erosion.value()
        if radius:
            y,x = np.ogrid[-radius:radius+1,-radius:radius+1]
            mask = cv2.erode(mask,(x*x+y*y<=radius*radius).astype('uint8'),borderType=cv2.BORDER_CONSTANT,borderValue=0)
        _, ids, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        keep = stats[:,cv2.CC_STAT_AREA] >= self.area.value()
        keep[0] = False
        lookup = np.where(keep,np.cumsum(keep),0).astype('int32')
        self.ids = lookup[ids]
        self.objects = stats[keep]
        self.paint_objects()

    def toggle(self,x,y):
        if self.timer.isActive():
            self.rebuild()
        label = self.ids[y,x]
        if label:
            matching = [p for p in self.selected_points if self.ids[p[1],p[0]] == label]
            if matching:
                self.selected_points = [p for p in self.selected_points if p not in matching]
            else:
                self.selected_points.append((x,y))
            self.paint_objects()

    def paint_objects(self):
        selected = {self.ids[y, x] for x, y in self.selected_points} - {0}
        self.labels = np.where(np.isin(self.ids, list(selected)), self.ids, 0).astype(np.int32)
        rgb = cv2.cvtColor(self.base, cv2.COLOR_GRAY2RGB)
        for i,(x,y,w,h,_) in enumerate(self.objects,1):
            color = (0, 240, 240) if i in selected else (255, 120, 80)
            roi = rgb[y:y+h,x:x+w]
            contours,_ = cv2.findContours((self.ids[y:y+h,x:x+w]==i).astype('uint8'),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(roi,contours,-1,color,2)
            cv2.putText(rgb,str(i),(x,max(12,y+12)),cv2.FONT_HERSHEY_SIMPLEX,.45,color,1)
        h,w = self.band.shape
        pix = QPixmap.fromImage(QImage(rgb.data,w,h,rgb.strides[0],QImage.Format_RGB888))
        self.view.image.setPixmap(pix)
        self.view.scene().setSceneRect(self.view.image.boundingRect())
        self.view.fitInView(self.view.image, Qt.KeepAspectRatio)
        count = len(selected)
        self.apply.setEnabled(count > 0)
        self.status.setText(f"{count}/{len(self.objects)} selected")

    def finish(self):
        self.rebuild()
        if self.labels.any():
            self.accept()
