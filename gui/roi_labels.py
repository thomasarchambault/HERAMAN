"""Place ROI callouts outside region bounds, in display coordinates."""
from PyQt5.QtCore import QRectF, QPointF


def nearest_outline_point(points, point):
    """Connect to the actual polygon edge, rather than its bounding box."""
    best, distance = None, float('inf')
    for a, b in zip(points, [*points[1:], points[0]]):
        dx, dy = b[0]-a[0], b[1]-a[1]
        length = dx*dx+dy*dy
        t = max(0., min(1., ((point.x()-a[0])*dx+(point.y()-a[1])*dy)/length)) if length else 0.
        candidate = QPointF(a[0]+t*dx, a[1]+t*dy)
        squared = (candidate.x()-point.x())**2+(candidate.y()-point.y())**2
        if squared < distance:
            best, distance = candidate, squared
    return best


def place_roi_labels(bounds, sizes, viewport):
    """Return label rectangles and leader endpoints; avoid regions and labels."""
    occupied = []
    result = []
    for target, (width, height) in zip(bounds, sizes):
        if not viewport.intersects(target):
            result.append(None)
            continue
        candidates = []
        for ring in range(1, 16):
            gap = 8 + (ring-1)*(height+4)
            for x, y in ((target.right()+gap, target.top()-height-gap),
                         (target.right()+gap, target.center().y()-height/2),
                         (target.center().x()-width/2, target.top()-height-gap),
                         (target.left()-width-gap, target.center().y()-height/2),
                         (target.center().x()-width/2, target.bottom()+gap),
                         (target.left()-width-gap, target.top()-height-gap),
                         (target.right()+gap, target.bottom()+gap),
                         (target.left()-width-gap, target.bottom()+gap)):
                rect = QRectF(x, y, width, height)
                if not viewport.contains(rect):
                    continue
                overlap = 0.
                for obstacle in [*bounds, *occupied]:
                    intersection = rect.intersected(obstacle.adjusted(-3, -3, 3, 3))
                    overlap += max(0., intersection.width())*max(0., intersection.height())
                candidates.append((overlap, ring, rect))
            if candidates and min(c[0] for c in candidates) == 0:
                break
        if not candidates:
            result.append(None)
            continue
        rect = min(candidates, key=lambda c: (c[0], c[1]))[2]
        occupied.append(rect)
        center = rect.center()
        start = QPointF(max(target.left(), min(target.right(), center.x())),
                        max(target.top(), min(target.bottom(), center.y())))
        end = QPointF(max(rect.left(), min(rect.right(), start.x())),
                      max(rect.top(), min(rect.bottom(), start.y())))
        result.append((rect, start, end))
    return result
