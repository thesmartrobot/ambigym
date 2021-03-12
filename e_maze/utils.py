import numpy as np


def orientation(p1, p2, p3):
    """
    The orientation of the points is:
    - clockwise if > 0
    - aligned if == 0
    - counterclockwise if < 0
    """
    val = ((p2[1] - p1[1]) * (p3[0] - p2[0])) - (
        (p2[0] - p1[0]) * (p3[1] - p2[1])
    )
    return val


def line_segment_line_segment_intersection(pnt1, pnt2, pnt3, pnt4):
    """
    Point of intersection between two line segments.
    """
    denom = np.array(
        [
            [pnt1[0] - pnt2[0], pnt3[0] - pnt4[0]],
            [pnt1[1] - pnt2[1], pnt3[1] - pnt4[1]],
        ]
    )
    div = np.linalg.det(denom)
    if div == 0:
        # line segments are parallel
        return None

    num = np.array(
        [
            [pnt1[0] - pnt3[0], pnt3[0] - pnt4[0]],
            [pnt1[1] - pnt3[1], pnt3[1] - pnt4[1]],
        ]
    )
    mul = np.linalg.det(num)
    t = mul / div
    if t < 0.0 or t > 1.0:
        # line 1 segment does not intersect line 2
        return None

    num = np.array(
        [
            [pnt1[0] - pnt2[0], pnt1[0] - pnt3[0]],
            [pnt1[1] - pnt2[1], pnt1[1] - pnt3[1]],
        ]
    )
    mul = np.linalg.det(num)
    u = -mul / div
    if u < 0.0 or u > 1.0:
        # line 1 does not intersect line 2 segment
        return None

    return pnt1 + t * (pnt2 - pnt1)


def line_segment_circle_intersection(pnt1, pnt2, radius):
    """
    Points of intersection between a line segment and a circle with
    center (0,0).
    """
    diff = pnt2 - pnt1
    dr = np.linalg.norm(diff)
    det = pnt1[0] * pnt2[1] - pnt2[0] * pnt1[1]

    dscr = radius ** 2 * dr ** 2 - det ** 2

    if dscr < 0:
        return None, None

    def sgn(a):
        if a < 0:
            return -1
        else:
            return 1

    x1 = (det * diff[1] - sgn(diff[1]) * diff[0] * np.sqrt(dscr)) / dr ** 2
    y1 = (-det * diff[0] - abs(diff[1]) * np.sqrt(dscr)) / dr ** 2
    x2 = (det * diff[1] + sgn(diff[1]) * diff[0] * np.sqrt(dscr)) / dr ** 2
    y2 = (-det * diff[0] + abs(diff[1]) * np.sqrt(dscr)) / dr ** 2

    if dscr == 0:
        return np.array((x1, y1)), None
    else:
        return np.array((x1, y1)), np.array((x2, y2))


def distance_point_to_line_segment(pnt, spnt1, spnt2):
    segment = spnt2 - spnt1
    segment_length = np.linalg.norm(segment)
    segment_unit = segment / segment_length

    point = spnt2 - pnt
    point_scaled = point / segment_length

    t = np.dot(segment_unit, point_scaled)
    if t < 0.0:
        t = 0.0
    elif t > 1.0:
        t = 1.0
    nearest = segment * t
    distance = np.linalg.norm(nearest - point)
    nearest = nearest + spnt2
    return distance


def fuzzy_equal(a, b, threshold=1e-6):
    return abs(a - b) < threshold