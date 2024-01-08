from shapely.geometry import LineString, box
from shapely import affinity

def iou2d(box2d_1, box2d_2):
    """
    Args:
        box2d_1, box2d_2: [xmin, ymin, xmax, ymax]
   
    Returns:
        iou2d
    """
    xmin1, ymin1, xmax1, ymax1 = box2d_1
    a1 = (xmax1 - xmin1) * (ymax1 - ymin1)
    xmin2, ymin2, xmax2, ymax2 = box2d_2
    a2 = (xmax2 - xmin2) * (ymax2 - ymin2)

    overlap_x = min(xmax1, xmax2) - max(xmin1, xmin2)
    overlap_y = min(ymax1, ymax2) - max(ymin1, ymin2)
    if overlap_x > 0 and overlap_y > 0:
        overlap_xy = overlap_x * overlap_y
        return overlap_xy / (a1 + a2 - overlap_xy)
    else:
        return 0.

def iou3d(box3d_1, box3d_2):
    """
    Args:
        box3d_1, box3d_2: [x, y, z, l, w, h, yaw]
   
    Returns:
        iou3d
    """
    x1, y1, z1, l1, w1, h1, yaw1 = box3d_1
    v1 = l1 * w1 * h1
    ls1 = LineString([[0, z1 - h1/2,],  [0, z1 + h1/2]])
    poly1 = box(x1 - l1/2, y1 - w1/2, x1 + l1/2, y1 + w1/2)
    poly_rot1 = affinity.rotate(poly1, yaw1, use_radians=True)

    x2, y2, z2, l2, w2, h2, yaw2 = box3d_2
    v2 = l2 * w2 * h2
    ls2 = LineString([[0, z2 - h2/2,],  [0, z2 + h2/2]])
    poly2 = box(x2 - l2/2, y2 - w2/2, x2 + l2/2, y2 + w2/2)
    poly_rot2 = affinity.rotate(poly2, yaw2, use_radians=True)

    overlap_xy = poly_rot1.intersection(poly_rot2).area
    overlap_z = ls1.intersection(ls2).length
    overlap_xyz = overlap_xy * overlap_z
    return overlap_xyz / (v1 + v2 - overlap_xyz)


if __name__ == "__main__":
    box2d_1 = [0,0,10,10]
    box2d_2 = [5,5,15,15]
    box2d_3 = [2,5,10,10]
    box2d_4 = [11,5,15,12]
    print(iou2d(box2d_1, box2d_2)) # 0.14285714285714285
    print(iou2d(box2d_1, box2d_3)) # 0.4
    print(iou2d(box2d_1, box2d_4)) # 0.0
    box3d_1 = [0,0,0,2,2,2,3.1415/2]
    box3d_2 = [0,-0.2,-0.2,2,2,2,3.1415/3]
    box3d_3 = [0.5,0.5,0.5,0.5,0.5,0.5,3.1415/9]
    box3d_4 = [5,5,5,2,2,2,-3.1415]
    print(iou3d(box3d_1, box3d_2)) # 0.5872843950165108
    print(iou3d(box3d_1, box3d_3)) # 0.015625000000000003
    print(iou3d(box3d_1, box3d_4)) # 0.0


    
