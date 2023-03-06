import cv2
import numpy as np
def get_surface_normal_by_depth(depth, K=None):
    """
    depth: (h, w) of float, the unit of depth is meter
    K: (3, 3) of float, the depth camere's intrinsic
    """
    K = [[1, 0], [0, 1]] if K is None else K
    fx, fy = K[0][0], K[1][1]

    # dz_dv, dz_du = np.gradient(depth)  # u, v mean the pixel coordinate in the image
    dz_dv = cv2.Sobel(depth, cv2.CV_32F, 1, 0, ksize=7)     
    dz_du = cv2.Sobel(depth, cv2.CV_32F, 0, 1, ksize=7)
    # u*depth = fx*x + cx --> du/dx = fx / depth
    du_dx = fx / depth  # x is xyz of camera coordinate
    dv_dy = fy / depth

    dz_dx = dz_du * du_dx
    dz_dy = dz_dv * dv_dy
    # cross-product (1,0,dz_dx)X(0,1,dz_dy) = (-dz_dx, -dz_dy, 1)
    normal_cross = np.dstack((-dz_dx, -dz_dy, np.ones_like(depth)))
    # normalize to unit vector
    normal_unit = normal_cross / np.linalg.norm(normal_cross, axis=2, keepdims=True)
    # set default normal to [0, 0, 1]
    normal_unit[~np.isfinite(normal_unit).all(2)] = [0, 0, 1]
    return normal_unit

if __name__ == "__main__":
    depth_map = cv2.imread("untitled.png", cv2.IMREAD_ANYDEPTH)
    normal_map = get_surface_normal_by_depth(depth_map)

    vis_normal = lambda normal: np.uint8((normal + 1) / 2 * 255)[..., ::-1]


    cv2.imshow("Normal Map", vis_normal(normal_map))
    cv2.waitKey(0)