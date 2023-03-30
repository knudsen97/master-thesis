import cv2
import os
import numpy as np
import matplotlib.pyplot as plt



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

    depth[np.where(depth==0)] = 1
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

def get_normal_v1(depth):

    # Calcluate gradient of depth map
    zx = cv2.Sobel(depth, cv2.CV_32F, 1, 0, ksize=7)     
    zy = cv2.Sobel(depth, cv2.CV_32F, 0, 1, ksize=7)

    # zx_min = np.min(zx)
    # zx_median = np.median(zx)
    # zx_max = np.max(zx)
    # print(f"zx min:{zx_min}, max:{zx_max}, median:{zx_median}")


    # Imshow the gradients
    cv2.imshow("zx", zx)
    cv2.imshow("zy", zy)
    cv2.waitKey(0)

    normals = np.dstack((-zx, -zy, np.ones_like(depth)))
    n = np.linalg.norm(normals, axis=2, keepdims=True)
    # normals[:, :, 0] /= n
    # normals[:, :, 1] /= n
    # normals[:, :, 2] /= n
    # normals= ((normals + 1.0) / 2.0 * 255.0).astype(np.uint8) # This step is for visualization
    normals /= np.maximum(n, 1e-7)
    return normals

# https://stackoverflow.com/questions/53350391/surface-normal-calculation-from-depth-map-in-python
# NOTE: suuuuuuuuper slow 
def get_normal_bedrick_kiq(depth, image):
    normals = np.array(image, dtype="float32")
    h,w,d = image.shape
    for i in range(1,w-1):
        for j in range(1,h-1):
            t = np.array([i,j-1,depth[j-1,i]],dtype="float64")
            f = np.array([i-1,j,depth[j,i-1]],dtype="float64")
            c = np.array([i,j,depth[j,i]] , dtype = "float64")
            d = np.cross(f-c,t-c)
            n = d / np.sqrt((np.sum(d**2)))
            normals[j,i] = n
    return normals

# https://answers.opencv.org/question/150490/depth-map-to-normal-map-conversion/
def get_normal_lucidus(_depth):
    depth = _depth.copy()
    depth = depth.astype("float64")
    h, w = np.shape(depth)
    normals = np.zeros((h, w, 3))

    def normalizeVector(v):
        length = np.sqrt(v[0]**2 + v[1]**2 + v[2]**2)
        v = v/length
        return v

    for x in range(1, h-1):
        for y in range(1, w-1):
            
            dzdx = depth[x+1, y] - depth[x-1, y]
            dzdy = depth[x, y+1] - depth[x, y-1]

            d = (-dzdx, -dzdy, 1.0)

            n = normalizeVector(d)

            normals[x,y] = n * 0.5 + 0.5

    normals *= 255

    normals = normals.astype('uint8')
    return normals

def main():

    filename = "tilt.png"
    image = cv2.imread(filename)
    depth = cv2.imread(filename, cv2.IMREAD_ANYDEPTH)

    point = {
        "mid container" : (320,200),
        "box down left" : (425,354),
        "box above the down left one": (430, 287)
        }

    center = np.array(point["box above the down left one"])
    cv2.circle(image, center, 5, (0, 0, 255), -1)

    # Apply median and gaussian filter to reduce noise
    depth = cv2.medianBlur(depth, 5)
    depth = cv2.GaussianBlur(depth, (5, 5), 0)


    # Normalize depth map
    depth_map_normalized = ((depth - np.min(depth)) / np.ptp(depth)).astype(np.float32)


    vis_normal = lambda normal: np.uint8((normal + 1) / 2 * 255)[..., ::-1]
    normal_v2 = get_surface_normal_by_depth(depth_map_normalized)
    # normals = get_normal_v1(depth_map_normalized)
    normals = get_normal_lucidus(depth)

    # Calculate an average normal vector for the largest object's center
    radius = 3
    n_avg = np.zeros(3)
    count = 0
    for i in range(center[0]-radius, center[0]+radius):
        for j in range(center[1]-radius, center[1]+radius):
            n = normals[i, j]
            start = np.array((i, j))
            end = np.array((int(i + n[0]*50), int(j + n[1]*50)))

            # cv2.arrowedLine(image, start, end, (0, 255, 0), 1)

            n_avg += n
            count += 1
        # print("")

    n_avg /= count
    # n_avg = n_avg.astype(int)

    # Draw arrowed line of average normal in red
    scale = 0.5
    start_x = (center[0], center[1])
    end_x = (int(center[0] + n_avg[0]*scale), int(center[1]))
    start_y = (center[0], center[1])
    end_y = (int(center[0]), int(center[1] + n_avg[1]*scale))
    cv2.arrowedLine(image, start_x, end_x, (0, 0, 255), 2)
    cv2.arrowedLine(image, start_y, end_y, (0, 255, 0), 2)

    print("average normal: ", n_avg)

    # p = PredictionProcessor.pixel2CameraCoordinate(center, depth[center[0], center[1]] * 1e-4)
    # Get neighboring normals
    n1 = normal_v2[center[0]-1, center[1]]
    n2 = normal_v2[center[0], center[1]-1]

    # Calculate surface tangent vectors
    t1 = np.cross(n1, np.array([0, 0, 1]))
    t2 = np.cross(n2, np.array([0, 0, 1]))

    surface_normal = np.cross(t1, t2)
    surface_normal /= np.linalg.norm(surface_normal)

    # # Check if normal is perpendicular to surface
    normal = normals[center[0], center[1]]
    if np.dot(normal, surface_normal) > 0:
        print("The normal is perpendicular to the surface")
    else:
        print("The normal is not perpendicular to the surface")



    #----------------------------------------------------------------------

    # Calculate world coordinates of center of largest object
    # d = depth[center[0], center[1]] * 1e-4
    # camera_3dpoint = predictionProcessor.pixel2CameraCoordinate(center, d)
    # print(f"World coordinates in camera reference: \n{camera_3dpoint}")
    
    # Calculate transformation matrix using world coordinates, normal vector and camera extrinsics
    # n_avg = n_avg / np.linalg.norm(n_avg) # normalize
    # T = predictionProcessor.computeTransformationMatrix(camera_3dpoint, n_avg)
    # print(f"Transformation matrix: \n{T}")

    # # Display images
    # cv2.imshow("normal_v2", vis_normal(normal_v2))
    # cv2.imshow("image", image)
    # cv2.imshow("depth", depth)
    # cv2.imshow("normal", normals)
    # cv2.waitKey(0)

    # Display images using matplotlib
    f, axarr = plt.subplots(2,2)
    axarr[0,0].set_title('normal_v2')
    axarr[0,0].imshow(normal_v2)
    
    axarr[0,1].set_title('image')
    axarr[0,1].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    axarr[1,0].set_title('depth map')
    axarr[1,0].imshow(depth, cmap="gray")

    axarr[1,1].set_title('normal')
    axarr[1,1].imshow(cv2.cvtColor(normals, cv2.COLOR_BGR2RGB))

    plt.subplots_adjust(left=0,
                        bottom=0.1,
                        right=1,
                        top=0.9,
                        wspace=-0.5,
                        hspace=0.4)
    plt.show()


if __name__ == "__main__":
    main()