import cv2
import os
import numpy as np
import pcl
import open3d as o3d

class PredictionProcessor:
    def __init__(self, extrinsics, intrinsics, lowerBound=(40, 30, 30), upperBound=(80, 255, 255)):
        self.extrinsic = extrinsics
        self.intrinsic = intrinsics

        # Lower and upper bounds for HSV color range
        self.lowerBound = lowerBound
        self.upperBound = upperBound


    # Returns a sorted list of centroids for the largest connected components in the prediction (largest to smallest)
    def computeCentroids(self, image):
        # Preprocess image
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Filter green pixels in hsv image
        mask = cv2.inRange(hsv, self.lowerBound, self.upperBound)

        # Find connected components in mask
        n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)

        # Sort centroids by area (descending) using stats[:, cv2.CC_STAT_AREA]
        centroids_sorted = centroids[stats[:, cv2.CC_STAT_AREA].argsort()[::-1]]

        return centroids_sorted

    def pixel2CameraCoordinate(self, pixel, depth):
        # Convert pixel to homogeneous coordinates
        pixel_homogeneous = np.array([pixel[0], pixel[1], 1])

        # Compute world coordinates
        camera_point_homogeneous = np.linalg.inv(self.intrinsic) @ pixel_homogeneous * depth

        return camera_point_homogeneous

    def computeTransformationMatrix(self, world, n_avg):
        # Compute rotation matrix
        R = np.identity(3)
        R[2, :] = n_avg
        R[0, :] = np.cross(R[2, :], np.array([0, 0, 1]))
        R[1, :] = np.cross(R[2, :], R[0, :])

        # Compute translation vector
        t = world

        # Compute transformation matrix
        T = np.identity(4)
        T[:3, :3] = R
        T[:3, 3] = t

        return T



class DataLoader:
    def __init__(self):
        pass

    def load_image(self, path, flag=cv2.IMREAD_COLOR):
        # Check if image exists
        if not os.path.exists(path):
            raise FileNotFoundError("Image not found")
        
        return cv2.imread(path, flag)

    def load_txt(self, path, rows, cols, printData=False):
        # Check if file exists
        if not os.path.exists(path):
            raise FileNotFoundError("File not found")

        # Load file
        data = np.loadtxt(path, dtype=np.float32)
        if printData:
            print(f"type / type: {type(data)} {data.shape}")
            print(data)
        
        return data

    def create_mask(self, mask):
        # Create empty target mask
        target_mask = np.array(mask, dtype=np.uint8)

        # Create mask for each class
        target_mask[np.where((mask == [0, 0, 0]).all(axis=2))]       = [0, 0, 255]
        target_mask[np.where((mask == [128, 128, 128]).all(axis=2))] = [0, 255, 0]
        target_mask[np.where((mask == [255, 255, 255]).all(axis=2))] = [255, 0, 0]

        return target_mask

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

def create_point_cloud(depth):
    # Convert depth map to point cloud
    points = []
    for i in range(depth.shape[0]):
        for j in range(depth.shape[1]):
            z = depth[i, j]
            if z > 0:
                x = j - depth.shape[1] / 2
                y = i - depth.shape[0] / 2
                points.append([x, y, z])
    
    cloud = pcl.PointCloud_PointXYZ(np.array(points, dtype=np.float32))
    pcl.

    return cloud

def main():
    # Load image, GT mask, depth map, camera extrinsics and intrinsics
    loader = DataLoader()
    idName = "000028-0"
    image = loader.load_image(f"data/color-input/{idName}.png")
    mask = loader.load_image(f"data/label/{idName}.png")
    depth = loader.load_image(f"data/depth-input/{idName}.png", cv2.IMREAD_UNCHANGED)
    # Create artificial prediction from GT data
    prediction = loader.create_mask(mask) # Denne skal erstattes med prediction og er i BGR format
    # Load camera extrinsics and intrinsics
    printData = False
    extrinsics = loader.load_txt(f"data/camera-pose/{idName}.txt", 4, 4, printData)
    intrinsics = loader.load_txt(f"data/camera-intrinsics/{idName}.txt", 3, 3, printData)

    # Create prediction processor and define custom lower and upper bounds for HSV color range
    # These values are used to filter out green pixels in the prediction
    lowerBound = (40, 30, 30)
    upperBound = (80, 255, 255)
    predictionProcessor = PredictionProcessor(extrinsics, intrinsics, lowerBound, upperBound)

    # Find largest connected component (green object) in prediction
    centers = predictionProcessor.computeCentroids(prediction)

    # Draw circles at center of the largest connected component
    # Convert array of centers to int
    centers = centers.astype(int)
    center = centers[1]
    center = np.array((203,338))
    cv2.circle(image, center, 5, (0, 0, 255), -1)



    #----------------------------------------------------------------------

    # Apply median and gaussian filter to reduce noise
    depth = cv2.medianBlur(depth, 5)
    depth = cv2.GaussianBlur(depth, (5, 5), 0)

    # Normalize depth map
    depth_map_normalized = ((depth - np.min(depth)) / np.ptp(depth)).astype(np.float32)

    # Calcluate gradient of depth map
    zx = cv2.Sobel(depth_map_normalized, cv2.CV_32F, 1, 0, ksize=7)     
    zy = cv2.Sobel(depth_map_normalized, cv2.CV_32F, 0, 1, ksize=7)

    # Imshow the gradients
    cv2.imshow("zx", zx)
    cv2.imshow("zy", zy)

    normals = np.dstack((-zx, -zy, np.ones_like(depth_map_normalized)))
    n = np.linalg.norm(normals, axis=2, keepdims=True)
    # normals[:, :, 0] /= n
    # normals[:, :, 1] /= n
    # normals[:, :, 2] /= n
    # normals= ((normals + 1.0) / 2.0 * 255.0).astype(np.uint8) # This step is for visualization
    normals /= np.maximum(n, 1e-7)


    vis_normal = lambda normal: np.uint8((normal + 1) / 2 * 255)[..., ::-1]
    normal_v2 = get_surface_normal_by_depth(depth_map_normalized, intrinsics)

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
    start_x = (center[0], center[1])
    end_x = (int(center[0] + n_avg[0]*100), int(center[1]))
    start_y = (center[0], center[1])
    end_y = (int(center[0]), int(center[1] + n_avg[1]*100))
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

    cv2.imshow("normal_v2", vis_normal(normal_v2))

    #----------------------------------------------------------------------

    cloud = create_point_cloud(depth)

    # Visualize point cloud using Open3D
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(cloud.to_array())

    o3d.visualization.draw_geometries([pcd])


    # Calculate world coordinates of center of largest object
    # d = depth[center[0], center[1]] * 1e-4
    # camera_3dpoint = predictionProcessor.pixel2CameraCoordinate(center, d)
    # print(f"World coordinates in camera reference: \n{camera_3dpoint}")
    
    # Calculate transformation matrix using world coordinates, normal vector and camera extrinsics
    # n_avg = n_avg / np.linalg.norm(n_avg) # normalize
    # T = predictionProcessor.computeTransformationMatrix(camera_3dpoint, n_avg)
    # print(f"Transformation matrix: \n{T}")


    # Display images
    cv2.imshow("image", image)
    cv2.imshow("prediction", prediction)
    cv2.imshow("depth", depth)
    cv2.imshow("normal", normals)
    cv2.waitKey(0)





if __name__ == "__main__":
    main()
    