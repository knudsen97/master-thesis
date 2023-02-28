import cv2
import os
import numpy as np
import open3d as o3d

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

    def pixel2point(self, point2d, depth, depth_scale=1000):
        # Convert pixel coordinate to 3D point
        z = depth[point2d[1], point2d[0]]/depth_scale
        x = (point2d[0] - self.intrinsic[0,2]) * z / self.intrinsic[0,0]
        y = (point2d[1] - self.intrinsic[1,2]) * z / self.intrinsic[1,1]
        point_3d = np.array([ [x, y, z] ], dtype=np.float64)

        return point_3d
    
    def computeCloudDistances(self, pc, center_3d):
        # Create point cloud from 3d point
        pc_center_3d = o3d.geometry.PointCloud()
        pc_center_3d.points = o3d.utility.Vector3dVector(center_3d)
        pc_center_3d.transform([[1, 0, 0, 0],
                                [0, -1, 0, 0],
                                [0, 0, -1, 0],
                                [0, 0, 0, 1]])

        # Compute distances between points in pc and pc_center_3d & find index of closest point
        distances = pc.compute_point_cloud_distance(pc_center_3d)
        return distances
    
    def computeRotationMatrixFromNormal(self, normal):
        # Compute rotation matrix from normal
        # Find 2 vectors orthogonal to normal
        if normal[0] != 0 or normal[1] != 0:
            u = np.array([1, 0, 0])
        else:
            u = np.array([0, 1, 0])
        v = np.cross(normal, u)
        u = np.cross(v, normal)

        # Normalize vectors
        u = u / np.linalg.norm(u)
        v = v / np.linalg.norm(v)
        normal = normal / np.linalg.norm(normal)

        # Create rotation matrix
        R = np.vstack((u, v, normal)).T
        return R

 


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
    extrinsics = loader.load_txt(f"data/camera-pose/{idName}.txt", 4, 4, printData=False)
    intrinsics = loader.load_txt(f"data/camera-intrinsics/{idName}.txt", 3, 3, printData=False)

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
    # center = np.array((203,338))
    cv2.circle(image, center, 5, (0, 0, 255), -1)



    #----------------------------------------------------------------------
    #-------------------------- OPEN3D ------------------------------------
    #----------------------------------------------------------------------
    # Convert depth map to o3d image
    depth_image = o3d.geometry.Image(depth)

    # Convert intrinsics to o3d camera intrinsic and create o3d camera
    camera_intrinsics = o3d.camera.PinholeCameraIntrinsic(intrinsics.shape[1], intrinsics.shape[0], intrinsics[0,0], intrinsics[1,1], intrinsics[0,2], intrinsics[1,2])
    camera_extrinsics = np.eye(4) # TODO: Kig eventuelt på dette og brug kamera extrinsics hvis nødvendigt?
    
    # Create point cloud from depth map
    depth_scale = 1000
    pc = o3d.geometry.PointCloud.create_from_depth_image(depth_image, camera_intrinsics, camera_extrinsics, depth_scale=depth_scale)

    # Remove outliers
    pc.voxel_down_sample(voxel_size=0.0005)
    pc.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.0)

    # Flip point cloud
    pc.transform([[1, 0, 0, 0],
                  [0, -1, 0, 0],
                  [0, 0, -1, 0],
                  [0, 0, 0, 1]])
    
    # Compute surface normals
    pc.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30))
    pc.normalize_normals()

    # Pixel coordinate to 3D point
    center_3d = predictionProcessor.pixel2point(point2d=center, depth=depth, depth_scale=depth_scale)

    # Compute distances between points in pc and pc_center_3d & find index of closest point
    distances = predictionProcessor.computeCloudDistances(pc, center_3d)

    # Get index of closest point
    index = np.argmin(distances)

    # Get normal vector and 3d point at closest point
    pos_3d = np.asarray(pc.points[index])
    normal = np.asarray(pc.normals[index])

    # Check if normal z component is negative and flip normal if so
    if normal[2] < 0:
        normal = -normal

    normalized_normal = normal / np.linalg.norm(normal)
    print(f"Normal and normalized normal at {center} is: \n {normal} \n {normalized_normal}")

    # Create normal vector line
    scale = 1
    line_points = [pos_3d, pos_3d + normal*scale]
    line_colors = [[1, 0, 0], [1, 0, 0]]

    # Create line set to visualize normal vector
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(line_points)
    line_set.lines = o3d.utility.Vector2iVector([[0, 1]])
    line_set.colors = o3d.utility.Vector3dVector(line_colors)

    # Compute rotation matrix from normal
    R = predictionProcessor.computeRotationMatrixFromNormal(normal)
    # print("Rotation matrix:\n", R)

    # Tests to confirm that R is a valid rotation matrix
    # print("R dot n:", R.dot(normal)) # Should be equal to normal
    # print("R^T R:\n", np.matmul(R.T, R), "\n RR^T\n", np.matmul(R, R.T)) # Should be identity matrix
    # print("det(R):", np.linalg.det(R))   # Should be 1
    # v = np.array([1, 0, 0])
    # v_rotated = R.dot(v)
    # print("v_rotated:", v_rotated)
    # dot_product = np.dot(v_rotated, normal)
    # print("dot_product:", dot_product)

    #---------------------- VISUALIZE -------------------------------------
    # Display images
    cv2.imshow("image", image)
    cv2.imshow("prediction", prediction)
    cv2.imshow("depth", depth)
    # cv2.imshow("normal", normals)

    o3d.visualization.draw_geometries([pc, line_set])
    cv2.waitKey(0)



if __name__ == "__main__":
    main()
    