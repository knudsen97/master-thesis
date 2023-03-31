from PredictionProcessor import *

def main():
    calculate_transformation()
    # # Load image, GT mask, depth map, camera extrinsics and intrinsics
    # loader = DataLoader()
    # idName = "000028-0"
    # datapath = "../../data/"
    # image = loader.load_image(f"{datapath}color-input/{idName}.png")
    # mask = loader.load_image(f"{datapath}label/{idName}.png")
    # depth = loader.load_image(f"{datapath}depth-input/{idName}.png", cv.IMREAD_UNCHANGED)
    # # depth_syn = loader.load_image(f"data_synthetic/depth-input/depth0.png", cv2.IMREAD_UNCHANGED)

    # # Convert depth_syn to 1 channel
    # # depth = cv2.cvtColor(depth_syn, cv2.COLOR_BGR2GRAY)
    # # print(depth[0])
    # # print(depth_syn[0])

    # # print(f"depth: {depth.shape} {depth.dtype} {depth.min()} {depth.max()} {type(depth)}")
    # # print(f"depth_syn: {depth_syn.shape} {depth_syn.dtype} {depth_syn.min()} {depth_syn.max()} {type(depth_syn)}")

    # # Create artificial prediction from GT data
    # prediction = loader.create_mask(mask) # Denne skal erstattes med prediction og er i BGR format


    # # Load camera extrinsics and intrinsics
    # extrinsics = loader.load_txt(f"{datapath}camera-pose/{idName}.txt", 4, 4, printData=False)
    # intrinsics = loader.load_txt(f"{datapath}camera-intrinsics/{idName}.txt", 3, 3, printData=False)

    # # Create prediction processor and define custom lower and upper bounds for HSV color range
    # # These values are used to filter out green pixels in the prediction
    # lowerBound = (40, 30, 30)
    # upperBound = (80, 255, 255)
    # predictionProcessor = PredictionProcessor(extrinsics, intrinsics, lowerBound, upperBound)

    # # Find largest connected component (green object) in prediction
    # centers = predictionProcessor.computeCentroids(prediction)

    # # Draw circles at center of the largest connected component
    # # Convert array of centers to int
    # centers = centers.astype(int)
    # center = centers[1]
    # # center = np.array((203,338))
    # cv.circle(image, center, 5, (0, 0, 255), -1)



    # #----------------------------------------------------------------------
    # #-------------------------- OPEN3D ------------------------------------
    # #----------------------------------------------------------------------
    # # Convert depth map to o3d image
    # depth_image = o3d.geometry.Image(depth)

    # # Convert intrinsics to o3d camera intrinsic and create o3d camera
    # camera_intrinsics = o3d.camera.PinholeCameraIntrinsic(intrinsics.shape[1], intrinsics.shape[0], intrinsics[0,0], intrinsics[1,1], intrinsics[0,2], intrinsics[1,2])
    # camera_extrinsics = np.eye(4) # TODO: Kig eventuelt på dette og brug kamera extrinsics hvis nødvendigt?
    
    # # Create point cloud from depth map
    # depth_scale = 1000
    # pc = o3d.geometry.PointCloud.create_from_depth_image(depth_image, camera_intrinsics, camera_extrinsics, depth_scale=depth_scale)

    # # Remove outliers
    # pc.voxel_down_sample(voxel_size=0.0005)
    # pc.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.0)

    # # Flip point cloud
    # pc.transform([[1, 0, 0, 0],
    #               [0, -1, 0, 0],
    #               [0, 0, -1, 0],
    #               [0, 0, 0, 1]])
    
    # # Compute surface normals
    # pc.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30))
    # pc.normalize_normals()

    # # Pixel coordinate to 3D point
    # center_3d = predictionProcessor.pixel2point(point2d=center, depth=depth, depth_scale=depth_scale)

    # # Compute distances between points in pc and pc_center_3d & find index of closest point
    # distances = predictionProcessor.computeCloudDistances(pc, center_3d)

    # # Get index of closest point
    # index = np.argmin(distances)

    # # Get normal vector and 3d point at closest point
    # pos_3d = np.asarray(pc.points[index])
    # normal = np.asarray(pc.normals[index])

    # # Check if normal z component is negative and flip normal if so
    # if normal[2] < 0:
    #     normal = -normal

    # normalized_normal = normal / np.linalg.norm(normal)
    # print(f"Normal and normalized normal at {center} is: \n {normal} \n {normalized_normal}")

    # # Create normal vector line
    # scale = 1
    # line_points = [pos_3d, pos_3d + normal*scale]
    # line_colors = [[1, 0, 0], [1, 0, 0]]

    # # Create line set to visualize normal vector
    # line_set = o3d.geometry.LineSet()
    # line_set.points = o3d.utility.Vector3dVector(line_points)
    # line_set.lines = o3d.utility.Vector2iVector([[0, 1]])
    # line_set.colors = o3d.utility.Vector3dVector(line_colors)

    # # Compute rotation matrix from normal
    # R = predictionProcessor.computeRotationMatrixFromNormal(normal)
    # # print("Rotation matrix:\n", R)

    # # Tests to confirm that R is a valid rotation matrix
    # # print("R dot n:", R.dot(normal)) # Should be equal to normal
    # # print("R^T R:\n", np.matmul(R.T, R), "\n RR^T\n", np.matmul(R, R.T)) # Should be identity matrix
    # # print("det(R):", np.linalg.det(R))   # Should be 1
    # # v = np.array([1, 0, 0])
    # # v_rotated = R.dot(v)
    # # print("v_rotated:", v_rotated)
    # # dot_product = np.dot(v_rotated, normal)
    # # print("dot_product:", dot_product)

    # #---------------------- VISUALIZE -------------------------------------
    # # Display images
    # cv.imshow("image", image)
    # cv.imshow("prediction", prediction)
    # cv.imshow("depth", depth)
    # # cv2.imshow("normal", normals)

    # o3d.visualization.draw_geometries([pc, line_set])
    # cv.waitKey(0)

if __name__ == "__main__":
    main()