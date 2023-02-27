import cv2
import os
import numpy as np

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



def get_surface_normal_by_depth(depth, K=None, method='gradient'):
    """
    depth: (h, w) of float, the unit of depth is meter
    K: (3, 3) of float, the depth camere's intrinsic
    """
    K = [[1, 0], [0, 1]] if K is None else K
    fx, fy = K[0][0], K[1][1]

    depth[depth == 0] = 1e-7
    normal_unit = np.zeros((depth.shape[0], depth.shape[1], 3))

    if method == 'gradient':
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
    elif method == 'conv':
        # Create conv kernels
        kernel_x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
        kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

        kernel_x_flip = np.flip(kernel_x, axis=0)
        kernel_y_flip = np.flip(kernel_y, axis=1)

        # Apply conv kernels
        dz_dx = cv2.filter2D(depth, -1, kernel_x_flip, borderType=cv2.BORDER_CONSTANT)
        dz_dy = cv2.filter2D(depth, -1, kernel_y_flip, borderType=cv2.BORDER_CONSTANT)

        # Compute normal
        normal_cross = np.dstack((-dz_dx, -dz_dy, np.ones_like(depth)))
        normal_unit = normal_cross / np.linalg.norm(normal_cross, axis=2, keepdims=True)
        normal_unit[~np.isfinite(normal_unit).all(2)] = [0, 0, 1]


    return normal_unit

def get_average_normal(normals, center, radius, image=None, draw=False):
    n_avg = np.zeros(3)
    count = 0
    for i in range(center[0]-radius, center[0]+radius):
        for j in range(center[1]-radius, center[1]+radius):
            n = normals[i, j]
            start = np.array((i, j))
            end = np.array((int(i + n[0]*50), int(j + n[1]*50)))

            if draw:
                cv2.arrowedLine(image, start, end, (0, 255, 0), 1)

            n_avg += n
            count += 1

    n_avg /= count
    return n_avg

def circular_gaussian_mask(radius, sigma):
    x = np.arange(-radius, radius+1)
    y = np.arange(-radius, radius+1)
    xx, yy = np.meshgrid(x, y)
    mask = np.exp(-0.5*(xx**2 + yy**2) / sigma**2)
    return mask

def get_weighted_average_normal(normals, center, radius, sigma=0.1, image=None, draw=False, weight_func=None):
    n_avg = np.zeros(3)
    weights_sum = 0
    scale = 50

    # create circular mask with specified radius and sigma
    circular_mask = circular_gaussian_mask(radius, sigma)
    circular_mask = cv2.resize(circular_mask, (normals.shape[1], normals.shape[0]))
    print("center in func:", center)
    # iterate over pixels in circular region
    for i in range(center[0]-radius, center[0]+radius+1):
        for j in range(center[1]-radius, center[1]+radius+1):
            # check if pixel is within image bounds
            if i < 0 or i >= normals.shape[0] or j < 0 or j >= normals.shape[1]:
                continue

            # calculate weight for pixel using weight_func
            weight = weight_func(circular_mask[i,j]) if weight_func is not None else circular_mask[i,j]

            # calculate weighted normal and accumulate
            n = normals[i, j]
            n_avg += n * weight
            weights_sum += weight

            # draw arrow on image if requested
            if draw and image is not None:
                start = np.array((i, j))
                end = np.array((int(i + n[0]*scale), int(j + n[1]*scale)))
                cv2.arrowedLine(image, tuple(start), tuple(end), (0, 255, 0), 1)

    # normalize accumulated normal by sum of weights
    if weights_sum != 0:
        n_avg /= weights_sum

    if draw:
        start = np.array((center[0], center[1]))
        end = np.array((int(center[0] + n_avg[0]*scale), int(center[1] + n_avg[1]*scale)))
        cv2.arrowedLine(image, tuple(start), tuple(end), (255, 0, 0), 2)

    return n_avg

def get_weighted_average_normal_v2(normals, center, radius, sigma=0.1, image=None, draw=False, weight_func=None, angle_threshold=np.pi/2):
    n_avg = np.zeros(3)
    weights_sum = 0
    scale = 50

    # create circular mask with specified radius and sigma
    circular_mask = circular_gaussian_mask(radius, sigma)
    circular_mask = cv2.resize(circular_mask, (normals.shape[1], normals.shape[0]))

    # compute average normal for circular region
    for i in range(center[0]-radius, center[0]+radius+1):
        for j in range(center[1]-radius, center[1]+radius+1):
            # check if pixel is within image bounds
            if i < 0 or i >= normals.shape[0] or j < 0 or j >= normals.shape[1]:
                continue

            # calculate weight for pixel using weight_func
            weight = weight_func(circular_mask[i,j]) if weight_func is not None else circular_mask[i,j]

            # accumulate normal and weight
            n = normals[i, j]
            n_avg += n * weight
            weights_sum += weight

            # draw arrow on image if requested
            if draw and image is not None:
                start = np.array((i, j))
                end = np.array((int(i + n[0]*scale), int(j + n[1]*scale) ))
                cv2.arrowedLine(image, tuple(start), tuple(end), (0, 255, 0), 1)

    # normalize accumulated normal by sum of weights
    if weights_sum != 0:
        n_avg /= weights_sum

    # filter out noisy normals by comparing to average normal
    filtered_normals = []
    for i in range(center[0]-radius, center[0]+radius+1):
        for j in range(center[1]-radius, center[1]+radius+1):
            # check if pixel is within image bounds
            if i < 0 or i >= normals.shape[0] or j < 0 or j >= normals.shape[1]:
                continue

            # calculate angle between normal and average normal
            n = normals[i, j]
            angle = np.arccos(np.dot(n, n_avg) / (np.linalg.norm(n) * np.linalg.norm(n_avg)))

            # only include normal if angle is within threshold
            if angle < angle_threshold:
                filtered_normals.append(n)

    # compute new average normal for filtered normals
    n_filtered_avg = np.mean(filtered_normals, axis=0)

    if draw:
        start = np.array( (center[0], center[1]) )
        end = np.array(( int(center[0] + n_filtered_avg[0]*scale), int(center[1] + n_filtered_avg[1]*scale) ))
        cv2.arrowedLine(image, tuple(start), tuple(end), (255, 0, 0), 2)

    return n_filtered_avg

# def get_surface_normal(depth_map, center, K, image=None, draw=False):

#     u, v = np.meshgrid(np.arange(depth_map.shape[1]), np.arange(depth_map.shape[0]))

#     # compute x, y, z coordinates of each pixel
#     x = (u - K[0,2]) * depth_map / K[0,0]
#     y = (v - K[1,2]) * depth_map / K[1,1]
#     z = depth_map

#     # compute surface normal for each pixel
#     normals = np.zeros((depth_map.shape[0], depth_map.shape[1], 3))
#     normals[:,:,0] = (x - np.roll(x, 1, axis=1))
#     normals[:,:,1] = (y - np.roll(y, 1, axis=0))
#     normals[:,:,2] = (z - np.roll(z, 1, axis=0)) + (z - np.roll(z, 1, axis=1))

#     # normalize surface normals
#     normals = normals / np.linalg.norm(normals, axis=2, keepdims=True)


#     return normals

def convert_depth_to_point_cloud(depth_map, K):
    u, v = np.meshgrid(np.arange(depth_map.shape[1]), np.arange(depth_map.shape[0]))

    # Convert K to be in meters instead of millimeters
    K = K / 1000

    # compute x, y, z coordinates of each pixel
    x = (u - K[0,2]) * depth_map / K[0,0]
    y = (v - K[1,2]) * depth_map / K[1,1]
    z = depth_map

    # convert to point cloud
    point_cloud = np.zeros((depth_map.shape[0], depth_map.shape[1], 3))
    point_cloud[:,:,0] = x
    point_cloud[:,:,1] = y
    point_cloud[:,:,2] = z

    # print(f"point_cloud: {point_cloud.shape}")
    # print(point_cloud[0,0,:])

    return point_cloud

def PCA(data, correlation = False, sort = True):
    mean = np.mean(data, axis=0)

    data_adjust = data - mean

    #: the data is transposed due to np.cov/corrcoef syntax
    if correlation:
        matrix = np.corrcoef(data_adjust.T)
        
    else:
        matrix = np.cov(data_adjust.T) 

    eigenvalues, eigenvectors = np.linalg.eig(matrix)

    if sort:
        #: sort eigenvalues and eigenvectors
        sort = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[sort]
        eigenvectors = eigenvectors[:,sort]

    return eigenvalues, eigenvectors

def PCA_RANSAC(data, n_iterations=100, sample_size=3, inlier_threshold=0.1, correlation=False, sort=True):
    best_normal = None
    best_inliers = None

    for i in range(n_iterations):
        # Randomly sample a subset of the data
        sample_indices = np.random.choice(data.shape[0], sample_size, replace=False)
        sample = data[sample_indices]

        # Fit a plane using PCA
        eigenvalues, eigenvectors = PCA(sample, correlation=correlation, sort=sort)
        normal = eigenvectors[:,2]

        # Compute the distance from each point to the plane
        distances = np.abs(data.dot(normal) - data.dot(normal).mean())

        # Count the number of inliers within the threshold
        inliers = distances < inlier_threshold
        n_inliers = np.count_nonzero(inliers)

        # Update the best model if the current model has more inliers
        if best_inliers is None or n_inliers > best_inliers:
            best_inliers = n_inliers
            best_normal = normal

    return best_normal

# def get_surface_normals(point_cloud, center, image=None, draw=False):
#     # Use PCA to compute surface normal
#     # https://math.stackexchange.com/questions/99299/best-fitting-plane-given-a-set-of-points

#     # Extract point cloud around center
#     radius = 5
#     point_cloud_sample = point_cloud[center[0]-radius:center[0]+radius+1, center[1]-radius:center[1]+radius+1, :]
#     print(f"point_cloud_sample: {point_cloud_sample.shape}")

#     # Use PCA to compute surface normal
#     eigenvalues, eigenvectors = PCA(point_cloud_sample.reshape(-1,3), correlation=False, sort=True)
#     normal = eigenvectors[:,2]
#     print(f"normal: {normal.shape}\n {normal}")

#     # Draw surface normal
#     if draw:
#         scale = 100000
#         start = center
#         end = np.array(( int(center[0] + normal[0]*scale), int(center[1] + normal[1]*scale) ))
#         cv2.arrowedLine(image, tuple(start), tuple(end), (255, 0, 0), 2)


#     return normal

def get_surface_normals(point_cloud, center, image=None, draw=False):
    # Extract point cloud around center
    radius = 5
    point_cloud_sample = point_cloud[center[0]-radius:center[0]+radius+1, center[1]-radius:center[1]+radius+1, :]
    print(f"point_cloud_sample: {point_cloud_sample.shape}")

    # Use RANSAC to compute surface normal
    normal = PCA_RANSAC(point_cloud_sample.reshape(-1,3), correlation=True, sort=True)

    # Draw surface normal
    if draw:
        scale = 100
        start = center
        end = np.array(( int(center[0] + normal[0]*scale), int(center[1] + normal[1]*scale) ))
        cv2.arrowedLine(image, tuple(start), tuple(end), (255, 0, 0), 2)

    return normal

    

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
    # center = np.array((203,338))
    cv2.circle(image, center, 5, (0, 0, 255), -1)



    #----------------------------------------------------------------------

    depth = (depth*1e-4).astype(np.float32)
    # Remove noise from depth map
    depth = cv2.medianBlur(depth, 3)
    depth = cv2.GaussianBlur(depth, (3, 3), 3)

    # Normalize depth map
    # depth_map_normalized = ((depth - np.min(depth)) / np.ptp(depth)).astype(np.float32)

    # Calculate surface normals
    normals = get_surface_normal_by_depth(depth, intrinsics, method="gradient")
    # point_cloud = convert_depth_to_point_cloud(depth, intrinsics)
    # tests = get_surface_normals(point_cloud, center, image=image, draw=True)

    # Calculate an average normal vector for the largest object's center
    radius = 3
    sigma = 2
    n_avg = get_weighted_average_normal_v2(normals, center, radius, sigma, image=image, draw=True)#, weight_func=weight_func)
    print("average normal: ", n_avg)

    #----------------------------------------------------------------------


    # Display images
    cv2.imshow("image", image)
    cv2.imshow("prediction", prediction)
    cv2.imshow("depth", depth)
    cv2.imshow("normal", normals)

    cv2.waitKey(0)





if __name__ == "__main__":
    main()
    