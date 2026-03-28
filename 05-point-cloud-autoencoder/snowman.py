import numpy as np
import torch
from tqdm import tqdm

def make_ellipsoid_snowman(n_points=2048, snowman_type=None, noise=0.01, min_center_x=-1, max_center_x=1, min_center_y=-1, max_center_y=1, min_center_z=0, max_center_z=3):
    # Create a snowman
    # Type 1 (arms): two ellipsoids + two cuboid on lower ellipsoid for arms
    # Type 2 (nose): two ellipsoids + one cuboid on upper ellipsoid for nose
    # Type 3 (triple): three ellipsoids
    def ellipsoid(center, axes, n_points):
        phi = np.random.uniform(0, 2 * np.pi, n_points)
        costheta = np.random.uniform(-1, 1, n_points)
        u = np.random.uniform(0, 1, n_points)

        theta = np.arccos(costheta)
        r = u ** (1/3)

        x = center[0] + axes[0] * r * np.sin(theta) * np.cos(phi)
        y = center[1] + axes[1] * r * np.sin(theta) * np.sin(phi)
        z = center[2] + axes[2] * r * np.cos(theta)

        return x, y, z

    def cuboid(center, axes, n_points):
        ux = np.random.uniform(-axes[0], axes[0], n_points)
        uy = np.random.uniform(-axes[1], axes[1], n_points)
        uz = np.random.uniform(-axes[2], axes[2], n_points)

        x = center[0] + ux
        y = center[1] + uy
        z = center[2] + uz

        return x, y, z

    # Define the centers and axes of the ellipsoids
    if snowman_type is None:
        snowman_type = np.random.choice(['arms', 'nose', 'triple'])
    
    assert snowman_type in ['arms', 'nose', 'triple'], "Invalid snowman type. Choose from 'arms', 'nose', or 'triple'."

    if snowman_type == 'arms':
        ellipse_points = n_points // 3
        arm_point1 = (n_points - 2 * ellipse_points) // 2
        arm_point2 = n_points - 2 * ellipse_points - arm_point1
    elif snowman_type == 'nose':
        ellipse_points = n_points // 3
        nose_points = n_points - 2 * ellipse_points
    elif snowman_type == 'triple':
        ellipse_points = n_points // 3
        third_ellipse_points = n_points - 2 * ellipse_points

    center_x = np.random.uniform(min_center_x, max_center_x)
    center_y = np.random.uniform(min_center_y, max_center_y)
    center_z1 = np.random.uniform(min_center_z, max_center_z)
    center_z2 = np.random.uniform(center_z1 + 0.5, max_center_z + 0.5)

    axes1 = np.random.uniform(0.5, 1.0, size=3)
    axes2 = np.random.uniform(0.3, 0.8, size=3)
    centers = [(center_x, center_y, center_z1), (center_x, center_y, center_z2)]
    axes = [axes1, axes2]

    points = []
    for center, axis in zip(centers, axes):
        x, y, z = ellipsoid(center, axis, ellipse_points)
        points.append(np.vstack((x, y, z)).T)

    if snowman_type == 'arms':
        xy = np.random.choice([0, 1])
        arm_length = np.random.uniform(0.5, 1.0)
        arm_width = np.random.uniform(0.1, 0.2)
        
        if xy == 0:
            arm_centers = [(center_x - axes1[0] - arm_length / 2, center_y, center_z1), (center_x + axes1[0] + arm_length / 2, center_y, center_z1)]
            arm_axes = (arm_length, arm_width, arm_width)
        else:
            arm_centers = [(center_x, center_y - axes1[1] - arm_length / 2, center_z1), (center_x, center_y + axes1[1] + arm_length / 2, center_z1)]
            arm_axes = (arm_width, arm_length, arm_width)

        x, y, z = cuboid(arm_centers[0], arm_axes, arm_point1)
        points.append(np.vstack((x, y, z)).T)

        x, y, z = cuboid(arm_centers[1], arm_axes, arm_point2)
        points.append(np.vstack((x, y, z)).T)

    elif snowman_type == 'nose':
        nwes = np.random.choice([0, 1, 2, 3])
        nose_length = np.random.uniform(0.3, 0.5)
        nose_width = np.random.uniform(0.1, 0.2)

        if nwes == 0:
            nose_axes = (nose_length, nose_width, nose_width)
            nose_center = (center_x + axes2[0] + nose_length / 2, center_y, center_z2)
        elif nwes == 1:
            nose_axes = (nose_length, nose_width, nose_width)
            nose_center = (center_x - axes2[0] - nose_length / 2, center_y, center_z2)
        elif nwes == 2:
            nose_axes = (nose_width, nose_length, nose_width)
            nose_center = (center_x, center_y + axes2[1] + nose_length / 2, center_z2)
        else:
            nose_axes = (nose_width, nose_length, nose_width)
            nose_center = (center_x, center_y - axes2[1] - nose_length / 2, center_z2)

        x, y, z = cuboid(nose_center, nose_axes, nose_points)
        points.append(np.vstack((x, y, z)).T)

    elif snowman_type == 'triple':
        center_z3 = np.random.uniform(center_z2 + 0.5, max_center_z + 1.0)
        axes3 = np.random.uniform(0.2, 0.6, size=3)
        centers.append((center_x, center_y, center_z3))
        axes.append(axes3)
        x, y, z = ellipsoid(centers[-1], axes[-1], third_ellipse_points)
        points.append(np.vstack((x, y, z)).T)

    snowman_points = np.vstack(points)
    snowman_points += noise * np.random.randn(*snowman_points.shape)  # Add noise
    return snowman_points


def generate_snowmans(n_snowmans=1000, snowman_types = None, n_points_per_snowman=2048):
    snowmans = []
    if snowman_types is None:
        snowman_types = [None for i in range(n_snowmans)]
    for i in tqdm(range(n_snowmans), desc="Generating Snowmans"):
        snowman = make_ellipsoid_snowman(snowman_type=snowman_types[i], n_points=n_points_per_snowman)
        snowmans.append(snowman)
    return np.array(snowmans, dtype=np.float32)


def normalize_point_cloud_np(pc):
    # pc: (N, 3)
    center = np.mean(pc, axis=0, keepdims=True)
    pc = pc - center
    scale = np.max(np.linalg.norm(pc, axis=1))
    return pc / (scale + 1e-8)


def normalize_batch_point_clouds_np(batch_pc):
    # batch_pc: (B, N, 3)
    return np.stack([normalize_point_cloud_np(pc) for pc in batch_pc], axis=0).astype(np.float32)


n_snowmans = 2000
snowman_types = ['arms', 'nose', 'triple'] * (n_snowmans // 3) + ['arms'] * (n_snowmans % 3)

snowmans = generate_snowmans(n_snowmans=n_snowmans, snowman_types=snowman_types, n_points_per_snowman=2048)
snowmans = normalize_batch_point_clouds_np(snowmans)  # sample-wise center/scale normalization

snowmans_t = torch.from_numpy(snowmans).float()
trainset = torch.utils.data.TensorDataset(snowmans_t, snowmans_t)