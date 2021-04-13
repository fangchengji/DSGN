import numpy as np

def project_image_to_rect(uv_depth, P):
    # uv_depth (3, N)

    c_u = P[0,2]
    c_v = P[1,2]
    f_u = P[0,0]
    f_v = P[1,1]
    b_x = P[0,3]/(-f_u) # relative 
    b_y = P[1,3]/(-f_v)

    # use camera coordinate
    n = uv_depth.shape[1]
    x = ((uv_depth[0]-c_u)*uv_depth[2])/f_u + b_x
    y = ((uv_depth[1]-c_v)*uv_depth[2])/f_v + b_y
    return np.stack([x, y, uv_depth[2]], axis=0)

def project_disp_to_depth(points_cam, Proj, baseline=0.54):
    xs, ys, disp = points_cam[0:1], points_cam[1:2], points_cam[2:3]
    _, w, h, d = disp.shape

    mask = disp > 0
    depth = Proj[0,0] * baseline / (disp + 1. - mask)
    points = np.concatenate([xs, ys, depth], axis=0)
    points = points.reshape((3, -1))

    # camera coordinate
    cloud = project_image_to_rect(points, Proj)
    cloud = cloud.reshape(3, w, h, d)
    return cloud

def clip_boxes(boxes, size, remove_empty=False):
    boxes[:, [0,2]] = boxes[:, [0,2]].clip(0, size[0] - 1)
    boxes[:, [1,3]] = boxes[:, [1,3]].clip(0, size[1] - 1)
    if remove_empty:
        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        return boxes, keep
    else:
        return boxes

def get_dimensions(corners):
    assert corners.shape == (3, 8)
    height_group = [(0, 4), (1, 5), (2, 6), (3, 7)]
    width_group = [(0, 1), (2, 3), (4, 5), (6, 7)]
    length_group = [(0, 3), (1, 2), (4, 7), (5, 6)]
    vector_group = [(0, 3), (1, 2), (4, 7), (5, 6)]
    height = 0.0
    width = 0.0
    length = 0.0
    vector = np.zeros(2, dtype=np.float32)
    for index_h, index_w, index_l, index_v in zip(height_group, width_group, length_group, vector_group):
        height += np.linalg.norm(corners[:, index_h[0]] - corners[:, index_h[1]])
        width += np.linalg.norm(corners[:, index_w[0]] - corners[:, index_w[1]])
        length += np.linalg.norm(corners[:, index_l[0]] - corners[:, index_l[1]])
        vector[0] += (corners[:, index_v[0]] - corners[:, index_v[1]])[0]
        vector[1] += (corners[:, index_v[0]] - corners[:, index_v[1]])[2]

    height, width, length = height*1.0/4, width*1.0/4, length*1.0/4
    rotation_y = -np.arctan2(vector[1], vector[0])

    return [height, width, length, rotation_y]

def generate_depth_map_from_rect_points(pc_rect, height, width, calib, min_depth=2):
    pts_2d = calib.project_rect_to_image(pc_rect)
    fov_inds = (pts_2d[:, 0] < width) & (pts_2d[:, 0] >= 0) & \
               (pts_2d[:, 1] < height) & (pts_2d[:, 1] >= 0)
    fov_inds = fov_inds & (pc_rect[:, 2] >= min_depth)       # depth > 2
    imgfov_pc_rect = pc_rect[fov_inds, :]
    imgfov_pts_2d = pts_2d[fov_inds, :]
    depth_map = np.zeros((height, width)) + 1000           # set 1000 for default
    imgfov_pts_2d = np.round(imgfov_pts_2d).astype(int)
    # clip to image width and height
    imgfov_pts_2d[:, 0] = np.clip(imgfov_pts_2d[:, 0], 0, width - 1)
    imgfov_pts_2d[:, 1] = np.clip(imgfov_pts_2d[:, 1], 0, height - 1)
    for i in range(imgfov_pts_2d.shape[0]):
        depth = imgfov_pc_rect[i, 2]
        # Use the min depth if more than one points projected to the same image pixel 
        depth_map[imgfov_pts_2d[i, 1], imgfov_pts_2d[i, 0]] = min(depth, 
                                                                  depth_map[imgfov_pts_2d[i, 1], imgfov_pts_2d[i, 0]])
    depth_map[depth_map == 1000] = -1       # set -1 for default
    
    return depth_map