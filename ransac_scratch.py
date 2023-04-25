import open3d as o3d
import numpy as np
import copy

def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp],
                                      zoom=0.4459,
                                      front=[0.9288, -0.2951, -0.2242],
                                      lookat=[1.6784, 2.0612, 1.4451],
                                      up=[-0.3402, -0.9189, -0.1996])

def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    print(":: RANSAC registration on downsampled point clouds.")
    print("   Since the downsampling voxel size is %.3f," % voxel_size)
    print("   we use a liberal distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
    return result

def ransac_feature_matching_run(source_down, target_down, source_fpfh, target_fpfh, voxel_size):
    ransac_feature_matching_scratch(source_down, target_down, source_fpfh, target_fpfh, voxel_size,
                                    ransac_n=2)

def ransac_feature_matching_scratch(source, target, source_fpfh, target_fpfh, voxel_size,
                                    ransac_n=3, max_correspondence:float=None, checkers=None, criteria=None):
    if ransac_n < 3 or max_correspondence <= 0.0:
        return o3d.registration.RegistrationResult()
    
    mutual_filter = True
    mutual_consistent_ratio = 2.
    corrs = get_correspondence_from_features(source_fpfh, target_fpfh, mutual_filter, mutual_consistent_ratio)
    return registration_ransac_based_on_corrs(source, target, corrs, 
                                              max_correspondence,
                                              estimation,
                                              ransac_n,
                                              checkers,
                                              criteria)

def registration_ransac_based_on_corrs(source, target, corrs, max_correspondence, estimation, ransac_n,
                                       checkers, criteria):
    raise NotImplementedError

def get_correspondence_from_features(feature1, feature2, mutual_filter:bool, mutual_consistent_ratio:float=None):
    from sklearn.neighbors import NearestNeighbors
    num_searches = 2 if mutual_filter else 1
    features = [feature1, feature2]
    num_pts = [feature1.shape[1], feature2.shape[1]]
    corres = [None] * num_searches
    kMaxThreads = 4 # or your desired number of threads

    kOuterThreads = min(kMaxThreads, num_searches)
    kInnerThreads = max(kMaxThreads // num_searches, 1)

    for k in range(num_searches):
        nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(features[1-k].T)
        distances, indices = nbrs.kneighbors(features[k].T, return_distance=True)
        corres[k] = [(i, indices[i][0]) for i in range(num_pts[k])]

    if not mutual_filter:
        return corres[0]

    corres_mutual = []
    num_src_pts = num_pts[0]
    for i in range(num_src_pts):
        j = corres[0][i][1]
        if corres[1][j][1] == i:
            corres_mutual.append((i, j))

    if len(corres_mutual) >= mutual_consistent_ratio * num_src_pts:
        print("{} correspondences remain after mutual filter".format(len(corres_mutual)))
        return corres_mutual

    print("Too few correspondences ({:d}) after mutual filter, fall back to original correspondences.".format(len(corres_mutual)))
    return corres[0]

def preprocess_point_cloud(pcd, voxel_size):
    print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh

def prepare_dataset(voxel_size):
    print(":: Load two point clouds and disturb initial pose.")

    demo_icp_pcds = o3d.data.DemoICPPointClouds()
    source = o3d.io.read_point_cloud(demo_icp_pcds.paths[0])
    target = o3d.io.read_point_cloud(demo_icp_pcds.paths[1])
    trans_init = np.asarray([[0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0],
                             [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    source.transform(trans_init)
    draw_registration_result(source, target, np.identity(4))

    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
    return source, target, source_down, target_down, source_fpfh, target_fpfh

def main():
    voxel_size = 0.05  # means 5cm for this dataset
    source, target, source_down, target_down, source_fpfh, target_fpfh = prepare_dataset(voxel_size)
    ransac_feature_matching_run(source_down, target_down, source_fpfh, target_fpfh, voxel_size)
    # result_ransac = execute_global_registration(source_down, target_down,
    #                                         source_fpfh, target_fpfh,
    #                                         voxel_size)
    # draw_registration_result(source_down, target_down, result_ransac.transformation)


if __name__ == '__main__':
    main()
