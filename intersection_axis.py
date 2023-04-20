import open3d as o3d
import numpy as np


def get_surface_intersection_with_axes(model:o3d.geometry.TriangleMesh, model_origin:np.ndarray=None):
    if model_origin is None:
        model_origin = model.get_center()

    max_bound = model.get_max_bound()
    min_bound = model.get_min_bound()

    intersection_with_axes = {}
    max_origin = np.vstack([model_origin, model_origin, model_origin])
    min_origin = np.vstack([model_origin, model_origin, model_origin])
    for i in range(3):
        max_origin[i, i] = max_bound[i]
        min_origin[i, i] = min_bound[i]
        if i == 0:
            intersection_with_axes['x_max'] = max_origin[i]
            intersection_with_axes['x_min'] = min_origin[i]
        if i == 1:
            intersection_with_axes['y_max'] = max_origin[i]
            intersection_with_axes['y_min'] = min_origin[i]
        if i == 2:
            intersection_with_axes['z_max'] = max_origin[i]
            intersection_with_axes['z_min'] = min_origin[i]
    return intersection_with_axes

def draw_intersection_axes(model:o3d.geometry.TriangleMesh, intersections:dict):
    spheres = o3d.geometry.TriangleMesh()
    for key in intersections.keys():
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.5)
        sphere.translate(intersections[key])
        if 'x' in key:
            sphere.paint_uniform_color([1, 0, 0])
        if 'y' in key:
            sphere.paint_uniform_color([0, 1, 0])
        if 'z' in key:
            sphere.paint_uniform_color([0, 0, 1])
        spheres += sphere
    o3d.visualization.draw_geometries([model, spheres])

if __name__ == '__main__':

    model = o3d.io.read_triangle_mesh('./Cola_bottle.stl')
    model.compute_vertex_normals()
    object_origin=model.get_center()
    origin_model = o3d.geometry.TriangleMesh.create_coordinate_frame(origin=object_origin, size=5)

    intr_sect_axes = get_surface_intersection_with_axes(model, object_origin)
    draw_intersection_axes(model, intr_sect_axes)


    print("Intersections : ", intr_sect_axes)