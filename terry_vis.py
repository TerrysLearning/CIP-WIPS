# A special visualisation code

from typing import Any
import open3d as o3d
import numpy as np
from copy import deepcopy
import random

def get_color_lib():
    color_lib = np.array([
            [0, 1, 1],      # Cyan
            [0, 0, 1],      # Blue
            [0, 1, 0],      # Green
            [1, 1, 0],      # Yellow
            [1, 0.8, 0],        # Gold
            [1, 0.5, 0.5],      # Pink
            [0.6, 0.8, 0.2],    # Lime
            [1, 0, 0],          # Red
            [0.5, 0, 0.5],     # Purple
            [0.5, 0.5, 0],     # Olive
            [0, 0.5, 0.5],     # Teal
            [0.5, 0, 0],       # Maroon
            [1, 0, 1],         # Magenta
            [0, 0.8, 0.8],      # Turquoise
            [0.5, 0, 0.5],      # Indigo
            [0.7, 0.7, 0.7],    # Silver
            [1, 0.5, 0],        # Orange
            [0.5, 0, 0.5],      # Indigo
            [0, 1, 0.5],        # Spring green 
            [0, 0.8, 0.8],      # Turquoise
            [0.8, 0.2, 0.8],    # Fuchsia
            [0.6, 0.6, 0.6],    # Gray
        ])
    return color_lib


def color_indexes1():
    c_indexes = np.array([1, 14, 8, 20, 8, 7, 10, 17, 17,  6,  0,  1,  3, 20, 10,  1,  2, 21,  3, 15, 12,  4,  7, 12,13, 5, 9,14])
    c_indexes[9] = 6
    c_indexes[19] = 15
    c_indexes[17] = 21
    c_indexes[10] = 1 # 0
    c_indexes[11] = 4 
    c_indexes[16] = 2
    c_indexes[18] = 3
    c_indexes[13] = 0 # 1 
    return c_indexes

def color_indexes2():
    color_indexes1 = np.array([0,1,2,3,4,5,6,8,4,2,1,0])
    return color_indexes1


def get_cylinders(box, color=[1,0,0]):
    line_set = o3d.geometry.LineSet.create_from_axis_aligned_bounding_box(box)
    cylinders = []
    radius = 0.01
    for i in range(len(line_set.lines)):
        line_indices = line_set.lines[i]
        start_index, end_index = line_indices[0], line_indices[1]
        start_point = np.asarray(line_set.points)[start_index]
        end_point = np.asarray(line_set.points)[end_index]

        if abs(start_point[0]-end_point[0])<0.01 and abs(start_point[1]-end_point[1])<0.01:
            rotation_matrix = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        elif abs(start_point[0]-end_point[0])<0.01 and abs(start_point[2]-end_point[2])<0.01:
            rotation_matrix = np.array([[1, 0, 0],[0, 0, -1], [0, 1, 0]])
        elif abs(start_point[1]-end_point[1])<0.01 and abs(start_point[2]-end_point[2])<0.01:
            rotation_matrix = np.array([[0, 0, 1],[0, 1, 0], [-1, 0, 0]])

        height = np.linalg.norm(end_point - start_point)
        center = (start_point + end_point) / 2.0
        cylinder_mesh = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=height)
        cylinder_mesh.translate(center)
        cylinder_mesh.rotate(rotation_matrix, center=center)

        num_vertices = len(cylinder_mesh.vertices)
        colors = np.tile(color, (num_vertices, 1))
        cylinder_mesh.vertex_colors = o3d.utility.Vector3dVector(colors)

        cylinders.append(cylinder_mesh)
    return cylinders




class TerryVis():
    def __init__(self, coords, colors) -> None:
        self.coords = coords
        self.colors = (colors - np.min(colors, axis=0))
        self.colors = self.colors/np.max(self.colors, axis=0)
        self.instance_labels = None
        self.semantic_labels = None 
        self.superpoints = None 
        self.mesh = None

    def set_instances(self, instance_labels):
        self.instance_labels = instance_labels

    def set_semantics(self, semantic_labels):
        self.semantic_labels = semantic_labels
    
    def set_superpoints(self, superpoints):
        self.superpoints = superpoints

    def set_mesh(self, mesh):
        self.mesh = mesh

    def see_scene(self):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.coords)  
        pcd.colors = o3d.utility.Vector3dVector(self.colors)
        draw_items = [pcd]
        # blue is z, red is y, green is x
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
        draw_items.append(coord_frame)
        o3d.visualization.draw_geometries(draw_items)

    def see_sps(self, seed=0, percentage=0.5):
        if self.superpoints is None:
            raise ValueError("Call the function set_superpoints first")
        else:
            np.random.seed(seed)
            colors = self.colors.copy()
            for sp in np.unique(self.superpoints):
                sp_mask = self.superpoints==sp
                colors[sp_mask] = (1-percentage)*colors[sp_mask] + percentage*np.random.rand(3)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(self.coords)  
            pcd.colors = o3d.utility.Vector3dVector(colors)
            o3d.visualization.draw_geometries([pcd])

    def draw_sps(self, seed=0, percentage=0.5):
        if self.superpoints is None:
            raise ValueError("Call the function set_superpoints first")
        else:
            np.random.seed(seed)
            
            colors = self.colors.copy()
            for sp in np.unique(self.superpoints):
                sp_mask = self.superpoints==sp
                colors[sp_mask] = (1-percentage)*colors[sp_mask] + percentage*np.random.rand(3)
            mesh = deepcopy(self.mesh)
            mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
            o3d.visualization.draw_geometries([mesh])


    def see_boxes(self, max_corners, min_corners, box_color=np.array([1,0,0])):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.coords)
        pcd.colors = o3d.utility.Vector3dVector(self.colors)
        boxes = []
        for i in range(len(max_corners)):
            box = o3d.geometry.AxisAlignedBoundingBox(min_bound=min_corners[i], max_bound=max_corners[i])
            box.color = box_color
            boxes.append(box)
        draw_items = [pcd] + boxes
        o3d.visualization.draw_geometries(draw_items)



    def see_mask(self, mask, percentage=0.5, color=np.array([1,0,0])):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.coords)  
        colors = self.colors.copy()
        colors[mask] = percentage * color + (1-percentage) * colors[mask] 
        pcd.colors = o3d.utility.Vector3dVector(colors)
        draw_items = [pcd]
        # blue is z, red is y, green is x
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
        draw_items.append(coord_frame)
        o3d.visualization.draw_geometries(draw_items)


    def draw_scene(self):
        if self.mesh is None:
            raise ValueError("Call the function set_mesh first")
        o3d.visualization.draw_geometries([self.mesh])


    def draw_all_instance_masks(self, percentage=0.5, loop_each=False, sepcific_color=False, seed=0):
        colors = self.colors.copy()
        mesh = deepcopy(self.mesh)
        color_lib = get_color_lib()
        if sepcific_color:
            c_indexes = color_indexes1()
        else:
            np.random.seed(seed)
            c_indexes = np.random.randint(0, len(color_lib), len(np.unique(self.instance_labels)))
        color_lib = color_lib[c_indexes]    
        i = 0
        for ins in np.unique(self.instance_labels):
            if ins ==-100:
                continue
            ins_mask = self.instance_labels == ins  
            if loop_each:
                self.see_mask(ins_mask)
            colors[ins_mask] = percentage * color_lib[i] + (1-percentage) * colors[ins_mask]
            i += 1
        mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
        o3d.visualization.draw_geometries([mesh])


    def draw_boxes(self, max_corners, min_corners, box_color=np.array([1,0,0]), sepcific_color=False, seed=0):
        if self.mesh is None:
            raise ValueError("Call the function set_mesh first")
        draw_items = [self.mesh]    
        color_lib = get_color_lib()
        if sepcific_color:
            c_indexes = color_indexes1()
        else:
            np.random.seed(seed)
            c_indexes = np.random.randint(0, len(color_lib), len(max_corners))
        color_lib = color_lib[c_indexes] 
        for i in range(len(max_corners)):
            box = o3d.geometry.AxisAlignedBoundingBox(min_bound=min_corners[i], max_bound=max_corners[i])
            cylinders = get_cylinders(box, color_lib[i])
            for c in cylinders:
                draw_items.append(c)
        # coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
        # draw_items.append(coord_frame)
        o3d.visualization.draw_geometries(draw_items) 


    def draw_box_with_instaces_masks(self, max_corners, min_corners, box_color=np.array([1,0,0]), seed=0, percentage=0.5):
        if self.mesh is None:
            raise ValueError("Call the function set_mesh first")
        draw_items = []    
        mesh = deepcopy(self.mesh)
        color_lib = get_color_lib()
        np.random.seed(seed)
        c_indexes = np.random.randint(0, len(color_lib), len(max_corners))
        color_lib = color_lib[c_indexes] 

        i = 0
        colors = self.colors.copy()
        for ins in np.unique(self.instance_labels):
           
            if ins ==-100:
                continue
            ins_mask = self.instance_labels == ins  
            colors[ins_mask] = percentage * color_lib[i] + (1-percentage) * colors[ins_mask]
            i += 1
        mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
        draw_items.append(mesh)
        
        for i in range(len(max_corners)):
            box = o3d.geometry.AxisAlignedBoundingBox(min_bound=min_corners[i], max_bound=max_corners[i])
            cylinders = get_cylinders(box)
            for c in cylinders:
                draw_items.append(c)
       
        o3d.visualization.draw_geometries(draw_items) 


    def draw_boxes_compare(self, max_corners, min_corners, box_color1=np.array([0.7,0.7,0.7]), box_color2=np.array([1,0,0])):
        if self.mesh is None:
            raise ValueError("Call the function set_mesh first")
        draw_items = [self.mesh]     
        for i in range(len(max_corners)):
            if i < 0.5 * len(max_corners):
                box_color = box_color2
            else:
                box_color = box_color1
            box = o3d.geometry.AxisAlignedBoundingBox(min_bound=min_corners[i], max_bound=max_corners[i])
            cylinders = get_cylinders(box, box_color)
            for c in cylinders:
                draw_items.append(c)
        # coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
        # draw_items.append(coord_frame)
        o3d.visualization.draw_geometries(draw_items)     



    def draw_boxes_compare2(self, max_corners, min_corners):
        if self.mesh is None:
            raise ValueError("Call the function set_mesh first")
        draw_items = [self.mesh]     
        color_lib = get_color_lib()
        n = int(0.5 * len(max_corners))
        # c_indexes = np.random.randint(0, len(color_lib), n)
        c_indexes = color_indexes2()
        color_lib = color_lib[c_indexes]    
        for i in range(len(max_corners)):
            if i < 0.5 * len(max_corners):
                box_color = color_lib[i]
            else:
                box_color = color_lib[i-n] * 0.5
            box = o3d.geometry.AxisAlignedBoundingBox(min_bound=min_corners[i], max_bound=max_corners[i])
            cylinders = get_cylinders(box, box_color)
            for c in cylinders:
                draw_items.append(c)
        # coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
        # draw_items.append(coord_frame)
        o3d.visualization.draw_geometries(draw_items)     

