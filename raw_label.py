import numpy as np
import torch
import os 
import open3d as o3d 
import cv2
from helper import * 
from terry_vis import TerryVis
import scipy.stats as stats
import multiprocessing 


class LabelInitialisation():
    def __init__(self, scene_folder, scan_id, noise=0.05, noise_mode=0, use_inclu=False) -> None:
        self.scene_folder = scene_folder
        self.scan_id = scan_id
        save_folder = 'raw_label_m' + str(noise_mode) + '_n' + str(noise).split('.')[1]
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        save_file = scan_id + '_info.pth'
        self.save_path = os.path.join(save_folder, save_file)
        self.noise = noise
        self.noise_mode = noise_mode  # 0 means looser box based on bounds, 1 means looser box of some specific distance, 2 means random looser
        self.use_inclu = use_inclu 
        
        # load color and points
        ply_path = os.path.join(self.scene_folder, self.scan_id + '_vh_clean_2.ply')
        self.align_txt_path = os.path.join(self.scene_folder, self.scan_id + '.txt')
        mesh = o3d.io.read_triangle_mesh(ply_path)
        self.coords = np.asarray(mesh.vertices) 
        self.colors = np.asarray(mesh.vertex_colors)

        # load gt label and superpoints
        file_stuff = os.path.join('gt_labels', scan_id+'_labels.pth')
        self.superpoints, self.semantic_labels, self.instance_labels = torch.load(file_stuff)
        self.semantic_labels = relabel2(self.semantic_labels)

        # raw 3D points label generation   
        self.ret_true = self.get_true_label()
        self.ret_ins, self.ret_baseline, self.max_corners, self.min_corners, self.b_sems, self.b_inst = self.get_rough_label()
    

    def get_true_label(self):
        '''
        return a dict with instance id and correspond superpoints under human annotation 
        '''
        ret_true = dict()
        instance_labels = self.instance_labels.astype(int)
        for ins in np.unique(instance_labels): 
            ret_true[ins] = []
        for sp in np.unique(self.superpoints):
            sp_mask = self.superpoints == sp
            sp_instance = stats.mode(instance_labels[sp_mask], keepdims=True)[0][0]
            ret_true[sp_instance].append(sp)
        for k,v in ret_true.items(): 
            ret_true[k] = np.array(v)
        return ret_true


    def add_noise_boxes(self, max_corners, min_corners):
        if self.noise_mode == 0:
            bounds = max_corners - min_corners
            extra = 0.5 *  bounds * self.noise   
        elif self.noise_mode == 1:
            extra = self.noise
        elif self.noise_mode == 2:
            extra = 0 
            print('Havent done this part yet')
            assert 0 
        return (max_corners + extra), (min_corners - extra)
    

    def get_rough_label(self):
        '''
        get raw positive points for each instance
        '''
        print('generating raw labels')
        Rt = get_align_matrix(self.align_txt_path)
        coords = mul(self.coords, Rt) # axis-align coords
        max_corners, min_corners, b_sems, b_insts = compute_boxes(self.instance_labels, self.semantic_labels, coords) # compute boxes
        max_corners, min_corners = self.add_noise_boxes(max_corners, min_corners)
        # prepare
        bounds = max_corners - min_corners
        bb_volume = np.prod(bounds, axis=1)
        box_occ = in_box(coords, max_corners[:,None], min_corners[:,None]) # each point is in which boxes
        activations_per_point = [np.argwhere(box_occ[:, i] == 1).reshape(-1) for i in range(len(coords))]  # each points relative box ids
        if self.use_inclu:
            inclu_pairs = inclusion_pairs(max_corners, min_corners, bb_volume, 0.9) 
        # get positive points for each bounding box
        ret_ins = dict()
        #ret_sp = dict()
        ret_baseline = dict()
        for ins in b_insts:
            ret_ins[ins] = [] 
            ret_baseline[ins] = []
        for sp in np.unique(self.superpoints):
            # ret_sp[sp] = []
            sp_mask = self.superpoints == sp 
            sp_box_ids_num = dict()  # number of box ids on each superpoint 
            for i, box_ids in enumerate(activations_per_point):
                if not sp_mask[i]:
                    continue
                for bid in box_ids:
                    if bid not in sp_box_ids_num.keys():
                        sp_box_ids_num[bid] = 1
                    else:
                        sp_box_ids_num[bid] += 1
            sp_box_ids = []
            for bid, num in sp_box_ids_num.items():
                if num >= 0.99*np.count_nonzero(sp_mask):
                    sp_box_ids.append(bid)
            if self.use_inclu:
                for p in inclu_pairs:
                    if (p[1] in sp_box_ids) and (p[0] in sp_box_ids):  # p_1 is the bigger box id, p2 is the smaller box id 
                        sp_box_ids.remove(p[1]) 
            for bid in sp_box_ids:
                ret_ins[bid].append(sp)
                #ret_sp[sp].append(bid)
            if len(sp_box_ids)>0:
                smallest_box_id = sp_box_ids[np.argmin(bb_volume[sp_box_ids])] 
                ret_baseline[smallest_box_id].append(sp)
        for k, v in ret_ins.items():
            ret_ins[k] = np.array(v)     
        # for k, v in ret_sp.items():
        #     ret_sp[k] = np.array(v)   
        for k, v in ret_baseline.items():
            ret_baseline[k] = np.array(v)   
        return ret_ins, ret_baseline, max_corners, min_corners, b_sems, b_insts
    

    def save(self):
        items = (self.ret_ins, self.ret_baseline, self.max_corners, \
                 self.min_corners, self.b_sems, self.b_inst, self.ret_true)
        torch.save(items, self.save_path)
        print('saved ', self.scan_id)

    
# ------------------------------------------------------------------------

class CheckLabel():
    def __init__(self, scene_folder, scan_id, noise=0.05, noise_mode=0) -> None:
        self.scan_id = scan_id
        self.noise = noise
        self.noise_mode = noise_mode
        self.scene_folder = scene_folder
        folder = 'raw_label_m' + str(noise_mode) + '_n' + str(noise).split('.')[1]
        file = scan_id + '_info.pth'
        self.file_path = os.path.join(folder, file)
        if not os.path.exists(self.file_path):
            print('The file havent load yet')
            raise ValueError
        self.ret_ins, self.ret_baseline,  self.max_corners, self.min_corners, \
            self.b_sems, self.b_inst, self.ret_true = torch.load(self.file_path)

        # load color and points
        ply_path = os.path.join(self.scene_folder, self.scan_id + '_vh_clean_2.ply')
        self.align_txt_path = os.path.join(self.scene_folder, self.scan_id + '.txt')
        mesh = o3d.io.read_triangle_mesh(ply_path)
        self.coords = np.asarray(mesh.vertices) 
        self.colors = np.asarray(mesh.vertex_colors)
        self.mesh = mesh

        # load gt info
        file_stuff = os.path.join('gt_labels', scan_id+'_labels.pth')
        self.superpoints, self.semantic_labels, self.instance_labels = torch.load(file_stuff)
    

    def vis_boxes(self):
        Rt = get_align_matrix(self.align_txt_path)
        coords = mul(self.coords, Rt) # axis-align coords
        vis = TerryVis(coords, self.colors)
        vis.see_boxes(self.max_corners, self.min_corners)


    def vis_raw(self):
        ret = self.ret_ins
        self.vis_inst_label(ret)
       
    def vis_true(self):
        ret = self.ret_true
        self.vis_inst_label(ret)

    def vis_baseline(self):
        ret = self.ret_baseline
        self.vis_inst_label(ret)

    def vis_inst_label(self, ret):
        vis = TerryVis(self.coords, self.colors)
        for k, v in ret.items():
            if k==-100:
                continue
            else:
                print('instance id : ', k)
                print('semantic label : ', self.b_sems[k])
                mask = np.zeros(len(self.coords), dtype=bool)
                for s in v:
                    mask[self.superpoints==s] = True
                vis.see_mask(mask)

    def wrong_num_raw(self):
        ret = self.ret_ins
        self.get_wrong_num(ret)
    
    def wrong_num_baseline(self):
        ret = self.ret_baseline
        self.get_wrong_num(ret)
    
    def get_wrong_num(self, ret):
        wrong_sps = 0
        true_total = 0
        raw_total = 0  
        for k, v in self.ret_true.items():
            if k==-100:
                continue
            if self.b_sems[k] != -100:
                l1 = set(v)
                l2 = set(ret[k])
                wrong_sps += len(l2) - len(l1 & l2)
                true_total += len(l1) 
                raw_total += len(l2)
        print('wrong sps : ', wrong_sps)
        print('true sps :', true_total)
        print('raw sps :', raw_total)


    def save_info_spf(self, ret, new):
        coords = self.coords - self.coords.mean(0)
        colors = self.colors * 2 -1 
        instance_labels = np.zeros(len(coords))-100
        semantic_labels = np.zeros(len(coords))-100
        for k, v in ret.items():
            mask = np.isin(self.superpoints, v)
            instance_labels[mask] = k
            sem_id = self.b_sems[k]
            if sem_id != -100:
                semantic_labels[mask] = sem_id -1 
        items = (coords, colors, self.superpoints, semantic_labels, instance_labels)
        semantic_labels = np.where(semantic_labels==0, -100, semantic_labels-1)
        if new:
            suffix = 'new'
        else:
            suffix = 'baseline'
        save_folder = suffix + '_label_m' + str(self.noise_mode) + '_n' + str(self.noise).split('.')[1]
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        file = self.scan_id + '_stuff.pth'
        torch.save(items, os.path.join(save_folder, file)) 
        print('saved', file)


    def save_info_softgroup(self, ret, new):
        'Background sem=0 no need saving superpoints'
        coords = self.coords - self.coords.mean(0)
        colors = self.colors * 2 -1 
        instance_labels = np.zeros(len(coords))-100
        semantic_labels = np.zeros(len(coords))
        for k, v in ret.items():
            if k==-100:
                continue
            mask = np.isin(self.superpoints, v)
            instance_labels[mask] = k
            semantic_labels[mask] = self.b_sems[k]
        items = (coords, colors, semantic_labels, instance_labels)
        if new:
            suffix = 'new'
        else:
            suffix = 'baseline'
        save_folder = suffix + '_label_m' + str(self.noise_mode) + '_n' + str(self.noise).split('.')[1]
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        file = self.scan_id + '_stuff.pth'
        torch.save(items, os.path.join(save_folder, file)) 
        print('saved', file)


    def save_info_box2mask(self, ret, new):
        self.semantic_labels = relabel2(self.semantic_labels)
        superpoints = np.asarray(self.superpoints, dtype=int)
        inst_per_sp = np.zeros(len(np.unique(self.superpoints)), dtype=int)-1
        for k, v in ret.items():
            if len(v) != 0:
                inst_per_sp[v] = k
        self.mesh.compute_vertex_normals()
        self.mesh.normalize_normals()
        normals = np.asarray(self.mesh.vertex_normals)
        Rt = get_align_matrix(self.align_txt_path)
        coords = mul(self.coords, Rt)
        items = (inst_per_sp, self.max_corners, self.min_corners, self.b_sems, coords, 
                 self.colors, superpoints, self.semantic_labels, self.instance_labels, normals)
        
        if new:
            suffix = 'new'
        else:
            suffix = 'baseline'
        save_folder = suffix + '_label_m' + str(self.noise_mode) + '_n' + str(self.noise).split('.')[1]
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        torch.save(items, os.path.join(save_folder , self.scan_id + '_stuff.pth'))
        print('saved', self.scan_id + '_stuff.pth')


    def save_info(self, type='spf', new=False):
        if not new:
            ret = self.ret_baseline
        else:
            corr_folder = 'corr_labels_m' + str(self.noise_mode) + '_n' + str(self.noise).split('.')[1]
            corr_file = self.scan_id + '_co.pth'
            ret = torch.load(os.path.join(corr_folder, corr_file))
        if type == 'spf':
            self.save_info_spf(ret, new)
        elif type == 'softgroup':
            self.save_info_softgroup(ret, new)
        elif type == 'box2mask':
            self.save_info_box2mask(ret, new)

    def test_acc(self, ret):
        wrong_sps = set()
        wrong_pts = np.zeros(len(self.superpoints), dtype=bool)
        for k, v in self.ret_true.items():
            if k==-100:
                continue
            if self.b_sems[k] != -100:
                l1 = set(v)
                l2 = set(ret[k])
                wrong_sps_ins = l2 - l1
                wrong_sps = wrong_sps | wrong_sps_ins
                wrong_pts = wrong_pts | np.isin(self.superpoints, np.array(list(wrong_sps)))
        return len(wrong_sps), np.count_nonzero(wrong_pts)
    

    def acc_compare_with_new(self):
        corr_folder = 'corr_labels_m' + str(self.noise_mode) + '_n' + str(self.noise).split('.')[1]
        corr_file = self.scan_id + '_co.pth'
        ret_new = torch.load(os.path.join(corr_folder, corr_file))
        acc_new = self.test_acc(ret_new)
        acc_base = self.test_acc(self.ret_baseline)
        return acc_new, acc_base



# ------------------------------------------------------------

def load_data(scan_id, scene_folder, noise, noise_mode, use_inclu=True):
    T = LabelInitialisation(scene_folder, scan_id, noise, noise_mode, use_inclu=use_inclu)
    T.save()


def check_data(scan_id, scene_folder, noise, noise_mode):
    T = CheckLabel(scene_folder, scan_id, noise, noise_mode)
    T.save_info(type='spf', new=True)
    # T.save_box2mask_info()
    # T.wrong_num_raw()
    # # T.wrong_num_baseline()
    # T.save_baseline_spf()
    # T.save_baseline_softgroup()
    # T.vis_baseline()
    # T.vis_boxes()
    # T.vis_raw()
    # T.vis_true()


def process(line):
    scan_id = line[:-1]
    scene_folder = os.path.join('ScanNetv2', 'scans', scan_id)
    noise = 0.0
    noise_mode = 0 
    use_inclu = True 
    if os.path.exists(os.path.join('raw_label_m0_n3', scan_id+'_info.pth')):
        print('already processed')
    else:
        load_data(scan_id, scene_folder, noise, noise_mode, use_inclu=use_inclu)
    # check_data(scan_id, scene_folder, noise, noise_mode)


def process2(line):
    scan_id = line[:-1]
    scene_folder = os.path.join('ScanNetv2', 'scans', scan_id)
    noise = 0.3
    noise_mode = 0 
    T = CheckLabel(scene_folder, scan_id, noise, noise_mode)
    T.save_info(type='softgroup', new=True)


def process3(line):
    scan_id = line[:-1]
    scene_folder = os.path.join('ScanNetv2', 'scans', scan_id)
    noise = 0.1
    noise_mode = 0
    T = CheckLabel(scene_folder, scan_id, noise, noise_mode)
    acc = T.acc_compare_with_new()
    print(scan_id, acc)
    return acc 
    
def run():
    np.random.seed(0)   
    file = open('scannetv2_train.txt')
    lines = file.readlines()
    max_workers = 16
    pool = multiprocessing.Pool(max_workers)
    results = pool.map(process2, lines)
    # ///////
    # total_wrong_sp_new = 0
    # total_wrong_sp_base = 0 
    # total_wrong_pt_new = 0 
    # total_wrong_pt_base = 0
    # for acc in results:
    #     if acc:
    #         acc_new, acc_base = acc
    #         total_wrong_sp_base += acc_base[0]
    #         total_wrong_pt_base += acc_base[1]
    #         total_wrong_sp_new += acc_new[0]
    #         total_wrong_pt_new += acc_new[1]
    # print('total_wrong_sp_base : ', total_wrong_sp_base)
    # print('total_wrong_pt_base : ', total_wrong_pt_base)
    # print('total_wrong_sp_new : ', total_wrong_sp_new)
    # print('total_wrong_pt_new : ', total_wrong_pt_new)
    # //////


def check(scan_id):
    scene_folder = os.path.join('ScanNetv2', 'scans', scan_id)
    noise = 0.0
    noise_mode = 0
    use_inclu = True
    # load_data(scan_id, scene_folder, noise, noise_mode, use_inclu=use_inclu)
    check_data(scan_id, scene_folder, noise, noise_mode)


if __name__ == '__main__':
    # check('scene0191_00')
    run()

   
    

