import os
import numpy as np
from terry_vis import TerryVis
from helper import * 
import open3d as o3d
import torch
from tqdm import tqdm
from segment_anything import SamPredictor, sam_model_registry
from terry_vis import TerryVis
from skimage.util import view_as_windows
from scipy import signal 


class TerryClass():
    def __init__(self, 
                 scan_id,
                 log=None,
                 predictor=None,
                 noise_mode=0,
                 noise=0.1,
                 beta = 0.5,
                 bg_w = 20,
                 vis_2dmasks = False,
                 vis_final = False,
                 test_final_acc = True,
                 print_final_acc = True,
                 consider_bg_obj= True,
                 shrink_box = False,
                 only_scanned_bg = False,
                 prompt_combine = False,
                 delete_sel = False,
                 greedy_threshold = 0.99
                ):
        self.scan_id = scan_id
        self.log = log 
        self.noise_mode = noise_mode
        self.noise = noise
        self.predictor = predictor
        self.beta = beta
        self.bg_w = bg_w
        self.vis_2dmasks = vis_2dmasks
        self.vis_final = vis_final
        self.test_final_acc = test_final_acc
        self.print_final_acc = print_final_acc
        self.consider_bg_obj = consider_bg_obj
        self.img_w = 640
        self.img_h = 480
        self.shrink_box = shrink_box
        self.only_scanned_bg = only_scanned_bg # some points are incomplete, don't use them as bg prompts
        self.prompt_combine = prompt_combine
        self.greedy_threshold = greedy_threshold
        self.delete_sel = delete_sel

        # set up pathes
        self.scene_folder = os.path.join('ScanNetv2', 'scans', scan_id)
        self.image_folder = 'images'
        self.image_path = os.path.join('images', scan_id + '_images')
        self.image_proj_folder = 'image_proj'
        self.image_proj_path = os.path.join('image_proj' , scan_id + '_pj.pth')
        self.raw_label_suffix = '_m'+str(noise_mode)+'_n'+str(noise).split('.')[1]
        self.select_res_folder = 'select_res' + self.raw_label_suffix
        self.select_res_path = os.path.join(self.select_res_folder, scan_id + '_se.pth')
        self.save_folder = 'corr_labels' + self.raw_label_suffix
        self.save_path = os.path.join(self.save_folder, scan_id + '_co.pth')
        if not os.path.exists(self.scene_folder):
            print('The scene data doesn\'t exist')
            raise ValueError
        if not os.path.exists(self.image_path):
            print('Images for {} are not loaded yet'.format(self.scan_id))
            raise ValueError 
        if not os.path.exists(self.image_proj_folder):
            os.mkdir(self.image_proj_folder)
        if not os.path.exists(self.select_res_folder):
            os.mkdir(self.select_res_folder)
        if not os.path.exists(self.save_folder):
            os.mkdir(self.save_folder)

        # load color and points
        ply_path = os.path.join(self.scene_folder, self.scan_id + '_vh_clean_2.ply')
        self.align_txt_path = os.path.join(self.scene_folder, self.scan_id + '.txt')
        mesh = o3d.io.read_triangle_mesh(ply_path)
        self.coords = np.asarray(mesh.vertices)
        self.colors = np.asarray(mesh.vertex_colors)

        # load superpoints
        file_stuff = os.path.join('gt_labels', scan_id+'_labels.pth')
        self.superpoints, _, _ = torch.load(file_stuff)

        # load raw_label information
        self.raw_la_path = os.path.join('raw_label' + self.raw_label_suffix, scan_id + '_info.pth')
        if not os.path.exists(self.raw_la_path):
            print('The file havent load yet')
            raise ValueError
        self.ret_ins, self.ret_baseline, self.max_corners, self.min_corners, \
            self.b_sems, self.b_inst, self.ret_true = torch.load(self.raw_la_path)


    # ----------------- 1. prepare the image info -----------------
    def view_proj(self, img_id):
        'get all the points in the view'
        pose_txt = os.path.join(self.image_path, 'pose' , str(img_id) + '.txt')
        P = np.loadtxt(pose_txt)
        intrinsic_txt = os.path.join(os.path.join(self.image_path, 'intrinsic', 'intrinsic_depth.txt')) 
        K = np.loadtxt(intrinsic_txt)
        point_coords_cam = mul(self.coords, np.linalg.inv(P))  # point coordinates under camera coordination 
        point_coords_img = mul(point_coords_cam, K)   # point coordinates under image coordination
        point_depth_img = point_coords_img[:,2]  
        point_coords_img = point_coords_img[:,:2] / point_coords_img[:, 2:] 
        mask_proj = (point_coords_img[:,0]>=0) & (point_coords_img[:,1]>=0) & (point_coords_img[:,0]<self.img_w) & \
            (point_coords_img[:,1]<self.img_h) & (point_depth_img>0) 
        point_coords_img = point_coords_img[mask_proj]
        point_depth_img = point_depth_img[mask_proj]
        point_coords_img = point_coords_img.astype(np.int32)
        'filter the visible points in the view'
        depth_img_path = os.path.join(self.image_path, 'depth', str(img_id)+'.png')
        depth_img = cv2.imread(depth_img_path, -1)            # read 16bit grayscale image
        depth_shift = 1000.0
        depth_img = depth_img/depth_shift
        rel_depth_img = depth_img[point_coords_img[:,1], point_coords_img[:,0]] # relavant depth of points in the camera view
        mask_visible = np.absolute(rel_depth_img - point_depth_img) <= 0.15
        mask_proj[mask_proj] = mask_visible
        point_coords_img = point_coords_img[mask_visible]
        return mask_proj, point_coords_img


    def proj_prepare(self):
        'prepare the project masks and project coordinates for each image of the scene'
        rgb_dir = os.path.join(self.image_path, 'color')
        image_info = []
        for img_id in range(len(os.listdir(rgb_dir))):
            mask_proj, point_coords_img = self.view_proj(img_id)
            mask_proj = rle_encode(mask_proj)
            info = (mask_proj, point_coords_img)
            image_info.append(info)
        torch.save(image_info, self.image_proj_path)
        print('Image project info saved to {}'.format(self.image_proj_path))


    # ----------------- 2. do the view selection -----------------
    class Instance_proj_info():
        def __init__(self, img_ids, visible_points, visible_point_coords) -> None:
            self.img_ids = img_ids     # list of image ids 
            self.visible_points = visible_points     # visible point ids 
            self.visible_point_coords = visible_point_coords  # instance project 2d coordinates
            self.color_imgs = []

        def add_color_imgs(self, img):
            self.color_imgs.append(img)


    def get_visible_points(self, instance_mask, mask_proj, point_coords_img):
        mask_proj = rle_decode(mask_proj)
        mask_proj_ins = instance_mask[mask_proj]  # mask_proj_ins is the mask of instance in the view 
        point_coords_img = point_coords_img[mask_proj_ins]
        visible_point_ids = np.arange(len(instance_mask))[instance_mask & mask_proj] # visible point ids
        return visible_point_ids, point_coords_img
    

    def greedy_selection(self):
        inst_img_infos = dict()
        image_infos = torch.load(self.image_proj_path)
        for k, v in self.ret_ins.items():
            if (not self.consider_bg_obj) and self.b_sems[k]==-100:
                continue
            else:
                # print('view selection for instance:', k)
                img_point_ids = []
                img_point_num = []
                img_point_loc = []
                # get instance mask
                inst_mask = np.isin(self.superpoints, v)
                # project intance_mask to each image
                for info in image_infos:
                    visible_points, point_coords_img = self.get_visible_points(inst_mask, *info)
                    visible_points = set(visible_points)
                    img_point_ids.append(visible_points)
                    img_point_num.append(len(visible_points))
                    img_point_loc.append(point_coords_img)
                img_point_ids = np.array(img_point_ids)
                img_point_num = np.array(img_point_num)
                # greedy view selection
                selected_img_ids = []
                total_ins_pts_num = np.count_nonzero(inst_mask)
                covered_ins_pts_num = 0
                img_point_ids_ = img_point_ids.copy()
                img_num = 0 
                while covered_ins_pts_num < self.greedy_threshold*total_ins_pts_num and img_num < 10:
                    max_img_id = np.argmax(img_point_num)
                    selected_img_ids.append(max_img_id)
                    covered_points = img_point_ids_[max_img_id]
                    covered_ins_pts_num += len(covered_points)
                    vec_set_difference = np.vectorize(lambda set_i: set_i - covered_points)
                    img_point_ids_ = vec_set_difference(img_point_ids_)
                    vec_set_num = np.vectorize(lambda set_i : len(set_i))
                    img_point_num = vec_set_num(img_point_ids_)
                    img_num += 1
                img_ids = np.array(selected_img_ids)
                if len(img_ids) > 0 :
                    img_point_ids = img_point_ids[img_ids]
                else:
                    img_point_ids = np.array([])
                img_point_loc2 = []
                for j in img_ids:
                    img_point_loc2.append(img_point_loc[j])
                inst_img_infos[k] = self.Instance_proj_info(img_ids, img_point_ids, img_point_loc2) 
        return inst_img_infos


    def view_select(self):
        'save the view selection results for the scene'
        inst_img_infos = self.greedy_selection()
        for k,v in inst_img_infos.items():
            for img_id in v.img_ids:
                color_img_path = os.path.join(self.image_path, 'color', str(img_id)+'.jpg')
                color_img = cv2.imread(color_img_path)
                color_img = cv2.resize(color_img, (640, 480))
                v.add_color_imgs(color_img)
        torch.save(inst_img_infos, self.select_res_path)
        print('Image selection info saved to {}'.format(self.select_res_path))


    # ----------------- 3. do the label assignment -----------------
    def get_box_prompt(self, visible_point_coords):
        if len(visible_point_coords) == 0:
            return None 
        box_max = np.max(visible_point_coords, 0)
        box_min = np.min(visible_point_coords, 0)
        if self.shrink_box:
            bound = box_max-box_min
            box_max[1] = box_max[1] - 0.1*(bound[1])
            # box_min[1] = box_min[1] + 0.05*(bound[1])
            box_max = np.round(box_max).astype(int)
            # box_min = np.round(box_min).astype(int)
        box = np.concatenate([box_min, box_max], axis=0)
        box = np.expand_dims(box, axis=0)
        return box 

    
    def get_bg_prompt(self, visible_point_coords, all_proj_points):
        img_M = np.zeros((self.img_w, self.img_h), dtype=bool)
        img_M[visible_point_coords[:,0], visible_point_coords[:,1]] = True 
        windows = view_as_windows(img_M, (self.bg_w, self.bg_w), step=self.bg_w)   
        patch_M = np.any(windows, axis=(-1,-2))
        kernel = np.array([[0, 1, 0],
                           [1, 0, 1],
                           [0, 1, 0]])
        conv = signal.convolve2d(patch_M, kernel, mode='same')
        if self.only_scanned_bg:
            img_V = np.zeros((self.img_w, self.img_h), dtype=bool)
            img_V[all_proj_points[:,0], all_proj_points[:,1]] = True 
            windows = view_as_windows(img_V, (self.bg_w, self.bg_w), step=self.bg_w)   
            patch_V = np.any(windows, axis=(-1,-2))
            indices = np.argwhere((patch_M == False) & (conv > 0) & (patch_V == True))
        else:
            indices = np.argwhere((patch_M == False) & (conv > 0))
        bg_ps =  indices * self.bg_w + 0.5 * self.bg_w
        bg_ps = bg_ps.astype(int)
        return bg_ps


    def sam_predict(self, img, prompt, prompt_type):
        predictor = self.predictor
        predictor.set_image(img)
        if prompt_type==0: # 0 means box prompt
            masks, _, _ = predictor.predict(box=prompt, multimask_output=False, return_logits=True)
            mask = masks[0]
            return np.exp(mask)/(1+np.exp(mask))
        # 1 means point prompt
        if len(prompt) == 0:
            return np.zeros((img.shape[0], img.shape[1]))
        Masks = np.zeros((len(prompt), img.shape[0], img.shape[1]))
        for i, p in enumerate(prompt):
            pt = np.expand_dims(p, axis=0)
            prompt_label = np.array([1])
            masks, _, _ = predictor.predict(point_coords=pt, point_labels=prompt_label, multimask_output=False, return_logits=True)
            Masks[i] = masks[0]
        mask = np.max(Masks, axis=0)
        return np.exp(mask)/(1+np.exp(mask))

    def sam_predict_merge(self, img, box_prompt, bg_prompt):
        predictor = self.predictor
        predictor.set_image(img)
        if len(bg_prompt) == 0:
            return self.sam_predict(img, box_prompt, 0)
        prompt_label = np.zeros(len(bg_prompt))
        masks, _,_ = predictor.predict(box=box_prompt, point_coords= bg_prompt, point_labels=prompt_label,  multimask_output=False, return_logits=True)
        mask = masks[0]
        return np.exp(mask)/(1+np.exp(mask))


    def get_sam_mask(self, visible_point_coords, all_proj_points, color_img):
        box_prompt = self.get_box_prompt(visible_point_coords)
        bg_prompt = self.get_bg_prompt(visible_point_coords, all_proj_points) 
        if not self.prompt_combine:
            if box_prompt is None:
                sam_mask_f = np.zeros((self.img_h, self.img_w))
            else:
                sam_mask_f = self.sam_predict(color_img, box_prompt, 0) #/////
            sam_mask_b = self.sam_predict(color_img, bg_prompt, 1) #/////
            sam_mask = sam_mask_f - self.beta * sam_mask_b
            # for visualisation purpose
            if self.vis_2dmasks:
                # foreground 
                see_points_image(visible_point_coords, color_img)
                see_box_img(box_prompt[0], color_img)
                see_heatmap_image(sam_mask_f, color_img, type=0)
                # background
                see_points_image(bg_prompt, color_img)
                see_heatmap_image(sam_mask_b, color_img, type=1)
        else:
            sam_mask = self.sam_predict_merge(color_img, box_prompt, bg_prompt)
            # for visualisation purpose
            if self.vis_2dmasks:
                # foreground 
                see_points_image(visible_point_coords, color_img)
                see_box_img(box_prompt[0], color_img)
                see_points_image(bg_prompt, color_img)
                see_heatmap_image(sam_mask, color_img, type=1)
        return sam_mask
    

    def conf_assign(self):
        inst_p_ids = dict()
        inst_p_confs = dict()
        inst_p_appear = dict()
        ins_project_infos = torch.load(self.select_res_path)
        if self.only_scanned_bg:
            image_project_infos = torch.load(self.image_proj_path)
        # initialise
        for k, v in self.ret_ins.items():
            if (not self.consider_bg_obj) and self.b_sems[k]==-100:
                continue
            else:
                inst_ids = np.arange(len(self.superpoints))[np.isin(self.superpoints, v)]
                inst_p_ids[k] = inst_ids
                inst_p_confs[k] = np.zeros(len(inst_ids))
                inst_p_appear[k] = np.zeros(len(inst_ids)) + 0.000001
        # update the confidence
        for k in inst_p_ids.keys():
            # print('confidence assign for instance : ', k)
            for i in range(len(ins_project_infos[k].img_ids)):
                visible_points = ins_project_infos[k].visible_points[i]
                visible_point_coords = ins_project_infos[k].visible_point_coords[i]
                color_img = ins_project_infos[k].color_imgs[i]
                new_conf = np.zeros(len(visible_points))
                # assign confidence with foreground & background prompts
                if self.only_scanned_bg:
                    img_id = ins_project_infos[k].img_ids[i]
                    all_proj_points = image_project_infos[img_id][1]
                sam_mask = self.get_sam_mask(visible_point_coords, all_proj_points, color_img)
                active_ps = sam_mask[visible_point_coords[:,1], visible_point_coords[:,0]]
                new_conf += active_ps
                # update confidence of the instance
                visible_mask = np.isin(inst_p_ids[k], np.array(list(visible_points))) 
                inst_p_confs[k][visible_mask] += new_conf
                inst_p_appear[k][visible_mask] += 1
            inst_p_confs[k] = inst_p_confs[k]/inst_p_appear[k]
        return inst_p_confs


    def superpoint_corr(self, inst_p_confs):
        print('start superpoint correlation ... ')
        ret_ins_new = dict()
        sp_num = np.max(self.superpoints)+1
        inst_num = len(self.ret_ins.keys())
        matrix = np.zeros((inst_num, sp_num))
        # smooth
        for k, v in self.ret_ins.items():
            if (not self.consider_bg_obj) and self.b_sems[k]==-100:
                continue
            else:
                inst_mask = np.isin(self.superpoints, v)
                mean_values = np.array([np.mean(inst_p_confs[k][self.superpoints[inst_mask] == sp]) for sp in v])
                acitve_mask = mean_values >= 0
                ret_ins_new[k] = v[acitve_mask]
                if len(acitve_mask) > 0:
                    matrix[k][ret_ins_new[k]] = mean_values[acitve_mask]
        # unique assign
        indexes = np.argmax(matrix, axis=0) # len of sp_num, which sp belong to which instance
        for k, v in self.ret_ins.items():
            if (not self.consider_bg_obj) and self.b_sems[k]==-100:
                continue
            else:
                all_sps = np.arange(sp_num)
                ins_sps_mask = np.isin(all_sps, ret_ins_new[k])
                max_sps_mask = indexes == k
                ret_ins_new[k] = all_sps[ins_sps_mask & max_sps_mask]                
        return ret_ins_new


    def label_assign(self):
        inst_p_confs = self.conf_assign()
        ret_ins_new = self.superpoint_corr(inst_p_confs)
        torch.save(ret_ins_new, self.save_path)
        print('The label assignment result has been saved to {}'.format(self.save_path))
        return ret_ins_new


    # ----------------- 4. visualise and evaluation -----------------
    def test_acc(self, ret):
        # get the wrong number of superpoints and points 
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
    

    def vis_inst_label(self, ret):
        vis = TerryVis(self.coords, self.colors) 
        for k, v in ret.items():
            if k==-100:
                continue
            if self.b_sems[k] != -100:
                print('instance id : ', k)
                mask = np.isin(self.superpoints, v)
                vis.see_mask(mask)
    

    def compare_with_true(self, ret):
        vis = TerryVis(self.coords, self.colors) 
        for k, v in ret.items():
            if k==-100:
                continue
            if self.b_sems[k] != -100:
                print('instance id : ', k)
                mask = np.isin(self.superpoints, v)
                vis.see_mask(mask)
                mask = np.isin(self.superpoints, self.ret_true[k])
                vis.see_mask(mask, color=np.array([0,1,0]))


    def write_log(self, acc_result):
        with open(self.log, 'a') as f:
            f.write('\n')
            f.write(self.scan_id + ':\n')
            f.write(acc_result)


    # ----------------- 5. main process function -----------------

    def process_step1(self):
        # This part is running on CPUs
        # First step: load the image projection information 
        if not os.path.exists(self.image_proj_path):
            print('Image project info loading ...')
            self.proj_prepare()
        # Second step: do the view selection
        if not os.path.exists(self.select_res_path):
            print('selection info loading ...')
            self.view_select()

    
    def process_step2(self):
        # This part is running on GPUs and CPUs
        ret_ins_new = self.label_assign()

        if os.path.exists(self.select_res_path):
            os.remove(self.select_res_path)
        if self.vis_final:
            self.compare_with_true(ret_ins_new)
        if self.test_final_acc:
            result = ''
            # acc = self.test_acc(self.ret_ins)
            # result = result + 'old accuracy : ' + str(acc[0]) + ' ' + str(acc[1]) + '\n' 
            acc_new = self.test_acc(ret_ins_new)
            result = result + 'new accuracy : ' + str(acc_new[0]) + ' ' + str(acc_new[1]) + '\n'
            acc_base = self.test_acc(self.ret_baseline)
            result = result + 'baseline accuracy :' + str(acc_base[0]) + ' ' + str(acc_base[1]) + '\n'
            if self.print_final_acc:
                print(result)
            self.write_log(result)  
            return acc_new, acc_base
        return None 

    def process(self):
        # First step: load the image projection information 
        if not os.path.exists(self.image_proj_path):
            print('Image project info loading ...')
            self.proj_prepare()
        # Second step: do the view selection
        if not os.path.exists(self.select_res_path):
            print('selection info loading ...')
            self.view_select()
        # Third step: do the label assignment
        if not os.path.exists(self.save_path):
            print('label optimization running for ', self.scan_id)
            ret_ins_new = self.label_assign()
        else: 
            print(self.scan_id, 'has been corrected before')
            ret_ins_new = torch.load(self.save_path)
        # delete some files for the space efficiency 
        if os.path.exists(self.select_res_path):
            os.remove(self.select_res_path)
    
        if self.vis_final:
            self.compare_with_true(ret_ins_new)
        if self.test_final_acc:
            result = ''
            # acc = self.test_acc(self.ret_ins)
            # result = result + 'old accuracy : ' + str(acc[0]) + ' ' + str(acc[1]) + '\n' 
            acc_new = self.test_acc(ret_ins_new)
            result = result + 'new accuracy : ' + str(acc_new[0]) + ' ' + str(acc_new[1]) + '\n'
            acc_base = self.test_acc(self.ret_baseline)
            result = result + 'baseline accuracy :' + str(acc_base[0]) + ' ' + str(acc_base[1]) + '\n'
            if self.print_final_acc:
                print(result)
            self.write_log(result)  
            return acc_new, acc_base
        
        return None 


    def check_result(self):      
        if not os.path.exists(self.save_path):
            print('The result path for {} haven\'t loaded yet'.format(self.scan_id))
            raise ValueError
        else:
            print('checking {}'.format(self.scan_id))
            ret = torch.load(self.save_path)
            self.compare_with_true(self.ret_baseline)
            # acc = self.test_acc(ret)
            # print('accuracy : ', acc[0], acc[1])

 




