import os 
import numpy as np
import torch
from SensorData import SensorData
from segment_anything import SamPredictor, sam_model_registry
from main import TerryClass
import datetime
import yaml
import time
import argparse
import multiprocessing


def set_up_image(output_path, filename):
    '''
    Output the rgb images and camera pose matrixes and instrict matrixes, using the Scannet build-in method
    '''
    # print('2D data loading ...')
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        sd = SensorData(filename)
        sd.export_color_images(os.path.join(output_path, 'color'))
        sd.export_depth_images(os.path.join(output_path, 'depth'))
        sd.export_poses(os.path.join(output_path, 'pose'))
        sd.export_intrinsics(os.path.join(output_path, 'intrinsic'))


def set_up_sam(device_id):
    torch.cuda.set_device(device_id)
    print('setting up sam ...')
    checkpoint = "sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    sam = sam_model_registry[model_type](checkpoint=checkpoint)
    sam.cuda()
    return SamPredictor(sam)


def set_up_log(config):
    log_folder = 'log'
    if not os.path.exists(log_folder):
        os.mkdir(log_folder)
    log_path = os.path.join(log_folder, 'log'+str(time.time()).split('.')[0]+'.txt')
    with open(log_path, 'w') as f:
        time_step = datetime.datetime.now()
        f.write(str(time_step)+'\n')
        for k,v in config.items():
            f.write(k+': '+ str(v) +'\n')
    return log_path


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/config01.yaml', help='path to config file')
    parser.add_argument('--check', type=bool, default=False, help='True: process the data; False: check the result')
    parser.add_argument('--file', type=str, default='scannet_test.txt', help='path to the file containing the scan ids')
    parser.add_argument('--step', type=int, default=1, help='0: step1; 2: step2')
    args = parser.parse_args()
    return args


def process_mul(args):
    scan_id, log, predictor, config = args
    if scan_id[0] == '#':
        return
    scene_folder = os.path.join('ScanNetv2', 'scans', scan_id)
    sens_path = os.path.join(scene_folder, scan_id+'.sens')
    set_up_image(os.path.join('images', scan_id+'_images'), sens_path)
    T = TerryClass(scan_id, log, predictor, **config)
    acc = T.process()
    return acc 

def process_1(args):
    scan_id, log, predictor, config = args
    if scan_id[0] == '#':
        return
    scene_folder = os.path.join('ScanNetv2', 'scans', scan_id)
    sens_path = os.path.join(scene_folder, scan_id+'.sens')
    set_up_image(os.path.join('images', scan_id+'_images'), sens_path)
    T = TerryClass(scan_id, log, predictor, **config)
    try:
        T.process_step1()
    except:
        f2 = open('rest.txt', 'a')
        print('something is wrong /////', scan_id)
        f2.write(scan_id+'\n')


def process_2(args):
    scan_id, log, predictor, config = args
    if scan_id[0] == '#':
        return
    T = TerryClass(scan_id, log, predictor, **config)
    # try
    acc = T.process_step2()
    return acc
    # except:
    #     f2 = open('rest.txt', 'a')
    #     f2.write(scan_id+'\n')
    #     return np.zeros(2), np.zeros(2)


if __name__ == '__main__':
    args = get_args()
    test_scans = open(args.file).read().splitlines()
    f = open(args.config)
    config = yaml.load(f, Loader=yaml.FullLoader)
    if not args.check:
        if config['vis_2dmasks']==True:
            log = None
        else:
            log = set_up_log(config)
        num_gpus = 2
        predictors = [set_up_sam(i) for i in range(num_gpus)]
        total_wrong_sp_new = 0
        total_wrong_sp_base = 0 
        total_wrong_pt_new = 0 
        total_wrong_pt_base = 0
        multiprocessing.set_start_method('spawn')

        if args.step == 1:
            print('step1 !!!!!!!')
            max_workers = 16
            pool = multiprocessing.Pool(processes=max_workers)
            arguments = [(scan_id, log, predictors[i%num_gpus], config) for i, scan_id in enumerate(test_scans)]
            pool.map(process_1, arguments)


        # ////////
        if args.step == 2:
            print('step2 !!!!!!!')
            max_workers = 4
            pool = multiprocessing.Pool(processes=max_workers)
            arguments = [(scan_id, log, predictors[i%num_gpus], config) for i, scan_id in enumerate(test_scans)]
            results = pool.map(process_2, arguments)
            for acc in results:
                if acc:
                    acc_new, acc_base = acc
                    total_wrong_sp_base += acc_base[0]
                    total_wrong_pt_base += acc_base[1]
                    total_wrong_sp_new += acc_new[0]
                    total_wrong_pt_new += acc_new[1]
            # ////////

        # # ////////
        # max_workers = 4
        # pool = multiprocessing.Pool(processes=max_workers)
        # arguments = [(scan_id, log, predictors[i%num_gpus], config) for i, scan_id in enumerate(test_scans)]
        # results = pool.map(process_mul, arguments)
        # for acc in results:
        #     if acc:
        #         acc_new, acc_base = acc
        #         total_wrong_sp_base += acc_base[0]
        #         total_wrong_pt_base += acc_base[1]
        #         total_wrong_sp_new += acc_new[0]
        #         total_wrong_pt_new += acc_new[1]
        # # ////////
        

        # for scan_id in test_scans:    
        #     if scan_id[0] == '#':
        #         continue
        #     scene_folder = os.path.join('ScanNetv2', 'scans', scan_id)
        #     sens_path = os.path.join(scene_folder, scan_id+'.sens')
        #     set_up_image(os.path.join('images', scan_id+'_images'), sens_path)
        #     T = TerryClass(scan_id, log, predictor, **config)
        #     acc = T.process()
        #     if not (acc is None):
        #         acc_new, acc_base = acc
        #         total_wrong_sp_base += acc_base[0]
        #         total_wrong_pt_base += acc_base[1]
        #         total_wrong_sp_new += acc_new[0]
        #         total_wrong_pt_new += acc_new[1] 

        print('total_wrong_sp_base: ', total_wrong_sp_base)
        print('total_wrong_pt_base: ', total_wrong_pt_base)
        print('total_wrong_sp_new: ', total_wrong_sp_new)
        print('total_wrong_pt_new: ', total_wrong_pt_new)

        f3 = open(log, 'a')
        f3.write('total_wrong_sp_base: '+str(total_wrong_sp_base)+'\n')
        f3.write('total_wrong_pt_base: '+str(total_wrong_pt_base)+'\n')
        f3.write('total_wrong_sp_new: '+str(total_wrong_sp_new)+'\n')
        f3.write('total_wrong_pt_new: '+str(total_wrong_pt_new)+'\n')

    else:
        for scan_id in test_scans:
            if scan_id[0] == '#':
                continue
            scene_folder = os.path.join('ScanNetv2', 'scans', scan_id)
            sens_path = os.path.join(scene_folder, scan_id+'.sens')
            T = TerryClass(scan_id, **config)
            # T.check_result()
            T.check_result()