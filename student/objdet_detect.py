# ---------------------------------------------------------------------
# Project "Track 3D-Objects Over Time"
# Copyright (C) 2020, Dr. Antje Muntzinger / Dr. Andreas Haja.
#
# Purpose of this file : Detect 3D objects in lidar point clouds using deep learning
#
# You should have received a copy of the Udacity license together with this program.
#
# https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013
# ----------------------------------------------------------------------
#

# general package imports
import numpy as np
import torch
import torch.nn.functional as F
from easydict import EasyDict as edict
import matplotlib.pyplot as plt
import pandas as pd
# add project directory to python path to enable relative imports
import os
import sys
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

# model-related
from tools.objdet_models.resnet.models import fpn_resnet
from tools.objdet_models.resnet.utils.evaluation_utils import decode, post_processing 

from tools.objdet_models.darknet.models.darknet2pytorch import Darknet as darknet
from tools.objdet_models.darknet.utils.evaluation_utils import post_processing_v2


# load model-related parameters into an edict
def load_configs_model(model_name='darknet', configs=None):

    # init config file, if none has been passed
    if configs==None:
        configs = edict()  

    # get parent directory of this file to enable relative paths
    curr_path = os.path.dirname(os.path.realpath(__file__))
    parent_path = configs.model_path = os.path.abspath(os.path.join(curr_path, os.pardir))    
    configs.min_iou = 0.6
    
    # set parameters according to model type
    if model_name == "darknet":
        configs.model_path = os.path.join(parent_path, 'tools', 'objdet_models', 'darknet')
        configs.pretrained_filename = os.path.join(configs.model_path, 'pretrained', 'complex_yolov4_mse_loss.pth')
        configs.arch = 'darknet'
        configs.batch_size = 4
        configs.cfgfile = os.path.join(configs.model_path, 'config', 'complex_yolov4.cfg')
        configs.conf_thresh = 0.5
        configs.distributed = False
        configs.img_size = 608
        configs.nms_thresh = 0.4
        configs.num_samples = None
        configs.num_workers = 4
        configs.pin_memory = True
        configs.use_giou_loss = False
    elif model_name == 'fpn_resnet':
        ####### ID_S3_EX1-3 START #######     
        #######
        configs.model_path = os.path.join(parent_path, 'tools', 'objdet_models', 'resnet')
        configs.pretrained_filename = os.path.join(configs.model_path, 'pretrained', 'fpn_resnet_18_epoch_300.pth')
        configs.arch = 'fpn_resnet'
        configs.num_layers = 18
        configs.batch_size = 4
        configs.K = 50 #number of top K
        configs.conf_thresh = 0.5
        configs.num_samples = None
        configs.num_workers = 4
        configs.pin_memory = True
        configs.distributed = False
        configs.input_size = (608, 608)
        configs.hm_size = (152, 152)
        configs.down_ratio = 4
        configs.max_objects = 50
        configs.imagenet_pretrained = False
        configs.head_conv = 64
        configs.num_classes = 3
        configs.num_center_offset = 2
        configs.num_z = 1
        configs.num_dim = 3
        configs.num_direction = 2  # sin, cos
        configs.heads = {
                'hm_cen': configs.num_classes,
                'cen_offset': configs.num_center_offset,
                'direction': configs.num_direction,
                'z_coor': configs.num_z,
                'dim': configs.num_dim
            }
        configs.num_input_features = 4

        #######
        ####### ID_S3_EX1-3 END #######     

    else:
        raise ValueError("Error: Invalid model name")

    # GPU vs. CPU
    configs.no_cuda = False # if true, cuda is not used
    configs.gpu_idx = 0  # GPU index to use.
    configs.device = torch.device('cpu' if configs.no_cuda else 'cuda:{}'.format(configs.gpu_idx))

    return configs


# load all object-detection parameters into an edict
def load_configs(model_name='fpn_resnet', configs=None):

    # init config file, if none has been passed
    if configs==None:
        configs = edict()    

    # birds-eye view (bev) parameters
    configs.lim_x = [0, 50] # detection range in m
    configs.lim_y = [-25, 25]
    configs.lim_z = [-1, 3]
    configs.lim_r = [0, 1.0] # reflected lidar intensity
    configs.bev_width = 608  # pixel resolution of bev image
    configs.bev_height = 608 

    # add model-dependent parameters
    configs = load_configs_model(model_name, configs)

    # visualization parameters
    configs.output_width = 608 # width of result image (height may vary)
    configs.obj_colors = [[0, 255, 255], [0, 0, 255], [255, 0, 0]] # 'Pedestrian': 0, 'Car': 1, 'Cyclist': 2

    return configs


# create model according to selected model type
def create_model(configs):

    print(configs.pretrained_filename)
    # check for availability of model file
    assert os.path.isfile(configs.pretrained_filename), "No file at {}".format(configs.pretrained_filename)

    # create model depending on architecture name
    if (configs.arch == 'darknet') and (configs.cfgfile is not None):
        print('using darknet')
        model = darknet(cfgfile=configs.cfgfile, use_giou_loss=configs.use_giou_loss)    
    
    elif 'fpn_resnet' in configs.arch:
        print('using ResNet architecture with feature pyramid')
        
        ####### ID_S3_EX1-4 START #######     
        #######
        num_layers = configs.num_layers
        model = fpn_resnet.get_pose_net(num_layers=num_layers, heads=configs.heads, head_conv=configs.head_conv,
                                        imagenet_pretrained=configs.imagenet_pretrained)

        #######
        ####### ID_S3_EX1-4 END #######     
    
    else:
        assert False, 'Undefined model backbone'

    # load model weights
    model.load_state_dict(torch.load(configs.pretrained_filename, map_location='cpu'))
    print('Loaded weights from {}\n'.format(configs.pretrained_filename))

    # set model to evaluation state
    configs.device = torch.device('cpu' if configs.no_cuda else 'cuda:{}'.format(configs.gpu_idx))
    model = model.to(device=configs.device)  # load model to either cpu or gpu
    model.eval()          

    return model


# detect trained objects in birds-eye view
def detect_objects(input_bev_maps, model, configs):

    # deactivate autograd engine during test to reduce memory usage and speed up computations
    with torch.no_grad():  

        # perform inference
        
        outputs = model(input_bev_maps)

        # decode model output into target object format
        if 'darknet' in configs.arch:

            # perform post-processing
            output_post = post_processing_v2(outputs, conf_thresh=configs.conf_thresh, nms_thresh=configs.nms_thresh) 
            detections = []
            for sample_i in range(len(output_post)):
                if output_post[sample_i] is None:
                    continue
                detection = output_post[sample_i]
                for obj in detection:
                    x, y, w, l, im, re, _, _, _ = obj
                    yaw = np.arctan2(im, re)
                    detections.append([1, x, y, 0.0, 1.50, w, l, yaw])    

        elif 'fpn_resnet' in configs.arch:
            # decode output and perform post-processing
            
            ####### ID_S3_EX1-5 START #######     
            #######
            
            detections = []
            hm = outputs['hm_cen'].sigmoid_()
            offsets = outputs['cen_offset'].sigmoid_()
            dirs = outputs['direction']
            boxdims = outputs['dim']
            batch_size, num_classes, height, width = hm.size()

            #plt.imshow(hm[0, 2].cpu().detach().numpy())
            #plt.show()
            #nms
            hmax = F.max_pool2d(hm, (3, 3), stride = 1, padding = 1) #keep size
            keep = (hmax==hm).float()
            hm = hm * keep


            #1, 3, 50 (last idx 152*152 flattened)
            topk_scores, topk_idxs = torch.topk(hm.view(batch_size, num_classes, -1), configs.K) 
            topk_xs = (topk_idxs // width).float()
            topk_ys = (topk_idxs % width).float()

            

            topk_score, topk_idx = torch.topk(topk_scores.view(batch_size, -1), configs.K)

            #############
            # we will be returning x, y, w, l, and yaw
            # concatenated in the last dimension, so output can be
            # batch_size * K(50) * [x, y, w, l, yaw]
            # we will add extra dimension to each of features to torch.cat them along the last dimension.
            #############

            #  for 50 top_top.. from which class(0~2)
            topk_classes = topk_idx//configs.K
            topk_classes = topk_classes.unsqueeze(2).float()
            
            #topk_score = topk_score.unsqueeze(2)

            topk_idxs = torch.gather(topk_idxs.view(batch_size, -1), 1, topk_idx).view(batch_size, configs.K)
            topk_xs = torch.gather(topk_xs.view(batch_size, -1), 1, topk_idx).view(batch_size, configs.K)
            topk_ys = torch.gather(topk_ys.view(batch_size, -1), 1, topk_idx).view(batch_size, configs.K)
            
            #heatmap = np.zeros((152, 152))
            #yy = topk_ys[0]
            #xx = topk_xs[0]
            #coords = torch.stack((yy,xx), dim=1).cpu().detach().numpy().astype(int)
            #for y,x in coords:
            #    heatmap[y, x] = 255
            #plt.imshow(heatmap)
            #plt.show()


            #offsets have dx and dy values in fractional pixels
            offsets = offsets.permute(0, 2, 3, 1).contiguous()
            topk_offsets = torch.gather(
                offsets.view(offsets.size(0), -1, offsets.size(3)), 
                1, topk_idxs.unsqueeze(2).expand(-1, -1, offsets.size(3))
            ).view(batch_size, configs.K, 2)

            #directions have real(cos) and im(sine) componenets per each pixel
            dirs = dirs.permute(0, 2, 3, 1).contiguous()
            topk_dirs = torch.gather(
                dirs.view(dirs.size(0), -1, dirs.size(3)),
                1, topk_idxs.unsqueeze(2).expand(-1, -1, dirs.size(3))
            )

            #atan2: in y,x order, radians, which is EQUAL to numpy.atan2
            topk_yaws = torch.atan2(topk_dirs[:,:,0], topk_dirs[:,:,1]).unsqueeze(2)

            #dimensions have x, y, z(means up) components per each pixel
            boxdims = boxdims.permute(0, 2, 3, 1).contiguous()
            topk_boxdims = torch.gather( #
                boxdims.view(boxdims.size(0), -1, boxdims.size(3)),
                1, topk_idxs.unsqueeze(2).expand(-1, -1, boxdims.size(3))
            )

            bound_size_x = configs.lim_x[1] - configs.lim_x[0] #50
            bound_size_y = configs.lim_y[1] - configs.lim_y[0] #50

            # apply offsets - should be x<-row, y<-col as well?
            topk_xs = topk_xs.unsqueeze(2) + topk_offsets[:, :, 0:1]
            topk_ys = topk_ys.unsqueeze(2) + topk_offsets[:, :, 1:2]

            # FIXED lim_x <-> lim_y since x is forward...
            topk_xs = configs.lim_x[0] + topk_xs * configs.down_ratio / configs.bev_height * bound_size_x 
            topk_ys = configs.lim_y[0] + topk_ys * configs.down_ratio / configs.bev_width * bound_size_y
            
            
            # required output format ([1, x, y, 0.0, 1.50, w, l, yaw])  
            c0 = torch.ones([batch_size, configs.K, 1], device=configs.device)
            c3 = torch.zeros([batch_size, configs.K, 1], device=configs.device)
            c4 = torch.ones([batch_size, configs.K, 1], device=configs.device) * 1.50


            detections = torch.cat([c0, topk_xs, topk_ys, c3, c4, topk_boxdims[:, :, 1:3], topk_yaws], dim=2)
            detections = detections[0]
            keep_indices = topk_score[0] > 0.2
            detections = detections[keep_indices, :]
            #df = pd.DataFrame({f"c{i}" : detections[:,i].detach().cpu().numpy() for i in range(detections.shape[-1])})

            print("end of detection phase")
            detection = detections.cpu().detach().numpy()


            print(detection.shape)
            return detection
            #
            #######
            ####### ID_S3_EX1-5 END #######     

            

    ####### ID_S3_EX2 START #######     
    #######
    # Extract 3d bounding boxes from model response
    print("student task ID_S3_EX2")
    objects = [] 

    ## step 1 : check whether there are any detections
    if len(detections)>0:
        bound_size_x = configs.lim_x[1] - configs.lim_x[0] #50
        bound_size_y = configs.lim_y[1] - configs.lim_y[0] #50
        ## step 2 : loop over all detections
        for detection in detections:
            ## step 3 : perform the conversion using the limits for x, y and z set in the configs structure
            id, x, y, z, h, w, l, yaw = detection
            xx = configs.lim_x[0] + y.item()/configs.img_size * bound_size_x
            yy = configs.lim_y[0] + x.item()/configs.img_size * bound_size_y
            w = w /configs.img_size * bound_size_y
            l = l /configs.img_size * bound_size_x
            ## step 4 : append the current object to the 'objects' array
            objects.append([id, xx, yy, z, h, w.item(), l.item(), yaw.item()])
    #######
    ####### ID_S3_EX2 START #######   
    
    return objects    

