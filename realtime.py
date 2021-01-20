from __future__ import print_function
import os
import argparse
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from data import cfg
from layers.functions.prior_box import PriorBox
from utils.nms_wrapper import nms
#from utils.nms.py_cpu_nms import py_cpu_nms
import cv2
from models.faceboxes import FaceBoxes
from utils.box_utils import decode
from utils.timer import Timer
from datetime import datetime

import sys
sys.path.append('../')

from sort.sort import *
from fast_agender.main import get_trt_model, handle_faces, get_trt_model
from fast_agender.data import AgenderDataset

fastagender_model = get_trt_model()

parser = argparse.ArgumentParser(description='FaceBoxes')
parser.add_argument('-m', '--trained_model', default='weights/FaceBoxes.pth', type=str, help='Trained state_dict file path to open')
parser.add_argument('--save_folder', default='eval/', type=str, help='Dir to save results')
parser.add_argument('--cpu', action="store_true", default=False, help='Use cpu inference')
parser.add_argument('--dataset', default='PASCAL', type=str, choices=['AFW', 'PASCAL', 'FDDB'], help='dataset')
parser.add_argument('--confidence_threshold', default=0.05, type=float, help='confidence_threshold')
parser.add_argument('--top_k', default=5000, type=int, help='top_k')
parser.add_argument('--nms_threshold', default=0.3, type=float, help='nms_threshold')
parser.add_argument('--keep_top_k', default=750, type=int, help='keep_top_k')
parser.add_argument('-s', '--show_image', action="store_true", default=False, help='show detection results')
parser.add_argument('--vis_thres', default=0.5, type=float, help='visualization_threshold')
args = parser.parse_args()



def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}

device = torch.cuda.current_device()
def load_model(model, pretrained_path, load_to_cpu):
    print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model

from torch2trt import TRTModule

def get_trt_model():
    """
    Get the fast face agender model in order to cache on memory
    """
    trt_path = "weights/model_trt.pth"
    model_trt = TRTModule()
    model_trt.load_state_dict(torch.load(trt_path))
    return model_trt


from datetime import datetime

if __name__ == '__main__':
    torch.set_grad_enabled(False)
    # net and model
    # net = FaceBoxes(phase='test', size=None, num_classes=2)    # initialize detector
    # net = load_model(net, args.trained_model, args.cpu)
    # net.eval()
    net = get_trt_model()
    
    print('Finished loading model!')
    print(net)
    cudnn.benchmark = True
    device = torch.device("cpu" if args.cpu else "cuda")
    net = net.to(device)


    # save file
    if not os.path.exists(args.save_folder):
        os.makedirs(args.save_folder)
    fw = open(os.path.join(args.save_folder, args.dataset + '_dets.txt'), 'w')

    # testing dataset
    # testset_folder = os.path.join('data', args.dataset, 'images/')
    # testset_list = os.path.join('data', args.dataset, 'img_list.txt')
    # with open(testset_list, 'r') as fr:
    #     test_dataset = fr.read().split()
    # num_images = len(test_dataset)

    # testing scale
    if args.dataset == "FDDB":
        resize = 3
    elif args.dataset == "PASCAL":
        resize = 2.5
    elif args.dataset == "AFW":
        resize = 1

    _t = {'forward_pass': Timer(), 'misc': Timer()}

    cam = cv2.VideoCapture(0)

    mot_tracker = Sort()

    # testing begin
    i = 0
    while True:
        start = datetime.now()
        _start = datetime.now()
        i = i + 1
        img_name = f'{i}'
        if cv2.waitKey(1) == 27:
            break  # esc to quit
        __start = datetime.now()
        _, img = cam.read()
        __end= datetime.now()
        print("___CAPTURE____: ", (__end - __start).total_seconds())
        # do mirroring
        img_raw = cv2.flip(img, 1)
        # image_path = testset_folder + img_name + '.jpg'
        # img_raw = cv2.imread(image_path, cv2.IMREAD_COLOR)
        img = np.float32(img_raw)
        if resize != 1:
            img = cv2.resize(img, None, None, fx=resize, fy=resize, interpolation=cv2.INTER_LINEAR)
        im_height, im_width, _ = img.shape
        scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.to(device)
        # print("______________________img____________Shape__________________", img.shape)
        scale = scale.to(device)

        _t['forward_pass'].tic()
        __start = datetime.now()
        loc, conf = net(img)  # forward pass
        __end= datetime.now()
        print("___DETECTION____: ", (__end - __start).total_seconds())

        _end = datetime.now()
        print("___SHORI___0____: ", (_end - _start).total_seconds())
        _start = datetime.now()

        _t['forward_pass'].toc()
        _t['misc'].tic()
        priorbox = PriorBox(cfg, image_size=(im_height, im_width))
        priors = priorbox.forward()
        priors = priors.to(device)
        prior_data = priors.data
        boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
        boxes = boxes * scale / resize
        boxes = boxes.cpu().numpy()
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
        _end = datetime.now()
        print("___SHORI___1____: ", (_end - _start).total_seconds())
        _start = datetime.now()

        # ignore low scores
        inds = np.where(scores > args.confidence_threshold)[0]
        boxes = boxes[inds]
        scores = scores[inds]

        # keep top-K before NMS
        order = scores.argsort()[::-1][:args.top_k]
        boxes = boxes[order]
        scores = scores[order]

        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        #keep = py_cpu_nms(dets, args.nms_threshold)
        keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
        dets = dets[keep, :]

        # keep top-K faster NMS
        dets = dets[:args.keep_top_k, :]
        _t['misc'].toc()
        _end = datetime.now()
        print("___SHORI___2____: ", (_end - _start).total_seconds())
        _start = datetime.now()

        # show image
        for b in dets:
            if b[4] < args.vis_thres:
                continue

            x1, y1, x2, y2 = int(b[0]), int(b[1]), int(b[2]), int(b[3])
            # print(x1, y1, x2, y2)
            face_img = img_raw[y1:y2, x1:x2]
            # cv2.imwrite(f'res/{i}.jpg', face_img)
            # preprocessing here 
            # print("_____face_img.shape_____", face_img.shape)
            height, width, chan = face_img.shape
            if height > 30 and width > 30:
                face_img = AgenderDataset.img_np_to_tensor(face_img)
                # make tensor as batch
                face_img = torch.unsqueeze(face_img, 0)
                face_img = face_img.to(device)
                # start = datetime.now()
                __start = datetime.now()
                res = handle_faces(fastagender_model, face_img)
                __end= datetime.now()
                print("___METAINFO____: ", (__end - __start).total_seconds())
                # end = datetime.now()
                # print("TOTAL_SEC: ", (end - start).total_seconds())
                cv2.putText(img_raw, res, (x1 - 10, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 0))

            text = "{:.4f}".format(b[4])
            b = list(map(int, b))
            cv2.rectangle(img_raw, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
            cx = b[0]
            cy = b[1] + 12
            cv2.putText(img_raw, text, (cx, cy), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))
            track_bbs_ids = mot_tracker.update(dets)
            # print("_______ids_________: ", track_bbs_ids)
            for _id in track_bbs_ids:
                _cx = int(_id[0])
                _cy = int(_id[1] - 20)
                _text = str(_id[4])
                cv2.putText(img_raw, _text, (_cx, _cy), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 0))

        _end = datetime.now()
        print("___SHORI___3____: ", (_end - _start).total_seconds())
        _start = datetime.now()

        end = datetime.now()
        fps = 1000 / ((end - start).total_seconds() * 1000)
        print("FPS: ", fps)
        img_raw = cv2.resize(img_raw, (0,0), fx=2, fy=2) 
        # print("FPS ", 1000 / (end - start).total_seconds())
        cv2.imshow('my webcam', img_raw)

        # cv2.imwrite(f'res/{i}.jpg', img_raw)

    fw.close()
    # Close modal window
    cv2.destroyAllWindows()
