import os
import time

import cv2
import torch
import torch.backends.cudnn as cudnn
from collections import OrderedDict

from src.utils import imgproc, file_utils
from src.model.detect_text.craft import CRAFT
from src.model.detect_text import craft_test_net

class TextDetectionService():
    def __init__(self, trained_model=None, image_list='', result_folder='./result/'):
        self.cuda = torch.cuda.is_available()
        self.refine_net = None
        self.text_threshold = 0.7
        self.link_threshold = 0.4
        self.low_text = 0.4
        self.poly = False
        # cuda
        if trained_model is None:
            self.trained_model = 'weights/craft_mlt_25k.pth'
        else:
            self.trained_model = trained_model
        # result_folder
        if not os.path.isdir(result_folder):
            os.mkdir(result_folder)
            self.result_folder = result_folder
        # image_list
        self.image_list = image_list

    def text_detection_by_craft(self):
        net = CRAFT()     # initialize

        print('Loading weights from checkpoint (' + self.trained_model + ')')
        if self.cuda:
            checkpoint = torch.load(self.trained_model)
            net.load_state_dict(self.copyStateDict(checkpoint))
            net = net.cuda()
            net = torch.nn.DataParallel(net)
            cudnn.benchmark = False
        else:
            net.load_state_dict(self.copyStateDict(torch.load(self.trained_model, map_location='cpu')))

        net.eval()

        # LinkRefiner
        # refine_net = None
        # if args.refine:
        #     from refinenet import RefineNet
        #     refine_net = RefineNet()
        #     print('Loading weights of refiner from checkpoint (' + args.refiner_model + ')')
        #     if args.cuda:
        #         refine_net.load_state_dict(copyStateDict(torch.load(args.refiner_model)))
        #         refine_net = refine_net.cuda()
        #         refine_net = torch.nn.DataParallel(refine_net)
        #     else:
        #         refine_net.load_state_dict(copyStateDict(torch.load(args.refiner_model, map_location='cpu')))

        #     refine_net.eval()
        #     args.poly = True

        t = time.time()

        # load data
        for k, image_path in enumerate(self.image_list):
            print("Test image {:d}/{:d}: {:s}".format(k+1, len(self.image_list), image_path), end='\r')
            image = imgproc.loadImage(image_path)

            bboxes, polys, score_text = craft_test_net(net, image, self.text_threshold, self.link_threshold, self.low_text, self.cuda, self.poly, self.refine_net)

            # save score text
            filename, file_ext = os.path.splitext(os.path.basename(image_path))
            mask_file = self.result_folder + "/res_" + filename + '_mask.jpg'
            cv2.imwrite(mask_file, score_text)

            file_utils.saveResult(image_path, image[:,:,::-1], polys, dirname=self.result_folder)

    def copyStateDict(self, state_dict):
        if list(state_dict.keys())[0].startswith("module"):
            start_idx = 1
        else:
            start_idx = 0
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = ".".join(k.split(".")[start_idx:])
            new_state_dict[name] = v
        return new_state_dict