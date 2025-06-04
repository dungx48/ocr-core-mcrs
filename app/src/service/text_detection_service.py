import os
import time

import cv2
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from collections import OrderedDict

from src.model.craft import CRAFT

class TextDetectionService():
    def __init__(self):
        self.cuda = True
        self.trained_model = ''
        pass

    def text_detection_by_craft(self):
        net = CRAFT()     # initialize

        # print('Loading weights from checkpoint (' + args.trained_model + ')')
        if self.cuda:
            net.load_state_dict(self.copyStateDict(torch.load(self.trained_model)))
        else:
            net.load_state_dict(self.copyStateDict(torch.load(self.trained_model, map_location='cpu')))

        if self.cuda:
            net = net.cuda()
            net = torch.nn.DataParallel(net)
            cudnn.benchmark = False

        net.eval()

        # LinkRefiner
        refine_net = None
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
        for k, image_path in enumerate(image_list):
            print("Test image {:d}/{:d}: {:s}".format(k+1, len(image_list), image_path), end='\r')
            image = imgproc.loadImage(image_path)

            bboxes, polys, score_text = test_net(net, image, args.text_threshold, args.link_threshold, args.low_text, args.cuda, args.poly, refine_net)

            # save score text
            filename, file_ext = os.path.splitext(os.path.basename(image_path))
            mask_file = result_folder + "/res_" + filename + '_mask.jpg'
            cv2.imwrite(mask_file, score_text)

            file_utils.saveResult(image_path, image[:,:,::-1], polys, dirname=result_folder)

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