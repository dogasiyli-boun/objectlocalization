import tutorial01 as t01
from helperFuncs import is_cuda_available, check_and_return_device
import localization_example_helper_funcs as t01_hf
import torch
import cv2
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign
from torchvision.models.detection import MaskRCNN

# x = [torch.rand(3, 300, 400).to(device), torch.rand(3, 500, 400).to(device)]

def im2model(img_path, device):
    img = cv2.imread(img_path)
    img = t01_hf.preprocess(img)
    img = torch.from_numpy(img).permute((0, 3, 1, 2)).float().to(device)
    return torch.squeeze(img)

def run_01():
    is_cuda_available()
    device = check_and_return_device()
    labels, boxes, img_list = t01.load_data()
    dataloader, valdataloader = t01.preprocess_data(img_list, boxes, labels)
    model1, acc_list1 = t01_hf.train(dataloader, valdataloader, model=None, num_of_epochs=20, start_from_scratch=True)
    return model1, device

def run_02(device):
    rand_img_path1 = t01_hf.get_random_img()
    rand_img_path2 = t01_hf.get_random_img()
    img1 = im2model(rand_img_path1, device)
    img2 = im2model(rand_img_path2, device)
    print(torch.Tensor.size(img1))
    return img1, img2

def run_03(backbone):
    anchor_generator = AnchorGenerator(sizes=((32, 64, 128),),
                                       aspect_ratios=((0.5, 1.0, 2.0),))
    roi_pooler = MultiScaleRoIAlign(featmap_names=['0'], output_size=7, sampling_ratio=2)
    mask_roi_pooler = MultiScaleRoIAlign(featmap_names=['0'], output_size=14, sampling_ratio=2)
    model_rc = MaskRCNN(backbone, min_size=256,
                     num_classes=2, rpn_anchor_generator=anchor_generator,
                     box_roi_pool=roi_pooler, mask_roi_pool=mask_roi_pooler)
    return model_rc

def run_v02_01():
    device = check_and_return_device()
    model_sample = t01_hf.SampleNetwork()
    return model_sample, device

def main():
    model1, device = run_01()
    img1, img2 = run_02(device)
    backbone = model1.class_fc2
    backbone.out_channels = 120
    model_rc = run_03(backbone)
    model_rc.eval()
    predictions = model_rc([img1, img2])