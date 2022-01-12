import torch
import torchvision.transforms
import torchvision.transforms as transforms




def pad_img(img, step_size=16):
    b, c, h, w = img.size()
    assert b==1

    padding_h = 0
    padding_w = 0
    if h % step_size != 0:
        padding_h = (h//step_size+1)*step_size - h
    if w % step_size != 0:
        padding_w = (w//step_size+1)*step_size - w

    padding_left = padding_w//2
    padding_right = padding_w-padding_left

    padding_top = padding_h//2
    padding_bottom = padding_h-padding_top

    img = img.view(c,h,w)
    fill = torch.mean(img).item()
    # print(torch.unique(img))
    # print(fill)
    padding_fun = torchvision.transforms.Pad(padding=(padding_left, padding_top, padding_right, padding_bottom), fill=fill)

    return padding_fun(img).view(b, c, h+padding_top+padding_bottom, w+padding_left+padding_right), padding_top, padding_left



