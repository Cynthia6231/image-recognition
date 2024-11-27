# -*- coding: utf-8 -*-
# @Author  : WZS
# @Software: PyCharm
from net import resnet, Alexnet
def choose_model(name, num_class, pretrained, progress):
    # resnet
    if name == "resnet18":
        model = resnet.resnet18(num_class=num_class, pretrained=pretrained, progress=progress)
        return model
    if name == "resnet34":
        model = resnet.resnet34(num_class=num_class, pretrained=pretrained, progress=progress)
        return model
    if name == "resnet50":
        model = resnet.resnet50(num_class=num_class, pretrained=pretrained, progress=progress)
        return model
    if name == "resnet101":
        model = resnet.resnet101(num_class=num_class, pretrained=pretrained, progress=progress)
        return model
    if name == "resnet152":
        model = resnet.resnet152(num_class=num_class, pretrained=pretrained, progress=progress)
        return model

    # Alexnet
    if name == "Alexnet":
        model = Alexnet.alexnet(num_class=num_class, pretrained=pretrained, progress=progress)
        return model

