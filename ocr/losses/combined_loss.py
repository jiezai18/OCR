

import paddle
import paddle.nn as nn

from .distillation_loss import DistillationCTCLoss
from .distillation_loss import DistillationDMLLoss
from .distillation_loss import DistillationDistanceLoss, DistillationDBLoss, DistillationDilaDBLoss


class CombinedLoss(nn.Layer):
    """
    CombinedLoss:
        a combionation of loss function
    """

    def __init__(self, loss_config_list=None):
        super().__init__()
        self.loss_func = []
        self.loss_weight = []
        assert isinstance(loss_config_list, list), (
            'operator config should be a list')
        for config in loss_config_list:
            assert isinstance(config,
                              dict) and len(config) == 1, "yaml format error"
            name = list(config)[0]
            param = config[name]
            assert "weight" in param, "weight must be in param, but param just contains {}".format(
                param.keys())
            self.loss_weight.append(param.pop("weight"))
            self.loss_func.append(eval(name)(**param))

    def forward(self, input, batch, **kargs):
        loss_dict = {}
        loss_all = 0.
        for idx, loss_func in enumerate(self.loss_func):
            loss = loss_func(input, batch, **kargs)
            if isinstance(loss, paddle.Tensor):
                loss = {"loss_{}_{}".format(str(loss), idx): loss}
            weight = self.loss_weight[idx]
            for key in loss.keys():
                if key == "loss":
                    loss_all += loss[key] * weight
                else:
                    loss_dict["{}_{}".format(key, idx)] = loss[key]
        loss_dict["loss"] = loss_all
        return loss_dict
