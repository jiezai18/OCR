

import copy
import paddle
import paddle.nn as nn

# det loss
from .det_db_loss import DBLoss
from .det_east_loss import EASTLoss
from .det_sast_loss import SASTLoss

# rec loss
from .rec_ctc_loss import CTCLoss
from .rec_att_loss import AttentionLoss
from .rec_srn_loss import SRNLoss

# cls loss
from .cls_loss import ClsLoss

# e2e loss
from .e2e_pg_loss import PGLoss

# basic loss function
from .basic_loss import DistanceLoss

# combined loss function
from .combined_loss import CombinedLoss

# table loss
from .table_att_loss import TableAttentionLoss

def build_loss(config):
    support_dict = [
        'DBLoss', 'EASTLoss', 'SASTLoss', 'CTCLoss', 'ClsLoss', 'AttentionLoss',
        'SRNLoss', 'PGLoss', 'CombinedLoss', 'TableAttentionLoss'
    ]
    config = copy.deepcopy(config)
    module_name = config.pop('name')
    assert module_name in support_dict, Exception('loss only support {}'.format(
        support_dict))
    module_class = eval(module_name)(**config)
    return module_class
