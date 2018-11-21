from model_pools.classify_model import *
from model_pools.tagging_model import *
from model_pools.tagging_classify_multi_task_model import TaggingMultitaskModel
from model_pools.match_lstm_model import MatchLSTMModel
from model_pools.multi_gpu_classify import MultiClassifyModel

model_pools = {
    "classify_model":ClassifyModel,
    "tagging_model":TaggingModel,
    "tagging_classify_model":TaggingMultitaskModel,
    "classify_matchrnn":MatchLSTMModel,
    "multi_gpu_classify":MultiClassifyModel,
}