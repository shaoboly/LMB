import sys

from data_reading.classify_data import *
from data_reading.tagging_read import *
from data_reading.match_rnn_combine_read import RnnMatchCombine


processors = {
      "cola": ColaProcessor,
      "mnli": MnliProcessor,
      "mrpc": MrpcProcessor,
      "ads": MrpcProcessor,
      "qic":QICProcessor,
      "qicm":QICProcessorMovie,
      "qimatch":QIMatch,
      "ner":TaggingProcessor,
      "matchcombine":RnnMatchCombine
  }
