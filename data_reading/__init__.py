import sys

from data_reading.classify_data import *
from data_reading.tagging_read import *


processors = {
      "cola": ColaProcessor,
      "mnli": MnliProcessor,
      "mrpc": MrpcProcessor,
      "ads": MrpcProcessor,
      "qic":QICProcessor,
      "qicm":QICProcessorMovie,
      "ner":TaggingProcessor,
  }
