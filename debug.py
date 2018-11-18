from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import config
import logging
from model_pools import modeling,model_pools
from data_reading import processors
import tensorflow as tf
from batcher import Batcher
import os
import numpy as np



def main(_):
    FLAGS = config.retype_FLAGS()
    Classify_model = model_pools["tagging_classify_model"]
    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

    task_name1 = "ner"
    task_name2 = "qicm"

    FLAGS_tagging = FLAGS._asdict()
    FLAGS_tagging["train_batch_size"] = int(FLAGS_tagging["train_batch_size"]/2)
    FLAGS_tagging["mode"] = "dev"
    FLAGS_tagging = config.generate_nametuple(FLAGS_tagging)

    FLAGS_classify = FLAGS._asdict()
    FLAGS_classify["train_batch_size"] = int(FLAGS_classify["train_batch_size"]/2)
    FLAGS_classify["mode"] = "dev"
    #pad to equal
    FLAGS_classify["train_batch_size"]+= FLAGS.train_batch_size - FLAGS_tagging.train_batch_size - FLAGS_classify["train_batch_size"]

    FLAGS_classify["train_file"] = FLAGS_classify["train_file_multi"]
    FLAGS_classify["dev_file"] = FLAGS_classify["dev_file_multi"]
    FLAGS_classify["test_file"] = FLAGS_classify["test_file_multi"]
    FLAGS_classify = config.generate_nametuple(FLAGS_classify)

    processor_tagging = processors[task_name1]()
    processor_classify = processors[task_name2]()

    tagging_batcher = Batcher(processor_tagging, FLAGS_tagging)
    classify_batcher = Batcher(processor_classify, FLAGS_classify)

    # create trainning model
    Bert_model = Classify_model(bert_config, tagging_batcher,classify_batcher, FLAGS)

    for step in range(0, Bert_model.num_train_steps):
        tagging_batch = Bert_model.tagging_batcher.next_batch()
        classify_batch = Bert_model.classify_batcher.next_batch()

        batch = Bert_model.classify_batcher.merge_multi_task(tagging_batch, classify_batch)

        results = Bert_model.run_dev_step(batch)


if __name__ == '__main__':
    tf.app.run()
