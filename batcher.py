import tensorflow as tf
import random
import tokenization
import logging


class Batcher(object):
    def __init__(self,processor,config):
        logging.info("Loading Data:")

        self._config = config
        self.processor = processor

        #preprocess_data
        self.build_features()

        #counter
        self.c_epoch = 0
        self.c_index = 0



    def build_features(self):
        processor = self.processor
        mode = self._config.mode
        config = self._config

        if mode=="train":
            examples = processor.get_train_examples(config.data_dir)
            random.shuffle(examples)
        elif mode=="dev":
            examples = processor.get_dev_examples(config.data_dir)
        elif mode=="test":
            examples = processor.get_test_examples(config.data_dir)
        else:
            raise ValueError("Only train dev test modes are supported: %s" % (mode))

        self.label_num = len(self.processor.get_labels())
        self.samples_number = len(examples)
        self.examples = examples

        label_list = self.processor.get_labels()
        self.feed_features = self.processor.convert_examples_to_features(self.examples,label_list,
                                                          self._config.max_seq_length,self._config)
        '''
        all_input_ids = []
        all_input_mask = []
        all_segment_ids = []
        all_label_ids = []
        for feature in self.feed_features:
            all_input_ids.append(feature.input_ids)
            all_input_mask.append(feature.input_mask)
            all_segment_ids.append(feature.segment_ids)
            all_label_ids.append(feature.label_id)'''

    def next_batch(self):
        if self.c_index>=self.samples_number:
            self.c_index=0
            self.c_epoch+=1
            return None

        e_index = self.c_index+self._config.train_batch_size

        features_list = list(self.feed_features[0].fd.keys())
        batch = {}
        for k in features_list:
            batch[k] = []

        for feature in self.feed_features[self.c_index:e_index]:
            for k in features_list:
                batch[k].append(feature.fd[k])
            '''
            batch["input_ids"].append(feature.input_ids)
            batch["input_mask"].append(feature.input_mask)
            batch["segment_ids"].append(feature.segment_ids)
            batch["label_ids"].append(feature.label_id)'''

        batch["real_length"] = len(batch["input_ids"])
        while len(batch["input_ids"])<self._config.train_batch_size:
            for k in features_list:
                batch[k].append(batch[k][-1])
            '''
            batch["input_ids"].append(batch["input_ids"][-1])
            batch["input_mask"].append(batch["input_mask"][-1])
            batch["segment_ids"].append(batch["segment_ids"][-1])
            batch["label_ids"].append(batch["label_ids"][-1])'''

        self.c_index=e_index
        return batch



    def reshuffle(self):
        pass


def input_fn_builder(features, seq_length, is_training, drop_remainder):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    all_input_ids = []
    all_input_mask = []
    all_segment_ids = []
    all_label_ids = []

    for feature in features:
        all_input_ids.append(feature.input_ids)
        all_input_mask.append(feature.input_mask)
        all_segment_ids.append(feature.segment_ids)
        all_label_ids.append(feature.label_id)

    def input_fn(params):
        """The actual input function."""
        batch_size = params["batch_size"]

        num_examples = len(features)

        # This is for demo purposes and does NOT scale to large data sets. We do
        # not use Dataset.from_generator() because that uses tf.py_func which is
        # not TPU compatible. The right way to load data is with TFRecordReader.
        d = tf.data.Dataset.from_tensor_slices({
                "input_ids":
                        tf.constant(
                                all_input_ids, shape=[num_examples, seq_length],
                                dtype=tf.int32),
                "input_mask":
                        tf.constant(
                                all_input_mask,
                                shape=[num_examples, seq_length],
                                dtype=tf.int32),
                "segment_ids":
                        tf.constant(
                                all_segment_ids,
                                shape=[num_examples, seq_length],
                                dtype=tf.int32),
                "label_ids":
                        tf.constant(all_label_ids, shape=[num_examples], dtype=tf.int32),
        })

        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=100)

        d = d.batch(batch_size=batch_size, drop_remainder=drop_remainder)
        return d

    return input_fn