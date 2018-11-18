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
            examples = processor.get_train_examples(config.data_dir,config.train_file)
            random.shuffle(examples)
        elif mode=="dev":
            examples = processor.get_dev_examples(config.data_dir,config.dev_file)
        elif mode=="test":
            examples = processor.get_test_examples(config.data_dir,config.test_file)
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

        batch["real_length"] = len(batch["input_ids"])
        while len(batch["input_ids"])<self._config.train_batch_size:
            for k in features_list:
                batch[k].append(batch[k][-1])

        batch["size"] = self._config.train_batch_size

        self.c_index=e_index
        return batch

    def merge_multi_task(self,batch1,batch2):
        if batch1==None or batch2==None:
            return None

        batch1_size = batch1["size"]
        batch2_size = batch2["size"]

        for k in batch2.keys():
            if k in batch1:
                batch1[k]+=batch2[k]
            else:
                batch1[k] = [batch2[k][0] for i in range(batch1_size)]+batch2[k]

        for k in batch1.keys():
            if k not in batch2:
                batch1[k]+=[batch1[k][0] for i in range(batch2_size)]

        batch1["task_mask"] = [1.0 for i in range(batch1_size)] + [0.0 for i in range(batch2_size)]
        return batch1





    def reshuffle(self):
        pass


