from model_pools import modeling
import tensorflow as tf
import optimization
import logging
import os

class ClassifyModel(object):
    def __init__(self,bert_config, batcher, hps):
        self.hps = hps
        self.bert_config = bert_config
        self.is_training = (self.hps.mode=="train")
        self.batcher = batcher

        self.num_train_steps = int(batcher.samples_number / hps.train_batch_size * hps.num_train_epochs)
        self.num_warmup_steps = int(self.num_train_steps * hps.warmup_proportion)

    def build_graph(self):
        self.graph = tf.Graph()
        _config = tf.ConfigProto(allow_soft_placement=True, gpu_options=tf.GPUOptions(allow_growth=True))
        self.gSess_train = tf.Session(config=_config, graph=self.graph)

        logging.debug("Graph id: {}{}".format(id(self.graph), self.graph))
        with self.graph.as_default():
            self._build_classify_model()
            self.train_op=None
            if self.is_training:
                self.train_op = optimization.create_optimizer(
                    self.loss, float(self.hps.learning_rate), self.num_train_steps, self.num_warmup_steps, self.hps.use_tpu)
                #self._load_init_bert_parameter()
            self._make_input_key()

            self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)
            self.global_step = tf.train.get_or_create_global_step()
            #self.gSess_train.run(tf.global_variables_initializer())


    def _add_placeholders(self):
        hps = self.hps
        input_ids = tf.placeholder(tf.int32, [hps.train_batch_size, hps.max_seq_length], name='input_ids')
        input_mask = tf.placeholder(tf.int32, [hps.train_batch_size, hps.max_seq_length], name='input_mask')
        segment_ids = tf.placeholder(tf.int32, [hps.train_batch_size, hps.max_seq_length], name='segment_ids')
        label_ids = tf.placeholder(tf.int32, [hps.train_batch_size], name='label_ids')

        '''self.learning_rate = tf.Variable(float(self.hps.learning_rate), trainable=False, dtype=tf.float32)
        self.learning_rate_decay_op = self.learning_rate.assign(
            self.learning_rate * self.hps.learning_rate_decay_factor)'''
        return input_ids,input_mask,segment_ids,label_ids


    def _build_classify_model(self):
        is_training = self.is_training
        num_labels = self.batcher.label_num

        input_ids, input_mask, segment_ids, label_ids = self._add_placeholders()
        self.input_ids, self.input_mask, self.segment_ids, self.label_ids = input_ids, input_mask, segment_ids, label_ids

        """Creates a classification model."""
        model = modeling.BertModel(
            config=self.bert_config,
            is_training=is_training,
            input_ids=input_ids,
            input_mask=input_mask,
            token_type_ids=segment_ids,
            use_one_hot_embeddings=self.hps.use_tpu)#use_one_hot_embeddings=Flags.tpu ?

        output_layer = model.get_pooled_output()

        hidden_size = output_layer.shape[-1].value
        output_weights = tf.get_variable(
            "output_weights", [num_labels, hidden_size],
            initializer=tf.truncated_normal_initializer(stddev=0.02))

        output_bias = tf.get_variable(
            "output_bias", [num_labels], initializer=tf.zeros_initializer())

        with tf.variable_scope("loss"):
            if is_training:
                # I.e., 0.1 dropout
                output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)

            logits = tf.matmul(output_layer, output_weights, transpose_b=True)
            logits = tf.nn.bias_add(logits, output_bias)
            log_probs = tf.nn.log_softmax(logits, axis=-1)

            one_hot_labels = tf.one_hot(label_ids, depth=num_labels, dtype=tf.float32)

            per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
            loss = tf.reduce_mean(per_example_loss)

        self.loss, self.per_example_loss, self.logits \
            = loss, per_example_loss, logits
        self.predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)


    def _load_init_bert_parameter(self):
        init_checkpoint = self.hps.init_checkpoint
        tvars = tf.trainable_variables()
        if init_checkpoint:
            (assignment_map,initialized_variable_names) = modeling.get_assigment_map_from_checkpoint(tvars, init_checkpoint)
            if self.hps.use_tpu:
                def tpu_scaffold():
                    tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
                    return tf.train.Scaffold()
                scaffold_fn = tpu_scaffold
            else:
                tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

            tf.logging.info("**** Trainable Variables ****")
            for var in tvars:
                init_string = ""
                if var.name in initialized_variable_names:
                    init_string = ", *INIT_FROM_CKPT*"
                tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,init_string)


    def _make_input_key(self):
        self.tensor_list = {"input_ids": self.input_ids,
                            "input_mask": self.input_mask,
                            "segment_ids": self.segment_ids,
                            "label_ids": self.label_ids,
                            "train_opt": self.train_op,
                            "loss":self.loss,
                            "per_example_loss":self.per_example_loss,
                            "logits":self.logits,
                            "predictions":self.predictions,
                            }
        self.input_keys = ["input_ids","input_mask","segment_ids","label_ids"]
        self.output_keys_train = ["loss","per_example_loss","train_opt"]
        self.output_keys_dev = ["loss", "logits","predictions"]


    def _make_feed_dict(self,batch):
        feed_dict = {}
        for k in self.input_keys:
            feed_dict[self.tensor_list[k]] = batch[k]

        return feed_dict

    def run_train_step(self,batch):
        to_return = {}
        for k in self.output_keys_train:
            to_return[k] = self.tensor_list[k]
        feed_dict = self._make_feed_dict(batch)
        return self.gSess_train.run(to_return, feed_dict)

    def run_dev_step(self,batch):
        to_return = {}
        for k in self.output_keys_dev:
            to_return[k] = self.tensor_list[k]
        feed_dict = self._make_feed_dict(batch)
        return self.gSess_train.run(to_return, feed_dict)


    def create_or_load_recent_model(self):
        with self.graph.as_default():
            if not os.path.isdir(self.hps.output_dir):
                os.mkdir(self.hps.output_dir)
            ckpt = tf.train.get_checkpoint_state(self.hps.output_dir)
            if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
                self.saver.restore(self.gSess_train, ckpt.model_checkpoint_path)
            else:
                logging.info("Created model with fresh parameters and bert.")
                self._load_init_bert_parameter()
                self.gSess_train.run(tf.global_variables_initializer())

    def load_specific_variable(self,v):
        with self.graph.as_default():
            return self.gSess_train.run(v)

    def save_model(self,checkpoint_basename,with_step = True):
        with self.graph.as_default():
            global_step = tf.train.get_or_create_global_step()
            if with_step:
                self.saver.save(self.gSess_train, checkpoint_basename, global_step=global_step)
            else:
                self.saver.save(self.gSess_train, checkpoint_basename)
            logging.info("model save {}".format(checkpoint_basename))

    def load_specific_model(self,best_path):
        with self.graph.as_default():
            self.saver.restore(self.gSess_train, best_path)


def create_model(bert_config, is_training, input_ids, input_mask, segment_ids,
                                 labels, num_labels, use_one_hot_embeddings):
    """Creates a classification model."""
    model = modeling.BertModel(
            config=bert_config,
            is_training=is_training,
            input_ids=input_ids,
            input_mask=input_mask,
            token_type_ids=segment_ids,
            use_one_hot_embeddings=use_one_hot_embeddings)

    # In the demo, we are doing a simple classification task on the entire
    # segment.
    #
    # If you want to use the token-level output, use model.get_sequence_output()
    # instead.
    output_layer = model.get_pooled_output()

    hidden_size = output_layer.shape[-1].value

    output_weights = tf.get_variable(
            "output_weights", [num_labels, hidden_size],
            initializer=tf.truncated_normal_initializer(stddev=0.02))

    output_bias = tf.get_variable(
            "output_bias", [num_labels], initializer=tf.zeros_initializer())

    with tf.variable_scope("loss"):
        if is_training:
            # I.e., 0.1 dropout
            output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)

        logits = tf.matmul(output_layer, output_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)
        log_probs = tf.nn.log_softmax(logits, axis=-1)

        one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)

        per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
        loss = tf.reduce_mean(per_example_loss)

        return (loss, per_example_loss, logits)

