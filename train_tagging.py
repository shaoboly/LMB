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

#os.environ["CUDA_VISIBLE_DEVICES"]=""

def create_train_eval_model(FLAGS):
    Classify_model = model_pools["tagging_model"]

    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)
    if FLAGS.max_seq_length > bert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length %d because the BERT model "
            "was only trained up to sequence length %d" %
            (FLAGS.max_seq_length, bert_config.max_position_embeddings))

    tf.gfile.MakeDirs(FLAGS.output_dir)

    # load custom processer from task name
    task_name = FLAGS.task_name.lower()
    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))


    processor = processors[task_name]()
    train_batcher = Batcher(processor, FLAGS)
    # create trainning model
    Bert_model = Classify_model(bert_config, train_batcher, FLAGS)
    Bert_model.build_graph()
    Bert_model.create_or_load_recent_model()


    FLAGS_eval = FLAGS._asdict()
    FLAGS_eval["mode"] = "dev"
    FLAGS_eval = config.generate_nametuple(FLAGS_eval)
    validate_batcher =Batcher(processor, FLAGS_eval)
    validate_model = Classify_model(bert_config, validate_batcher, FLAGS_eval)
    validate_model.build_graph()
    validate_model.create_or_load_recent_model()

    return Bert_model,validate_model



def train_with_eval(FLAGS):
    FLAGS = config.retype_FLAGS()
    Bert_model, validate_model = create_train_eval_model(FLAGS)

    checkpoint_basename = os.path.join(FLAGS.output_dir, "Bert-Classify")
    logging.info(checkpoint_basename)
    Bert_model.save_model(checkpoint_basename)
    Best_acc = eval_acc(FLAGS,validate_model,Bert_model)
    bestDevModel = tf.train.get_checkpoint_state(FLAGS.output_dir).model_checkpoint_path

    start_step = Bert_model.load_specific_variable(Bert_model.global_step)
    for step in range(start_step,Bert_model.num_train_steps):
        batch =Bert_model.batcher.next_batch()
        if batch==None:
            bestDevModel, Best_acc, acc = greedy_model_save(bestDevModel, checkpoint_basename, Best_acc, Bert_model,
                                                            validate_model, FLAGS)
            logging.info("Finish epoch: {}".format(Bert_model.batcher.c_epoch))
            logging.info("ACC {} Best_ACC: {}\n\n".format(acc, Best_acc))
            if Bert_model.batcher.c_epoch>=FLAGS.num_train_epochs:
                break
            continue

        results = Bert_model.run_train_step(batch)

        if step%100==0:
            logging.info("step {} loss: {}\n".format(step,results["loss"]))

        if step%FLAGS.save_checkpoints_steps==0 and step!=0:
            bestDevModel, Best_acc, acc = greedy_model_save(bestDevModel, checkpoint_basename, Best_acc, Bert_model, validate_model, FLAGS)
            logging.info("ACC {} Best_ACC: {}\n\n".format(acc, Best_acc))

    Bert_model.load_specific_model(bestDevModel)
    Bert_model.save_model(bestDevModel, False)

def greedy_model_save(bestDevModel,checkpoint_basename,Best_acc,Bert_model,validate_model,FLAGS):
    Bert_model.save_model(checkpoint_basename,False)
    acc = eval_acc(FLAGS, validate_model, Bert_model)
    if acc >= Best_acc:
        Bert_model.save_model(checkpoint_basename, True)
        bestDevModel = tf.train.get_checkpoint_state(FLAGS.output_dir).model_checkpoint_path
        Best_acc = acc
        logging.info("save new model")

    return bestDevModel,Best_acc,acc

def eval_acc(FLAGS,dev_model,train_model):
    dev_model.graph.as_default()
    dev_model.create_or_load_recent_model()
    dev_loss = 0
    valid_batcher = dev_model.batcher

    loss_all = []
    first_tokens_mask = []
    predictions = []
    labels = []
    while True:
        batch =dev_model.batcher.next_batch()
        if batch==None:
            break
        results = dev_model.run_dev_step(batch)
        loss_all.append(results["loss"])

        if batch["real_length"]<FLAGS.train_batch_size:
            results["predictions"] = list(results["predictions"])[:batch["real_length"]]
            batch["label_ids"] = batch["label_ids"][:batch["real_length"]]
            batch['first_token_positions'] = batch['first_token_positions'][:batch["real_length"]]
        predictions+=list(results["predictions"])
        labels+=batch["label_ids"]
        first_tokens_mask+=batch['first_token_positions']
        #print(batch["label_ids"],results["predictions"])

    loss = np.average(loss_all)
    total = len(predictions)

    word_cnt = 0
    word_cnt_acc = 0
    sentence_acc = 0
    for preds,labels,first_masks in zip(predictions,labels,first_tokens_mask):
        diff = True
        for p,r,m in zip(preds,labels,first_masks):
            if m==0:
                continue
            word_cnt+=1
            if p==r:
                word_cnt_acc+=1
            else:
                diff=False
        if diff:
            sentence_acc+=1

    sentence_acc = sentence_acc / total
    word_cnt_acc = word_cnt_acc/word_cnt
    logging.info("sentence accuracy: {} word accuracy: {} dev loss:{}".format(np.round(sentence_acc*100,4),np.round(word_cnt_acc*100,4),loss))

    train_model.graph.as_default()
    return sentence_acc

def eval(FLAGS):
    pass

def test(FLAGS):
    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)
    Classify_model = model_pools["tagging_model"]
    processor = processors[FLAGS.task_name.lower()]()

    validate_batcher = Batcher(processor, FLAGS)
    dev_model = Classify_model(bert_config, validate_batcher, FLAGS)
    dev_model.build_graph()
    dev_model.create_or_load_recent_model()
    #checkpoint_basename = os.path.join(FLAGS.output_dir, "Bert-Classify")

    dev_model.graph.as_default()
    dev_model.create_or_load_recent_model()

    loss_all = []
    logits_all = []
    predictions = []
    labels = []
    first_tokens_mask = []
    out_f = open(os.path.join(FLAGS.output_dir, "infer_resuts.tsv"),"w",encoding="utf-8")
    while True:
        batch = dev_model.batcher.next_batch()
        if batch == None:
            break
        results = dev_model.run_dev_step(batch)
        loss_all.append(results["loss"])

        if batch["real_length"] < FLAGS.train_batch_size:
            results["predictions"] = list(results["predictions"])[:batch["real_length"]]
            batch["label_ids"] = batch["label_ids"][:batch["real_length"]]
            batch['first_token_positions'] = batch['first_token_positions'][:batch["real_length"]]
        predictions += list(results["predictions"])
        labels += batch["label_ids"]
        first_tokens_mask += batch['first_token_positions']
        # print(batch["label_ids"],results["predictions"])

    loss = np.average(loss_all)
    total = len(predictions)

    word_cnt = 0
    word_cnt_acc = 0
    sentence_acc = 0
    for preds, labels, first_masks in zip(predictions, labels, first_tokens_mask):
        diff = True
        for p, r, m in zip(preds, labels, first_masks):
            if m == 0:
                continue
            word_cnt += 1
            if p == r:
                word_cnt_acc += 1
            else:
                diff = False
        if diff:
            sentence_acc += 1

    sentence_acc = sentence_acc / total
    word_cnt_acc = word_cnt_acc / word_cnt
    logging.info("sentence accuracy: {} word accuracy: {} dev loss:{}".format(np.round(sentence_acc * 100, 4),
                                                                              np.round(word_cnt_acc * 100, 4), loss))

    dev_model.graph.as_default()
    return sentence_acc


def main(_):
    FLAGS = config.FLAGS
    head = '[%(asctime)-15s %(levelname)s] %(message)s'
    logging.basicConfig(level=logging.INFO, format=head)
    logging.info('Starting seq2seq_attention in %s mode...', (FLAGS.mode))

    if FLAGS.mode == 'train':
        train_with_eval(FLAGS)
    elif FLAGS.mode == 'dev':
        pass
    elif FLAGS.mode == 'test':
        test(FLAGS)
    else:
        raise ValueError("The 'mode' flag must be one of train/eval/decode")


if __name__ == '__main__':
    tf.app.run()