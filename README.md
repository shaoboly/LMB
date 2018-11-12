# LMB
This model is modified from Bert by Bo.

The running script as below:
#requirement: tensorflow=1.11


export BERT_BASE_DIR=/home/v-boshao/code/bert-intent/model/uncased_L-12_H-768_A-12
export GLUE_DIR=/home/v-boshao/code/bert-intent/glue_data

#trainning:

python train.py \
  --task_name=QICM \
  --mode=train \
  --data_dir=$GLUE_DIR/QICMovie \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --max_seq_length=128 \
  --train_batch_size=32 \
  --learning_rate=2e-5 \
  --num_train_epochs=3.0 \
  --output_dir=./output

  
#test
python train.py \
  --task_name=QICM \
  --mode=test \
  --data_dir=$GLUE_DIR/QICMovie \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --max_seq_length=128 \
  --train_batch_size=32 \
  --learning_rate=2e-5 \
  --num_train_epochs=3.0 \
  --output_dir=./output
  
  
  
The model and results saved at output_dir. 
The accuracy I trained on movie domain is about 66%. 
