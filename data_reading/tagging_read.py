import os
#from data_reading.basic_read import *
import logging
import tokenization

class InputExample(object):
    def __init__(self, guid, words, tags):
        self.guid = guid
        self.words = words
        self.tags = tags

class InputFeatures(object):
    def __init__(self, input_ids, input_mask, segment_ids, first_token_positions, tags):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.first_token_positions = first_token_positions
        self.tags = tags
        self._covert_to_dict()

    def _covert_to_dict(self):
        self.fd ={}
        self.fd["input_ids"] = self.input_ids
        self.fd["input_mask"] = self.input_mask
        self.fd["segment_ids"] = self.segment_ids
        self.fd["tag_ids"] = self.tags
        self.fd["first_token_positions"] = self.first_token_positions

class TaggingProcessor(object):
    def get_train_examples(self, data_dir, fname=None):
        if fname!=None:
            return self._create_examples(os.path.join(data_dir, "open.train"), "train")
        return self._create_examples(os.path.join(data_dir, "open.train"), "train")

    def get_dev_examples(self, data_dir, fname=None):
        """Gets a collection of `InputExample`s for the dev set."""
        if fname!=None:
            return self._create_examples(os.path.join(data_dir, "open.train"), "dev")
        return self._create_examples(os.path.join(data_dir, "open.dev"), "train")

    def get_test_examples(self, data_dir, fname=None):
        """Gets a collection of `InputExample`s for the test set."""
        if fname!=None:
            return self._create_examples(os.path.join(data_dir, "open.train"), "test")
        return self._create_examples(os.path.join(data_dir, "open.test"), "train")

    def get_labels(self):
        return ['O', 'B']

    def _create_examples(self, file_path, set_type):
        examples = []
        f = open(file_path, 'r', encoding='utf-8', errors='ignore')
        words = []
        tags = []
        i = 0
        for line in f.readlines():
            data = line.strip()
            if data == "":
                guid = "%s-%d" % (set_type, i)
                examples.append(InputExample(guid=guid, words=words, tags=tags))
                words = []
                tags = []
                i = i + 1
            else:
                word, tag = data.split(' ')
                words.append(word)
                tags.append(tag)
        return examples

    def convert_examples_to_features(self,examples, tags_list, max_seq_length,config):

        tokenizer = tokenization.TaggingTokenizer(vocab_file=config.vocab_file, do_lower_case=config.do_lower_case)
        tag_map = {}
        for (i, tag) in enumerate(tags_list):
            tag_map[tag] = i

        features = []
        for (ex_index, example) in enumerate(examples):
            word_pieces, first_token_positions = tokenizer.tokenize(example.words)
            word_pieces = word_pieces[0:(max_seq_length - 2)]

            first_token_positions_temp = [position + 1 for position in first_token_positions if
                                          position < (max_seq_length - 2)]
            first_token_positions = [0] * max_seq_length
            tags = [0] * max_seq_length
            for i, tag in zip(first_token_positions_temp, example.tags):
                first_token_positions[i] = 1
                tags[i] = tag_map[tag]

            tokens = []
            segment_ids = []
            tokens.append("[CLS]")
            segment_ids.append(0)
            for token in word_pieces:
                tokens.append(token)
                segment_ids.append(0)
            tokens.append("[SEP]")
            segment_ids.append(0)
            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            input_mask = [1] * len(input_ids)
            while len(input_ids) < max_seq_length:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)
            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length

            if ex_index < 5:
                logging.info('*** Example ***')
                logging.info("guid: %s" % (example.guid))
                logging.info("tokens: %s" % (' '.join(tokens)))
                logging.info("input_ids: %s" % (' '.join([str(x) for x in input_ids])))
                logging.info("input_mask: %s" % (' '.join([str(x) for x in input_mask])))
                logging.info('segment_ids: %s' % (" ".join([str(x) for x in segment_ids])))
                logging.info('tags: %s' % (' '.join([str(x) for x in tags])))
                logging.info('first_token_positions: %s' % (' '.join([str(x) for x in first_token_positions])))

            features.append(
                InputFeatures(
                    input_ids=input_ids,
                    input_mask=input_mask,
                    segment_ids=segment_ids,
                    first_token_positions=first_token_positions,
                    tags=tags
                ))
        return features
