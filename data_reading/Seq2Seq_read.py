import tensorflow as tf
import os

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
                sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
                Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
                specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, output_ids, decoder_inp):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.decoder_inp = decoder_inp
        self.output_ids = output_ids

        self._covert_to_dict()
    def _covert_to_dict(self):
        self.fd ={}
        self.fd["input_ids"] = self.input_ids
        self.fd["input_mask"] = self.input_mask
        self.fd["segment_ids"] = self.segment_ids

        self.fd["decoder_inp"] = self.decoder_inp
        self.fd["output_ids"] = self.output_ids

class MSPaD():
    def get_train_examples(self, data_dir,fname=None):
        """See base class."""
        if fname!=None:
            return self._create_examples(self._read_tsv(os.path.join(data_dir, fname)), "train")
        return self._create_examples(
                self._read_tsv(os.path.join(data_dir, "train.txt.format")), "train")

    def get_dev_examples(self, data_dir,fname=None):
        """See base class."""
        if fname!=None:
            return self._create_examples(
                self._read_tsv(os.path.join(data_dir, fname)), "dev")
        return self._create_examples(
                self._read_tsv(os.path.join(data_dir, "dev.txt.format")), "dev")

    def get_test_examples(self, data_dir,fname=None):
        """See base class."""
        if fname!=None:
            return self._create_examples(
                self._read_tsv(os.path.join(data_dir, fname)), "test")
        return self._create_examples(
                self._read_tsv(os.path.join(data_dir, "new-test.tsv.format")), "test")

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    def convert_examples_to_features(self,examples, label_list, max_seq_length, config):
        """Loads a data file into a list of `InputBatch`s."""
        raise NotImplementedError()

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = tokenization.convert_to_unicode(line[1])
            label = tokenization.convert_to_unicode(line[2])
            examples.append(
                    InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with tf.gfile.Open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for i, line in enumerate(reader):
                lines.append(line)
            return lines
