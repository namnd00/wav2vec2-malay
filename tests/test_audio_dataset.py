from unittest import TestCase
from audio_dataset import MalayAudioDataset, DataCollatorCTCWithPadding, AudioProcessor
import pandas as pd
import unittest
import os
import torchaudio
from transformers import Wav2Vec2CTCTokenizer
import numpy as np


AUDIO_DIR = "/home/sanhlx/PycharmProjects/wav2vec2-malay/examples/datasets/waves"
ANNOTATION_DIR = "/home/sanhlx/PycharmProjects/wav2vec2-malay/examples/datasets/annotations.csv"
VOCAB_PATH = "/home/sanhlx/PycharmProjects/wav2vec2-malay/examples/datasets/vocab.json"
ANNOTATION_DF = pd.read_csv(ANNOTATION_DIR)

audioProcessor_test_class = AudioProcessor(vocab_path= VOCAB_PATH)
malayAudioDataset_test_class = MalayAudioDataset(annotation_df = ANNOTATION_DF, audio_dir = AUDIO_DIR,
                                                 audio_transforms= False, audio_processor= audioProcessor_test_class)
processor_test = audioProcessor_test_class.processor

dataCollatorCTCWithPadding_test_class = DataCollatorCTCWithPadding(processor=processor_test, padding=True)
Datatest = pd.read_csv(ANNOTATION_DIR)

tokenizer_check = Wav2Vec2CTCTokenizer(vocab_file=VOCAB_PATH, unk_token="[UNK]",
                                        pad_token="[PAD]", word_delimiter_token="|")
#Test MalayAudioDataset
class TestMalayAudioDataset(TestCase):

    # test_get_audio_sample_path
    def test__get_audio_sample_path(self):
        for i in range(len(Datatest)):
            path = malayAudioDataset_test_class._get_audio_sample_path(i)
            path_check = os.path.join(AUDIO_DIR, Datatest.iloc[i, 0])
            self.assertEqual(str(type(path)), str(type(path_check)))
            self.assertMultiLineEqual(path, path_check)
            self.assertEqual(path_check, path)
        print("Test get audio sample path OK")

    # test_get_labels
    def test__get_labels(self):
        for i in range(len(Datatest)):
            labels = malayAudioDataset_test_class._get_labels(i)
            labels_check = Datatest.iloc[i, 1].lower()
            self.assertEqual(str(type(labels)), str(type(labels_check)))
            self.assertMultiLineEqual(labels, labels_check)
            self.assertEqual(labels_check, labels)
        print("Test get label OK")

    #test_resample_if_necessary
    def test__resample_if_necessary(self):
        for i in range(len(Datatest)):
            path_check = os.path.join(AUDIO_DIR, Datatest.iloc[i, 0])
            audio_array_check, sr = torchaudio.load(path_check)
            audio_array = malayAudioDataset_test_class._resample_if_necessary(audio_array_check, sr)
            self.assertEqual(str(type(audio_array)), str(type(audio_array_check)))
            self.assertEqual(audio_array.shape, audio_array_check.shape)
            self.assertListEqual(list(audio_array[0]), list(audio_array_check[0]))
        print("Test resample if necessary OK")

    # def test__get_audio_transforms(self):
    #
    #     self.fail()


    #test_prepare_signal:
    def test__prepare_signal(self):
        for i in range(len(Datatest)):
            path_check = os.path.join(AUDIO_DIR, Datatest.iloc[i, 0])
            audio_array_check, sr = torchaudio.load(path_check)

            audio_array = malayAudioDataset_test_class._resample_if_necessary(audio_array_check, sr)

            prepare_array = malayAudioDataset_test_class._prepare_signal(audio_array, sr)

            self.assertEqual(audio_array.shape, prepare_array[0].shape )
            #self.assertEqual(str(type(audio_array)), str(type(prepare_array[0])))

            #print("mean_old: %f, var_old: %f, max_old: %f, min_old: %f" % (audio_array_check.mean(), audio_array_check.var(), audio_array_check.max(), audio_array_check.min()))
            #print("mean: %f, var: %f, max: %f, min: %f" % (prepare_array[0].mean(), prepare_array[0].var(), prepare_array[0].max(), prepare_array[0].min()))

            self.assertLessEqual(abs(prepare_array[0].mean()), 1e-6)
            self.assertGreaterEqual(prepare_array[0].var(), 0.9)
            self.assertLessEqual(prepare_array[0].var(), 1.1)
        print("Test prepare signal OK")
    #test_prepare_label:
    def test__prepare_label(self):
        for i in range(len(Datatest)):
            labels = malayAudioDataset_test_class._get_labels(i)
            prepare_label = malayAudioDataset_test_class._prepare_label(labels)
            prepare_label_check = tokenizer_check(labels).input_ids
            #self.assertEqual(str(type(audio_array)), str(type(prepare_array[0])))
            self.assertEqual(len(prepare_label_check), len(prepare_label))
            self.assertListEqual(prepare_label_check, prepare_label)
        print("Test prepare labels OK")


    #Test DataCollatorCTCWithPadding
    def test__DataCollatorCTCWithPadding(self):
        batch_size = 16
        batch_data = []
        batch_data_check = []

        array = np.arange(0, len(Datatest)-1)
        array_random = np.random.choice(array, batch_size)
        audio_max_len = 0
        label_max_len = 0
        for i in array_random:
            data_check = {}
            test = malayAudioDataset_test_class.__getitem__(i)

            path_file = os.path.join(AUDIO_DIR, Datatest.iloc[i, 0])
            audio_array_check, sr = torchaudio.load(path_file)
            audio_array = malayAudioDataset_test_class._resample_if_necessary(audio_array_check, sr)
            prepare_array = malayAudioDataset_test_class._prepare_signal(audio_array[0], sr)

            transcript = Datatest.iloc[i, 1].lower()
            labels = malayAudioDataset_test_class._prepare_label(transcript)
            data_check["input_values"] = prepare_array[0]
            data_check["labels"] = labels
            data_check["label_prepare"] = test["labels"]
            data_check["input_values_length"] = len(prepare_array[0])
            data_check["labels_len"] = len(labels)
            batch_data.append(test)
            batch_data_check.append(data_check)
            if len(labels) >= label_max_len:
                label_max_len = len(labels)
            if len(prepare_array[0]) >= audio_max_len:
                audio_max_len = len(prepare_array[0])

        batch_collatortest = dataCollatorCTCWithPadding_test_class(batch_data)

        for j in range(batch_size):
            idx_array = batch_data_check[j]["input_values_length"]
            idx_label = batch_data_check[j]["labels_len"]
            array_collor = np.array([batch_collatortest["input_values"][j][i].item() for i in range(len(batch_collatortest["input_values"][j]))])
            label_collor = np.array([batch_collatortest["labels"][j][i].item() for i in range(len(batch_collatortest["labels"][j]))])

            self.assertListEqual(list(array_collor[:idx_array]),
                                 list(batch_data_check[j]["input_values"]))
            self.assertListEqual(list(label_collor[:idx_label]),
                                      list(batch_data_check[j]["labels"]))


            self.assertEqual(sum(array_collor[idx_array:]), 0)
            self.assertEqual(sum(label_collor[idx_label:]), -100 * (len(batch_collatortest["labels"][j]) - idx_label))
            if idx_array != audio_max_len:
                self.assertEqual(array_collor[idx_array:].mean(), 0)
            else:
                self.assertLessEqual(abs(array_collor.mean()), 1e-6)
                self.assertGreaterEqual(array_collor.var(), 0.9)
                self.assertLessEqual(array_collor.var(), 1.1)
            if idx_label != label_max_len:
                self.assertEqual(label_collor[idx_label:].mean(), -100)

        print("Test DataCollatorCTCWithPadding OK")


# class Test(TestCase):
#     def test_compute_metrics(self):
#         self.fail()

if __name__ == "__main__":
    unittest.main(verbosity=2)