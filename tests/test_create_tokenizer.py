from unittest import TestCase
import unittest
from create_tokenizer import remove_special_characters, extract_all_chars, create_tokenizer
import pandas as pd
import re
import json
import datasets


chars_to_ignore_regex = ['"', "'", "*", "()", "[\]", "\-", ".","`", ",","_", "+/=%|"]
pattern_dot_decimal = "\S+\&\S+"
CHARS_TO_IGNORE = f'[{"".join(chars_to_ignore_regex)}]'


test1 = {"transcript": "abc-de*&%afgt-qhelo( vui nha", "output": "abc de & afgt qhelo vui nha"}
test2 = {"transcript": "cong hoa*xa hoi 'chu nghia (viet-nam) doc-lap=tu do+hanh phuc|", "output": "cong hoa xa hoi chu nghia viet nam doc lap tu do hanh phuc "}
test3 = {"transcript": "%my dating&hoa* chu' malaya-sixten-up) multi*infi =ta do ka phuc|", "output": " my dating&hoa chu malaya sixten up multi infi ta do ka phuc "}
test4 = {"transcript": "dengan tarikan. graviti yang-itu tarikan sahaja* bila kita Alif", "output": "dengan tarikan graviti yang itu tarikan sahaja bila kita alif"}
test5 = {"transcript": "akak-ni*lagi(Okey apa contoh persilangan popular) Banyaknya Okey contoh kan+macam filem'tempe-macam sakni do ataupun. filemok busters macam film Trans movers dia cuba nak buat Tranformers dalam bentuk yang bajet lebih rendah kemudian ada filem yang legend",
         "output": "akak ni lagi okey apa contoh persilangan popular banyaknya okey contoh kan macam filem tempe macam sakni do ataupun filemok busters macam film trans movers dia cuba nak buat tranformers dalam bentuk yang bajet lebih rendah kemudian ada filem yang legend"}

data_test = [test1, test2, test3, test4, test5]
df_test = pd.DataFrame({"transcript": [i["transcript"] for i in data_test], "output": [i["output"] for i in data_test]})


class Test(TestCase):

    def test_create_tokenizer(self):
        data = datasets.Dataset.from_pandas(df_test)
        data = data.map(remove_special_characters)
        vocabs = data.map(extract_all_chars)
        vocab_dict, _ = create_tokenizer(df_test)
        dict_char = {}

        print(set(dict_char.keys()))
        for i in vocabs["vocab"]:
            for j in i:
                if j in set(dict_char.keys()):
                    dict_char[j] = dict_char[j]+1
                else:
                    dict_char[j] = 1

        self.assertEqual(dict_char.keys(), vocab_dict.keys())
        vocab_dict["|"] = vocab_dict[" "]
        del vocab_dict[" "]
        vocab_dict["[UNK]"] = len(vocab_dict)
        vocab_dict["[PAD]"] = len(vocab_dict)
        dict_char["|"] = dict_char[" "]
        del dict_char[" "]
        dict_char["[UNK]"] = len(dict_char)
        dict_char["[PAD]"] = len(dict_char)
        self.assertEqual(dict_char.keys(), vocab_dict.keys())
        print("Test create tokenizer OK")



    def test_remove_special_characters(self):
        for test in data_test:
            data = remove_special_characters(test)
            print(data["transcript"])
            self.assertEqual(test["output"],data["transcript"])
        print("Test remove special characters OK")

if __name__ == "__main":
    unittest.main()