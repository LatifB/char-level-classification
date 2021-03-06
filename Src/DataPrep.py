from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import glob
import os
import random

import torch


class DataPrep:

    def __init__(self, path='data/names/*.txt'):
        self.all_files = glob.glob(path)

        self.category_lines = {}
        self.all_categories = []
        self.all_letters = string.ascii_letters # special char's and maybe lowercase?

        self.read_files()

        self.n_letters = len(self.all_letters)
        self.n_categories = len(self.all_categories)


    def unicode_to_ascii(self, s):
        return ''.join(
            c for c in unicodedata.normalize('NFD', s) 
            if unicodedata.category(c) != 'Mn' and 
            c in self.all_letters
        )
    
    def read_files(self):
        for filename in self.all_files:
            category = os.path.splitext(os.path.basename(filename))[0]
            self.all_categories.append(category)
            lines = self.read_lines(filename)
            self.category_lines[category] = [line for line in lines]
    
    def read_lines(self, filename):
        lines = open(filename, encoding='utf-8').read().strip().split('\n')
        return [self.unicode_to_ascii(line) for line in lines]

    def random_choice(self, l):
        return l[random.randint(0, len(l) - 1)]

    def random_training_sample(self):
        category = self.random_choice(self.all_categories)
        line = self.random_choice(self.category_lines[category])
        category_tensor = torch.tensor([self.all_categories.index(category)], dtype=torch.long)
        line_tensor = self.line2tensor(line)
        return category, line, category_tensor, line_tensor

    def letter2index(self, letter):
        return self.all_letters.find(letter)

    def line2tensor(self, line):
        tensor = torch.zeros(len(line), 1, self.n_letters)
        for i, letter in enumerate(line):
            letter_index = self.letter2index(letter)
            tensor[i, 0, letter_index] = 1
        return tensor

    def cat_from_output(self, output, n_predictions=3):
        topv, topi = output.topk(n_predictions, 1, True)
        outputs = []
        for i in range(n_predictions):
            outputs.append([
                            topv[i],
                            self.all_categories[topi[i]]
                           ]
                          )
        return outputs
