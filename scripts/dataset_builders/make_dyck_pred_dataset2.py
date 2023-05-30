import sys
import numpy as np
import torch
from collections import defaultdict
import random

sys.setrecursionlimit(5000)

all_pairs = ['()', '[]', '{}', '<>']
all_letters = ''
for elt in all_pairs:
    all_letters += elt

init_ascii = 48  ## corresponds to 0


class DyckLanguage():
    def __init__(self, num_pairs, p, q):
        self.pair_num = num_pairs
        self.pairs = all_pairs[:num_pairs]
        self.vocabulary = all_letters[:2 * num_pairs]
        self.n_letters = len(self.vocabulary)

        self.openpar = [elt[0] for elt in self.pairs]
        self.closepar = [elt[1] for elt in self.pairs]

        self.p = p
        self.q = q

    # returns the vocabulary
    def return_vocab(self):
        return self.vocabulary

    # generate a sample
    def generate(self, current_size, max_size):
        # Houston, we have a problem here. (Limit exceeded.)
        if current_size >= max_size:
            return ''

        prob = random.random()
        # Grammar: S -> (_i S )_i with prob p | SS with prob q | empty with prob 1 - (p+q)
        if prob < self.p:
            chosen_pair = np.random.choice(self.pairs)  # randomly pick one of the pairs.
            sample = chosen_pair[0] + self.generate(current_size + 2, max_size) + chosen_pair[1]
            if len(sample) <= max_size:
                return sample
        elif prob < self.p + self.q:
            sample = self.generate(current_size, max_size) + self.generate(current_size, max_size)
            if len(sample) <= max_size:
                return sample
        else:
            return ''

        return ''

    # generate 'num' number of samples
    def generate_list(self, num, min_size, max_size):
        arr = []
        size_info = defaultdict(list)
        counter = 0
        while counter < num:
            sample = self.generate(0, max_size)
            if sample not in arr and len(sample) >= min_size:
                counter += 1
                arr.append(sample)
                size_info[len(sample)].append(sample)
                if counter % 500 == 0:
                    print('{} samples generated.'.format(counter))

        return arr, size_info

    def output_generator(self, seq):
        output_seq = ''
        stack = []

        for elt in seq:
            dyck_counter = [0 for _ in range(self.pair_num)]

            if elt in self.openpar:
                stack.append(self.closepar[self.openpar.index(elt)])
            else:
                stack.pop()

            if len(stack) > 0:
                index = self.closepar.index(stack[-1])
                dyck_counter[index] = 1

            temp = np.nonzero(dyck_counter)

            binary_code = 0

            for base in temp[0]:
                binary_code += (2 ** (base))

            output_seq += chr(binary_code + init_ascii)

        return output_seq

    def depth_counter(self, seq):
        dyck_counter = np.zeros(self.pair_num)
        max_depth = np.zeros((len(seq), self.pair_num))
        counter = 0
        for elt in seq:
            indexl = 0
            if elt in self.openpar:
                indexl = self.openpar.index(elt)
                dyck_counter[indexl] += 1
            else:
                indexl = self.closepar.index(elt)
                dyck_counter[indexl] -= 1
            max_depth[counter] = dyck_counter
            counter += 1
        return max_depth

    def training_set_generator(self, num, min_size, max_size):
        input_arr, input_size_arr = self.generate_list(num, min_size, max_size)
        output_arr = []
        for seq in input_arr:
            output_arr.append(self.output_generator(seq))
        return input_arr, output_arr, input_size_arr


if __name__ == '__main__':
    num_pairs = 4
    p = 0.4
    q = 0.4
    num = 10000
    min_size = 10
    max_size = 100

    dyck = DyckLanguage(num_pairs, p, q)
    lst = dyck.generate_list(10, 6, 6)

    print(lst)

    print('Done.')