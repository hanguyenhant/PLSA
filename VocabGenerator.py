# class VocabGenerator:
# 	# min_df và max_df để loại bỏ những từ quá hiếm hoặc quá thông dụng
#     def __init__(self,min_df,max_df):
#         self.min_df = min_df
#         self.max_df = max_df

# 	def read_data(self, path):

# 		with open(path) as f:
# 			lines = f.read().splitlines()

# 		vocab = []

# 		for line in lines:
# 			text = line.split("<fff>")[2]
# 			words = text.split()
			
# 			for word in words:
# 				if word not in vocab:
# 					vocab.append(word)

# 		vocab = sorted(vocab)
		
# 		return vocab

# 	def save_to_file(self, path, vocab):
# 		with open(path, 'w') as f:
# 			f.write('\n'.join(vocab))

# if __name__ == '__main__':
# 	vocab_generator = VocabGenerator()
# 	# vocab = vocab_generator.read_data('train_processed.txt')
# 	# vocab_generator.save_to_file('vocab.txt', vocab)
# 	vocab = vocab_generator.read_data('data_processed.txt')
# 	vocab_generator.save_to_file('vocab_2.txt', vocab)

import numpy as np
from collections import defaultdict

class VocabGenerator:
    # min_df và max_df để loại bỏ những từ quá hiếm hoặc quá thông dụng
    def __init__(self,min_df,max_df):
        self.min_df=min_df
        self.max_df = max_df

    def read_data(self, data_path):
        print("Creating vocabulary...")
        self.data_path = data_path
        with open(self.data_path) as f:
            lines = f.read().splitlines()  # mỗi dòng là 1 văn bản
        corpus_size = len(lines)
        doc_count = defaultdict(int)
        for line in lines:
            features = line.split('<fff>')
            words = features[-1]  # lấy ra nội dung
            words = list(set(
                words.split()))  # phải dùng set thì 1 từ xuất hiện 2 lần trong 1 văn bản thì cũng chỉ tăng df lên 1 => sử dụng set để loại bỏ trùng lặp
            for w in words:
                doc_count[w] += 1

        vocab = [w for w, df in zip(doc_count.keys(), doc_count.values())
                if df/corpus_size >= self.min_df and df/corpus_size  <= self.max_df]

        vocab = sorted(vocab)

        print("Done")
        print("Vocab size: %d" % len(vocab))
        print("Saving vocabulary...")
        with open("vocab.txt", "w") as f:
            f.write('\n'.join(vocab))
        print("Saved")

if __name__ == '__main__':
    vocabgenerator = VocabGenerator(min_df = 0.0005, max_df = 0.02)
    vocabgenerator.read_data('data_processed.txt')












