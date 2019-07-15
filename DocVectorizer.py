from collections import defaultdict

class DocVectorizer:
	def __init__(self):
		self._data = []

	def read_data(self, path):

		with open('vocab.txt') as f:
			vocab = f.read().splitlines()

		with open(path) as f:
			lines = f.read().splitlines()

		#Xét trong mỗi văn bản
		for line in lines:
			doc_count = defaultdict(int)

			label, filename, text = line.split('<fff>')[0], line.split('<fff>')[1], line.split('<fff>')[2]

			words = text.split()

			#Đếm số lần xuất hiện của các từ trong văn bản đó
			for word in words:
				if word in vocab:
					doc_count[word] += 1

			content = ""

			for key in doc_count.keys():
				content += str(vocab.index(key)) + ":" + str(doc_count[key]) + " "

			self._data.append(label + "<fff>" + filename + "<fff>" + content)

	def write_to_file(self, path):
		with open(path, 'w') as f:
			f.write('\n'.join(self._data))

if __name__ == '__main__':
	doc_vectorizer = DocVectorizer()
	# doc_vectorizer.read_data('train_processed.txt')
	# doc_vectorizer.write_to_file('train_vectorizer.txt')
	doc_vectorizer.read_data('data_processed.txt')
	doc_vectorizer.write_to_file('data_vectorizer.txt')


