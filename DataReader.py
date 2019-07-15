from os import listdir
from os.path import isfile, join
# from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

class DataReader:
	def __init__(self):
		self._topic = []
		self._filepath = []
		# self.ps = PorterStemmer()
		self.stop_words = stopwords.words("english")
		self._data = []

	def read_topic_and_filename(self, path):
		for topic in listdir(path):

			if topic not in self._topic:
				self._topic.append(topic)

			topic_path = join(path, topic)

			for filename in listdir(topic_path):
				self._filepath.append(join(topic_path, filename))

	def collect_data(self):

		for file_path in self._filepath:

			topic, filename = self._topic.index(file_path.split('\\')[1]), file_path.split('\\')[2]

			with open(file_path, errors = "ignore") as f:
				text = f.read().lower()

				words = word_tokenize(text, 'english')

				words = [word for word in words if word not in self.stop_words and word.isalpha()]

				content = ' '.join(words)

				self._data.append(str(topic) + "<fff>" + filename + "<fff>" + content)	

	def save_data(self, path):
		with open(path, 'w') as f:
			f.write('\n'.join(self._data))

if __name__ == '__main__':
	data_reader = DataReader()
	# data_reader.read_topic_and_filename("20news-bydate-train")
	# data_reader.collect_data()
	# data_reader.save_data("train_processed.txt")

	# data_reader.read_topic_and_filename("20news-bydate-test")
	# data_reader.collect_data()
	# data_reader.save_data("test_processed.txt")
	data_reader.read_topic_and_filename('..//datasets//20news-bydate//20news-bydate-train')
	data_reader.collect_data()
	data_reader.save_data('data_processed.txt')

	# data_reader.read_topic_and_filename('..//dataset')
	# data_reader.collect_data()
	# data_reader.save_data('data_processed_gmm.txt')

