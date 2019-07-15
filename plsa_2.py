import numpy as np 
import time

def prepare_data(path):
	# K = 20 #number_of_topics
	K = 5

	with open('vocab.txt') as f:
		lines = f.read().splitlines() #number_of_words_in_vocab

	V = len(lines)

	with open(path) as f:
		lines = f.read().splitlines()

	D = len(lines) #number_of_docs

	doc_count = []

	for line in lines:

		word_count = dict()
		words = line.split('<fff>')[2].split()

		for word in words:
			word_count[int(word.split(':')[0])] = int(word.split(':')[1])

		doc_count.append(word_count) #doc_count có dạng: [{10:2, 12:3}, {11:3, 13:1, 12:4}, {4:2, 2:4, 1:2, 5:2}]

	return K, V, D, doc_count

class PLSA:
	def __init__(self, K, V, D, doc_count):

		self._K = K #number_of_topics
		self._V = V #number_of_words_in_vocab
		self._D = D #number_of_docs
		self._doc_count = doc_count

		#X = wordcountPerdoc
		self._X = np.zeros(shape = (D, V))
		for i, d in enumerate(self._doc_count):
			for key, value in d.items():
				self._X[i][key] = value

		self._theta = np.random.random(size = (D, K))
		self._beta = np.random.random(size = (K, V))
		self._T = np.zeros(shape = (D, V, K))

	def normalize(self):
		for d in range(self._D):

			normalization = np.sum(self._theta[d, :])

			for k in range(self._K):
				self._theta[d, k] /= normalization

		for k in range(self._K):

			normalization = np.sum(self._beta[k, :])

			for v in range(self._V):
				self._beta[k, v] /= normalization

	def E_step(self):
		for d in range(self._D):
			for v in range(self._V):

				denominator = 0

				for k in range(self._K):
					self._T[d, v, k] = self._theta[d, k] * self._beta[k, v]
					denominator += self._T[d, v, k]

				if denominator == 0:
					for k in range(self._K):
						self._T[d, v, k] = 0;
				else:
					for k in range(self._K):
						self._T[d, v, k] /= denominator;

	def M_step(self):
		#Tinh theta
		for d in range(self._D):
			for k in range(self._K):

				self._theta[d, k] = 0
				denominator = 0

				for v in range(self._V):
					self._theta[d, k] += self._X[d, v] * self._T[d, v, k]
					denominator += self._X[d, v]


				if denominator == 0:
					self._theta[d, k] = 1.0 / K
				else:
					self._theta[d, k] /= denominator

		#Tinh beta
		for k in range(self._K):

			denominator = 0
			print('K: ', k)

			for v in range(self._V):

				self._beta[k, v] = 0

				for d in range(self._D):
					self._beta[k, v] += self._X[d, v] * self._T[d, v, k]

				denominator += self._beta[k, v]

			if denominator == 0:
				for v in range(self._V):
					self._beta[k, v] = 1.0 / self._V
			else:
				for v in range(self._V):
					self._beta[k, v] /= denominator

	def LogLikelihood(self):
		loglikelihood = 0

		for d in range(self._D):
			for v in range(self._V):
				tmp = 0
				for k in range(self._K):
					tmp += self._theta[d, k] * self._beta[k, v]

				if tmp > 0:
					loglikelihood += self._X[d, v] * np.log(tmp)
		return loglikelihood

if __name__ == '__main__':
	K, V, D, doc_count = prepare_data('data_vectorizer.txt')
	
	print('K, V, D', K, V, D)
	plsa = PLSA(K, V, D, doc_count)
	plsa.normalize()

	L = 0

	for i in range(100):
		plsa.E_step()
		print('E_step lan ', i+1)
		plsa.M_step()
		print('M_step lan ', i+1)
		print("[", time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())), "] After the", i+1, "'s iteration  ", )
		L_new = plsa.LogLikelihood()
		print('Log Likelihood: ', L_new)
		print('Loss: ', abs(L - L_new))
		if abs(L - L_new) <= 10e-4:
			break
		else:
			L = L_new

	print('Theta: ', plsa._theta)
	print('Beta: ', plsa._beta)






		