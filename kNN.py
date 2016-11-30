from scipy.spatial import KDTree
import numpy as np
import networkx as nx

from synthetic_classifier_data import *

class kNN():
	def __init__(self, x_train, y_train):

		self.set_data(x_train, y_train)

	def set_data(self, x_train, y_train):
		self.x_train = x_train
		self.y_train = y_train
		self.n = y_train.shape[0]

		self.kd_tree = KDTree(self.x_train)

	def predict(self, x, k=3):
		predictions = np.zeros(x.shape[0])

		for i in range(x.shape[0]):
			d_i, n_i = self.kd_tree.query(x[i, :], k=k)

			predictions[i] = np.sum(.5 + .5 * self.y_train[n_i])/float(k)

		return predictions


if __name__ == '__main__':
	from pylab import *

	data = SyntheticClassifierData(25, 20)
	nn = kNN(data.x_train, data.y_train)

	ks = [1, 3, 5, 10]
	for j, k in enumerate(ks):
		hat_y_train = nn.predict(data.x_train, k=k)
		hat_y_test = nn.predict(data.x_test, k=k)

		subplot(len(ks), 2, 2*j + 1)
		plot(np.arrange(data.x_train.shape[0]), .5 + .5 * data.y_train, 'bo')
		plot(np.arrange(data.x_train.shape[0]), hat_y_train, 'rx')

		ylim([-.1, 1.1])
		ylabel("K=%s" % k)
		if j == 0:
			title("Training set reconstructions")

		subplot(len(ks), 2, 2*j + 2)
		plot(np.arrange(data.x_test.shape[0]), .5 + .5 * data.y_test, 'yo')
		plot(np.arrange(data.x_test.shape[0]), hat_y_test, 'rx')

		ylim([-.1, 1.1])
		if j == 0:
			title("Test set predictions")

	show()

