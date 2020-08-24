from perceptron import PerceptronClassifier
from arff import Arff
import numpy as np
from sklearn import linear_model

def main():

    myArff = Arff()
    myArff.load_arff('linsep2nonorigin.arff')
    # myArff.load_arff('data_banknote_authentication.arff')
   # myArff.load_arff('dataset_a.arff')
    # myArff.load_arff('voting.arff')


    data = np.array(myArff.data)
    X = data[:, 0: data.shape[1] - 1]
    y = data[:, data.shape[1] - 1]

    p = PerceptronClassifier()

    # p.fit(X, y, deterministic=10, shuffle=False)
    # Accuracy = p.score(X, y)

    X, y = p._shuffle_data(X, y)
    seventy_p_mark = int(data.shape[0] * .7)

    X_training = X[:seventy_p_mark]
    y_training = y[:seventy_p_mark]

    X_testing = X[seventy_p_mark:]
    y_testing = X[seventy_p_mark:]

    p.fit(X_training, y_training, deterministic=None, shuffle=True)




    # print("Accuray = [{:.2f}]".format(Accuracy))
    # print("Final Weights =", p.get_weights())
    # cool = p.get_weights()
    # print('hi')

    reg = linear_model.Perceptron()
    reg.fit(X, y)
    print(reg.coef_)
    print(reg.score(X, y))
    print('cool')



if __name__ == "__main__":
    main()