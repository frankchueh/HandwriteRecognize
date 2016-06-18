import mnist_loader
from sklearn import svm
import network, numpy

class SVMRecognize(object):
    """Hand writing recognization by using sklearn(module) SVM """
    def __init__(self, training_num):
        self.classification = None
        self.init_data(training_num)

    def init_data(self, training_num):
        training_data, validation_data, test_data = mnist_loader.load_data()
        # train
        classification = svm.SVC()
        classification.fit(training_data[0][:training_num], training_data[1][:training_num])
        self.classification = classification

    def convert_to_array(self, image):
        from PIL import Image
        img = Image.open(image).convert('L').resize((28, 28))
        img_array = list(img.getdata())
        for index in range(0, len(img_array)):
            # In the PIL image, while=255 and black=0, it's oppsite of our training data.
            result = 1 - float(img_array[index]) / 255
            img_array[index] = result
        return img_array

    def recognize(self, image):
        img_array = self.convert_to_array(image)
        return self.classification.predict(img_array)

class NetworkRecognize(object):
    """Hand writing recognization by using Neural Networks"""
    def __init__(self, learning_rate=5.0, epochs_time=10):
        self.net = network.Network([784, 30, 10])
        training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
        self.net.SGD(training_data, epochs_time, 10, learning_rate)

    def convert_to_array(self, image):
        from PIL import Image
        img = Image.open(image).convert('L').resize((28, 28))
        img_array = list(img.getdata())
        for index in range(0, len(img_array)):
            result = 1 - float(img_array[index]) / 255
            img_array[index] = result
        return numpy.reshape(img_array, (784, 1))

    def recognize(self, image):
        img_array = self.convert_to_array(image)
        return numpy.argmax(self.net.feedforward(img_array))

if __name__ == "__main__":
    svm = SVMRecognize(10000)
    nr = NetworkRecognize(epochs_time=1)
    print svm.recognize('../TestImage/0.png')[0]
    print nr.recognize('../TestImage/1.png')
