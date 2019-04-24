import argparse

import cv2
import matplotlib.pyplot as plt
import numpy as np

from neural_nets.dataset.DatasetLoader import DatasetLoader
from neural_nets.utils.DatasetProcessingUtils import remove_mean_on_train
from neural_nets.run.ModelSelector import ModelSelector


def predict(args):
    train_model, model_forward_run = ModelSelector().select(args.model).load(path=args.snapshot_from)
    print('Loaded model from ' + args.snapshot_from)

    test_model = train_model.to_test(model_forward_run)

    loader = DatasetLoader(directory=args.dataset_dir)
    train_labels, train_data = loader.load(dataset_name='fmnist_train.npy')

    train_data, train_mean = remove_mean_on_train(train_data=train_data)

    img = cv2.imread(args.image_path, 0)
    img = cv2.resize(img, (28, 28))
    img = img.astype(float)
    img = 255. - img
    img -= train_mean

    img = np.reshape(img, (1, 28, 28))
    plt.imshow(img[0], cmap='gray')
    plt.show()
    print(test_model.predict(img))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Trains specified model from scratch with specified parameters.')
    parser.add_argument("--dataset_dir", type=str, help="Path to data set.",
                        default=r'C:\Users\heorhii.berezovskyi\Documents\mnist-in-csv')
    parser.add_argument("--snapshot_from", type=str, help="Path to snapshot to load model.",
                        default=r'C:\Users\heorhii.berezovskyi\Documents\mnist-in-csv600.npz')
    parser.add_argument("--model", type=str, help="Model name.", default='convnet')
    parser.add_argument("--image_path", type=str, help="Path to image to predict.",
                        default=r'C:\Users\heorhii.berezovskyi\Documents\image32.jpg')

    _args = parser.parse_args()
    predict(_args)
