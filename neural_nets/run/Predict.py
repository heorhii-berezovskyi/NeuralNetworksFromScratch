import argparse

import cv2
import matplotlib.pyplot as plt
import numpy as np

from neural_nets.run.ModelSelector import ModelSelector


def predict(args):
    train_model, model_forward_run, train_mean = ModelSelector().select(args.model).load(path=args.snapshot_from)
    print('Loaded model from ' + args.snapshot_from)

    test_model = train_model.to_test(model_forward_run)

    img = cv2.imread(args.image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (args.image_size, args.image_size))
    img = img.astype(float)
    img = 255. - img

    img = np.reshape(img, (args.num_of_channels, args.image_size, args.image_size))
    img -= train_mean
    plt.imshow(img[0], cmap='gray')
    plt.show()
    print(test_model.predict(img))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Trains specified model from scratch with specified parameters.')
    parser.add_argument("--snapshot_from", type=str, help="Path to snapshot to load model.",
                        default=r'C:\Users\heorhii.berezovskyi\Documents\mnist-in-csv\model\convnet300.npz')
    parser.add_argument("--model", type=str, help="Model name.", default='convnet')
    parser.add_argument("--image_path", type=str, help="Path to image to predict.",
                        default=r'C:\Users\heorhii.berezovskyi\Documents\image22.jpg')

    parser.add_argument("--num_of_channels", type=int, help="Number of channels in a single image.", default=1)
    parser.add_argument("--image_size", type=int, help="Dimension of a square image.", default=28)

    _args = parser.parse_args()
    predict(_args)
