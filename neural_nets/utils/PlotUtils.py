import matplotlib.pyplot as plt


def plot(losses: list, test_accuracies: list):
    plt.figure(1)

    plt.subplot(211)

    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.plot(losses)

    plt.subplot(212)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.plot(test_accuracies)

    plt.show()
