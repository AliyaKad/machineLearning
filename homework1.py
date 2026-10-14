import numpy as np
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

class NET:

    def __init__(self, n_input=28 * 28, n_hidden=128, n_output=10, lr=0.1, epochs=20):
        self.weight_input_hidden = np.random.randn(n_input, n_hidden) * np.sqrt(2.0 / n_input)
        self.bias_hidden = np.zeros((1, n_hidden))

        self.weight_hidden_output = np.random.randn(n_hidden, n_output) * np.sqrt(2.0 / n_hidden)
        self.bias_output = np.zeros((1, n_output))

        self.lr = lr
        self.epochs = epochs

        self.hidden = None
        self.hidden_raw = None
        self.output = None

    def relu(self, layer):
        return np.maximum(0, layer)

    def relu_derivative(self, layer):
        return (layer > 0).astype(layer.dtype)

    def softmax(self, layer):
        layer = layer - np.max(layer, axis=1, keepdims=True)
        exp_layer = np.exp(layer)
        return exp_layer / np.sum(exp_layer, axis=1, keepdims=True)

    def cross_entropy(self, probs, y_one_hot):
        eps = 1e-12
        return -np.mean(np.sum(y_one_hot * np.log(probs + eps), axis=1))

    def forward_propagation(self, X):

        self.hidden_raw = X @ self.weight_input_hidden + self.bias_hidden
        self.hidden = self.relu(self.hidden_raw)

        z_output = self.hidden @ self.weight_hidden_output + self.bias_output
        self.output = self.softmax(z_output)
        return self.output

    def back_propagation(self, X, y_one_hot):
        N = X.shape[0]

        d_output = (self.output - y_one_hot) / N

        dW_hidden_output = self.hidden.T @ d_output
        db_output = np.sum(d_output, axis=0, keepdims=True)

        d_hidden = d_output @ self.weight_hidden_output.T
        d_hidden_raw = d_hidden * self.relu_derivative(self.hidden_raw)

        dW_input_hidden = X.T @ d_hidden_raw
        db_hidden = np.sum(d_hidden_raw, axis=0, keepdims=True)

        self.weight_hidden_output -= self.lr * dW_hidden_output
        self.bias_output -= self.lr * db_output
        self.weight_input_hidden -= self.lr * dW_input_hidden
        self.bias_hidden -= self.lr * db_hidden

    def one_hot_encodding(self, y, n_classes=10):
        y = np.asarray(y).astype(int)
        one_hot = np.zeros((y.size, n_classes))
        one_hot[np.arange(y.size), y] = 1
        return one_hot

    def fit(self, X, y, X_test=None, y_test=None, batch_size=128):
        X = X.reshape(X.shape[0], -1).astype(np.float32) / 255.0
        y_one_hot = self.one_hot_encodding(y)

        if X_test is not None:
            X_test = X_test.reshape(X_test.shape[0], -1).astype(np.float32) / 255.0
            y_test_one_hot = self.one_hot_encodding(y_test)

        accuracies = []
        losses = []
        test_accuracies = []
        test_losses = []

        n = X.shape[0]

        for e in range(self.epochs):

            order = np.random.permutation(n)
            X_shuffled = X[order]
            y_shuffled = y_one_hot[order]

            for start in range(0, n, batch_size):
                X_batch = X_shuffled[start:start + batch_size]
                y_batch = y_shuffled[start:start + batch_size]

                self.forward_propagation(X_batch)
                self.back_propagation(X_batch, y_batch)

            probs_train = self.forward_propagation(X)
            loss_train = self.cross_entropy(probs_train, y_one_hot)
            acc_train = np.mean(np.argmax(probs_train, axis=1) == y)

            losses.append(loss_train)
            accuracies.append(acc_train)

            msg = f"Эпоха {e + 1:2d}/{self.epochs} | loss = {loss_train:.4f} | acc = {acc_train:.4f}"

            if X_test is not None:
                probs_test = self.forward_propagation(X_test)
                loss_test = self.cross_entropy(probs_test, y_test_one_hot)
                acc_test = np.mean(np.argmax(probs_test, axis=1) == y_test)

                test_losses.append(loss_test)
                test_accuracies.append(acc_test)

                msg += f" | test_loss = {loss_test:.4f} | test_acc = {acc_test:.4f}"

            print(msg)

        return losses, accuracies, test_losses, test_accuracies

    def predict(self, X):
        X = X.reshape(X.shape[0], -1).astype(np.float32) / 255.0
        probs = self.forward_propagation(X)
        return np.argmax(probs, axis=1)


if __name__ == '__main__':
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    nn = NET(n_hidden=128, lr=0.1, epochs=20)
    losses, accuracies, test_losses, test_accuracies = nn.fit(
        X_train, y_train, X_test=X_test, y_test=y_test, batch_size=128
    )

    epochs_axis = np.arange(1, len(losses) + 1)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_axis, losses, label='train loss')
    plt.plot(epochs_axis, test_losses, label='test loss')
    plt.xlabel('Эпоха')
    plt.ylabel('Cross-Entropy Loss')
    plt.title('Loss')
    plt.legend()
    plt.grid(alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(epochs_axis, accuracies, label='train acc')
    plt.plot(epochs_axis, test_accuracies, label='test acc')
    plt.xlabel('Эпоха')
    plt.ylabel('Accuracy')
    plt.title('Accuracy')
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()
