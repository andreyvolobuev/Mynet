import mynet


class Model(mynet.Model):
    def __init__(self):
        self.l1 = mynet.Layer(1, 3, activation=mynet.relu)
        self.l2 = mynet.Layer(3, 1)
        self.optim = mynet.GradientDecent(self.parameters(), lr=0.1)


if __name__ == '__main__':
    model = Model()
    X_train  = [[1.00], [0.00], [0.50], [0.05], [0.50], [0.45], [0.99], [0.55]]
    y_target = [[0.00], [0.00], [1.00], [0.00], [1.00], [0.99], [0.00], [1.00]]

    old_loss = None

    for i in range(1001):
        y_pred = model.forward(X_train)
        loss = mynet.mse_loss(y_pred, y_target)

        if i % 10 == 0 and old_loss:
            assert loss < old_loss
            print(f'# EPOCH: {i}, LOSS:', loss.data)

        loss.backward()
        model.optim.step()
        model.optim.zero_grad()
        old_loss = loss