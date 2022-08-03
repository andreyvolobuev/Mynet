import mynet
from utils.eval import evaluate_model, train_test_split, get_random_batch


class Model(mynet.Model):
    def __init__(self):
        self.l1 = mynet.Layer(4, 12, activation=mynet.relu)
        self.l2 = mynet.Layer(12, 24, activation=mynet.relu)
        self.l3 = mynet.Layer(24, 12, activation=mynet.relu)
        self.l4 = mynet.Layer(12, 3, activation=mynet.softmax)
        self.optim = mynet.Adam(self.parameters())


def test_model_evaluate():
    # This test takes time...
    flowers = []
    with open('tests/test_data/iris.data', 'r') as f:
        for line in f.readlines():
            slength, swidth, plength, pwidth, type_ = line.replace('\n', '').split(',')
            type_ = {
                'Iris-setosa': (1, 0, 0),
                'Iris-versicolor': (0, 1, 0),
                'Iris-virginica': (0, 0, 1)
            }.get(type_)
            flowers.append([float(slength), float(swidth), float(plength), float(pwidth), *type_])

    X_train, y_train, X_test, y_test = train_test_split(flowers, 4, 80)

    model = Model()#.load('tests/test_data/classifier_0.model')

    BATCH_SIZE = 8
    N_EPOCH = 2000

    for n in range(N_EPOCH):
        X, y = get_random_batch(BATCH_SIZE, X_train, y_train)

        model.optim.zero_grad()
        out = model.forward(X)
        loss = mynet.cross_entropy_loss(y, out)
        loss.backward()
        model.optim.step()

        print(f'EPOCH {n}, loss: {loss}')
        # model.save('tests/test_data/classifier_0.model')

    score = evaluate_model(model, mynet.cross_entropy_loss, X_test, y_test)
    assert score < 0.05
    print(f"TOTAL LOSS SCORE IS: {score}, LESS THEN 5% WHICH IS PRETTY GOOD")