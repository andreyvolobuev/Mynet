from mynet import Value
from mynet import relu


def test_rmul():
    x = Value(8)
    y = Value(3)
    z = 4-x*y
    z.backward()
    assert x.grad.data == -3
    assert y.grad.data == -8
    assert z.data == -20


def test_mul():
    x = Value(8)
    y = Value(3)
    z = x*y-4
    z.backward()
    assert x.grad.data == 3
    assert y.grad.data == 8
    assert z.data == 20


def test_pow():
    x = Value(5)
    z = 4*x**3
    z.backward()
    assert x.grad.data == 300
    assert z.data == 500


def test_div():
    x = Value(3)
    y = Value(4)
    f = (4*x)/y
    f.backward()
    assert x.grad.data == 1
    assert y.grad.data == -0.75
    assert f.data == 3


def test_complicated():
    x = Value(2)
    y = Value(3)
    z = Value(4)
    f = 2*(x**2+2*y)/z
    f.backward()
    assert x.grad.data == 2
    assert y.grad.data == 1
    assert z.grad.data == -1.25
    assert f.data == 5


def test_super_complicated():
    x = Value(-2)
    y = Value(3.5)
    z = Value(-4.6)
    f = 3.3*(x**3 + 20/y + 4*x**2) - (y/2/x + z)/x + 1
    f.backward()
    assert x.grad.data == -14.787500000000001
    assert y.grad.data == -5.512755102040816
    assert z.grad.data == 0.5
    assert f.data == 43.519642857142856


def test_root():
    x = Value(9)
    f = x.root()
    f.backward()
    assert x.grad.data == 0.16666666666666666
    assert f.data == 3

    x = Value(27)
    f = x.root(3)
    f.backward()
    assert x.grad.data == 0.03703703703703702
    assert f.data == 3


def test_rdiv():
    x = Value(30)
    x /= 3
    assert x.data == 10


def test_exp():
    x = Value(2)
    y = x.exp()
    y.backward()
    assert x.grad.data == 7.3890560989306495
    assert y.data == 7.3890560989306495

    x = Value(2)
    y = x.exp() + 2*x
    y.backward()
    assert x.grad.data == 9.389056098930649
    assert y.data == 11.389056098930649

    x = Value(3)
    y = Value(4)
    z = Value(4)
    assert x < y
    assert 2*x > y
    assert 3*y == 4*x
    assert y == z
    assert y is not z


def test_relu():
    x = Value(3)
    y = Value(-2)
    z = relu([x * y])
    z[0].backward()
    assert x.grad.data == 0
    assert y.grad.data == 0
    assert z[0].data == 0

    x = Value(5)
    y = Value(-2)
    z = relu([x + y])
    z[0].backward()
    assert x.grad.data == 1
    assert y.grad.data == 1
    assert z[0].data == 3


def test_log():
    x = Value(6)
    y = x.log()
    y.backward()
    assert x.grad.data == 0.16666666666666666
    assert y.data == 1.791759469228055


def test_softmax():
    x = Value(6)
    y = Value(3)
    z = Value(1)
    inputs_ = [x, y, z]
    exp = [(i-max(inputs_)).exp() for i in inputs_]
    softmax = [i / sum(exp) for i in exp]
    a1, a2, a3 = exp
    assert a1.data == 1.0
    assert a2.data == 0.04978706836786395
    assert a3.data == 0.006737946999085469
    b1, b2, b3 = softmax
    assert b1.data == 0.9464991225528936
    assert b2.data == 0.047123416524664154
    assert b3.data == 0.0063774609224422985
    softmax[0].backward()
    assert x.grad.data == 0.05063853355949614
    assert y.grad.data == -0.044602272392289144
    assert z.grad.data == -0.006036261167207002
