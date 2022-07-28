from value import Value

if __name__ == '__main__':
	x = Value(8)
	y = Value(3)
	z = 4-x*y
	z.backward()
	assert x.grad.data == -3
	assert y.grad.data == -8
	assert z.data == -20

	x = Value(8)
	y = Value(3)
	z = x*y-4
	z.backward()
	assert x.grad.data == 3
	assert y.grad.data == 8
	assert z.data == 20

	x = Value(5)
	z = 4*x**3
	z.backward()
	assert x.grad.data == 300
	assert z.data == 500

	x = Value(3)
	y = Value(4)
	f = (4*x)/y
	f.backward()
	assert x.grad.data == 1
	assert y.grad.data == -0.75
	assert f.data == 3

	x = Value(2)
	y = Value(3)
	z = Value(4)
	f = 2*(x**2+2*y)/z
	f.backward()
	assert x.grad.data == 2
	assert y.grad.data == 1
	assert z.grad.data == -1.25
	assert f.data == 5

	x = Value(-2)
	y = Value(3.5)
	z = Value(-4.6)
	f = 3.3*(x**3 + 20/y + 4*x**2) - (y/2/x + z)/x + 1
	f.backward()
	assert x.grad.data == -14.787500000000001
	assert y.grad.data == -5.512755102040816
	assert z.grad.data == 0.5
	assert f.data == 43.519642857142856

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

	x = Value(30)
	x /= 3
	assert x.data == 10

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

	x = Value(3)
	y = Value(-2)
	z = (x * y).relu()
	z.backward()
	assert x.grad.data == 0
	assert y.grad.data == 0
	assert z.data == 0

	x = Value(5)
	y = Value(-2)
	z = (x + y).relu()
	z.backward()
	assert x.grad.data == 1
	assert y.grad.data == 1
	assert z.data == 3

	x = Value(6)
	y = x.log()
	y.backward()
	assert x.grad.data == 0.16666666666666666
	assert y.data == 1.791759469228055

	print('Looks good!')
