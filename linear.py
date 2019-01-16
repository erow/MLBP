import numpy as np

# init data
n = 1000
k = 2  # Todo not works
X = np.random.rand(n, k)*10   # x belongs to [0,10]
b = 0.1
theta = np.random.rand(1,k)
Y = np.dot(X, np.transpose(theta)) + b


# init parameters
tt = np.random.rand(1,k)
theta0 = 0

# gradient decrease
a = 0.0005*16
loss = 0
it = 0
while it<30000:
    diff = np.dot(X, np.transpose(tt)) + theta0 - Y
    tt -= a * np.sum(diff * X )/n
    theta0 -= a * np.sum(diff)/n
    loss1 = np.sum(np.square(diff))/2/n
    print("theta0:{},d:{}, loss:{}".format(theta0,a * np.sum(diff)/n, loss1))
    loss = loss1
    it += 1

print("real:{},{}".format(theta,b))
print("parameters:{},{}".format(tt,theta0))
