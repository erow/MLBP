import numpy as np

# init data
n = 1000
k = 3  # fixed
X = np.random.rand(n, k)*10   # x belongs to [0,10]
b = 0.1
theta = np.random.rand(1,k)
Y = np.dot(X, np.transpose(theta)) + b


# init parameters
tt = np.random.rand(k,1)
theta0 = 0

# gradient decrease
a = 0.01
loss = 0
it = 0
while it < 3500:
    diff = np.dot(X, tt) + theta0 - Y  # nx1
    tt -= a/n * np.dot(np.transpose(X), diff)  # kx1
    theta0 -= a/n * np.sum(diff)
    loss1 = np.sum(np.square(diff))/2/n
    print("theta0:{},d:{}, loss:{}".format(theta0,a * np.sum(diff)/n, loss1))
    loss = loss1
    it += 1

print("real:{},{}".format(theta,b))
print("parameters:{},{}".format(np.transpose(tt),theta0))
