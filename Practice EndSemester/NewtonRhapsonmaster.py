import numpy as np
import numpy.linalg as la
#PS6
#problem 1
def newton2(f, g, j, x, tol):
    k = 0
    norm = (f(x)**2 + g(x)**2)**(1/2)
    while norm >= tol:
        x = x - la.pinv(j(x))@np.array([f(x), g(x)])
        norm = (f(x)**2 + g(x)**2)**(1/2)
        k = k + 1
    return x, k, norm
def f1a(x):
    return x[0]**(1/5) + x[1]**(1/5) - 2
def f2a(x):
    return x[0]**(1/10) + x[1]**(1/10) - 2
def Jq1a(x):
    return np.array([[(1/5)*x[0]**(-4/5), (1/5)*x[1]**(-4/5)], [(1/10)*x[0]**(-9/10), (1/10)*x[1]**(-9/10)]])
def f1b(x):
    return (x[0]**(1/5) + x[1]**(1/5))**5 - 32
def f2b(x):
    return (x[0]**(1/10) + x[1]**(2/5))**4 - 16
def Jq1b(x):
    return np.array([[5 * (( x[0] ** (1/5) + x[1] ** (1/5) ) ** 4) * ((1/5) * x[0] ** (-4/5)),5 * (( x[0] ** (1/5) + x[1] ** (1/5) ) ** 4) * ((1/5) * x[1] ** (-4/5)),],[4 * (( x[0] ** (1/10) + x[1] ** (2/5) ) ** 3) * ((1/10) * x[0] ** (-9/10)),4 * (( x[0] ** (1/10) + x[1] ** (2/5) ) ** 3) * ((2/5) * x[1] ** (-3/5)),]])
'''
Author's Note : when we start from the original function given to us, 
negative values of y will go in the f2 function which will result in 
python throwing error, Thiru sir might have made a mistake or Maybe 
we have to check for positivity of x and y at each iteration.
'''
#Problem 2
def j2(x, c):
    if c == 'A':
        return np.array([
            [-2 - 1/x[0], -1, -1],
            [-1, -2 - 1/x[1], -1],
            [-1, -1, -2 - 1/x[2]]
        ])
    elif c == 'B':
        return np.array([
            [-2*(x[0] + x[1] + x[2]) - 4*x[0], -2*(x[0] + x[1] + x[2]), -2*(x[0] + x[1] + x[2])],
            [-2*(x[0] + x[1] + x[2]), -2*(x[0] + x[1] + x[2]) - 4*x[1], -2*(x[0] + x[1] + x[2])],
            [-2*(x[0] + x[1] + x[2]), -2*(x[0] + x[1] + x[2]), -2*(x[0] + x[1] + x[2]) - 4*x[2]]
        ])
    else:
        return np.array([
            [-2, -1, -1],
            [-1, -4, -1],
            [-1, -1, -2 -6*x[2]]
        ])
def fA(x):
    return [1 - x[0] - x[1] - x[2] - x[0] - np.log(x[0]) -  1, 1 - x[0] - x[1] - x[2] - x[1] - np.log(x[1]) -  1, 1 - x[0] - x[1] - x[2] - x[2] - np.log(x[2]) -  1]
def fB(x):
    return [1 - (x[0] + x[1] + x[2])**2 - 2*x[0]**2 - 1/2, 1 - (x[0] + x[1] + x[2])**2 - 2*x[1]**2 - 1/2, 1 - (x[0] + x[1] + x[2])**2 - 2*x[2]**2 - 1/2]
def fC(x):
    return [1 - x[0] - x[1] - x[2] - x[0] - 1, 1 - x[0] - x[1] - x[2] - x[1] - 2*x[1], 1 - x[0] - x[1] - x[2] - x[2] - 3*x[2]**2]
def newton3(f, c, j, x, tol):
    k = 0
    norm = (f(x)[0]**2 + f(x)[1]**2 + f(x)[2])**(1/2)
    while norm >= tol:
        x = x - la.pinv(j(x, c))@np.array([f(x)[0], f(x)[1], f(x)[2]])
        norm = (f(x)[0]**2 + f(x)[1]**2 + f(x)[2])**(1/2)
        k = k + 1
    return x, k, norm

def H(a):
    return np.array([[2, a], [a, 2]])
def grad(x, H):
    return H@x
def els(H, d):
    return (np.dot(d, d))/(d.T @ H @ d)
def descentWithExactLineSearch(H, x, tol):
    k = 0
    norm = la.norm(grad(x, H))
    while norm >= tol:
        d = -grad(x, H)
        alpha = els(H, d)
        x = x + alpha*d
        k += 1
        norm = la.norm(grad(x, H))
    return x, k
A = np.arange(-1.9, 2, 0.1)
# for a in A:
#     print(a, descentWithExactLineSearch(H(a), np.array([10, 10]), 1e-6))


def F(x):
    return x[0]*np.exp(-x[0]**2 - x[1]**2)
def G(x):
    return np.array([
        (1 - 2*x[0]**2)*np.exp(-x[0]**2 - x[1]**2),
        -2*x[1]*np.exp(-x[0]**2 - x[1]**2)
    ])
def btls(x):
    alpha = 1
    rho = 0.8
    c1 = 0.8
    g = G(x)
    while F(x - alpha*G(x)) - F(x) <= alpha*c1*(np.dot(G(x), -G(x))):
        alpha = rho*alpha
    return alpha
def solve(x):
    k = 0
    while la.norm(G(x)) >= 1e-6:
        d = -G(x)
        alpha = btls(x)
        x = x + alpha*d
        k += 1
    return x, k


#RosenBitchAss Block
def rosen(x):
    return 100*(x[1] - x[0]**2)**2 + (1 - x[0])**2
def rosenG(x):
    return np.array([200*(x[1] - x[0]**2)*(-2*x[0]) - 2*(1 - x[0]), 200*(x[1] - x[0]**2)])
def backTrackingLineSearch(x):
    alpha = 1
    rho = 0.8
    c1 = 0.8
    while rosen( x - alpha * rosenG(x) ) - rosen( x ) > c1*alpha*np.dot(rosenG(x), -rosenG(x)):
        alpha = rho*alpha
    return alpha
def rosenSolver(x):
    while la.norm( rosenG( x ) ) >= 1e-6  :
        d = - rosenG(x)
        alpha = backTrackingLineSearch(x)
        x = x + alpha*d
    return x
# print(rosenSolver(np.array([100, 100])))

'''
Author's Note : Problem set had rosenblock function wrong.
Also, please note that the values of rho, alpha can significantly affect your result.
Please beware of non-covergent loops multiplication by nans etc. if You get an error like that, consider changing intial point
or play around with values of c1, rho, alpha etc.
'''

#Problem 6, The head hurting, transpose shooting, Line searching shittiest of the shit shit taking a shit problem.
n = 1000
X = np.random.randint(-10,10,(n,n))
A = X@X.T + np.eye(n)
b = np.random.randint(-100, 100, (n,1))
L, V = la.eig(A)
x = np.zeros(n)
g = A@x + b
i = 0
while la.norm(g) >= 1e-6:
    if i >= n:
        break
    d = V[:, i]
    alpha = -(d.T @ g)/(d.T @ A @ d)
    x = x + alpha*d
    g = A@x + b.flatten()
    i += 1
    print(la.norm(A@x + b))

print(np.allclose(x.reshape(-1, 1), -la.inv(A)@b))

