from scipy.optimize import minimize

fun = lambda x: (0.25*x[0]*x[0]) + (x[1]*x[1])
bnds = ((0, None), (0, None))
x0 = [0.5,0]
cons = ({'type': 'eq', 'fun': lambda x:  5 - x[0] - x[1]},
         {'type': 'ineq', 'fun': lambda x: x[0] + 0.2 * x[1] - 3})
res = minimize(fun, x0, constraints=cons)
print (res.x)