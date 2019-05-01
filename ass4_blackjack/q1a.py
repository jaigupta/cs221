q=[[0, 0] for _ in range(5)]
v=[0 for _ in range(5)]

r=[20, -5, -5, -5, 100]


for iteration in range(20):
    for i in range(1, 4):
        q[i][0] = 0.8*(r[i-1]+v[i-1]) + 0.2*(r[i+1] + v[i+1])
        q[i][1] = 0.3*(r[i-1]+v[i-1]) + 0.7*(r[i+1] + v[i+1])
    for i in range(1, 4):
        v[i] = max(q[i][0], q[i][1])

    print("Iteration:", iteration)
    print("q:", q)
    print("v", v)
