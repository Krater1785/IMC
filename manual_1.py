import numpy as np

M = np.array([[1,1.45,0.52,0.72],
              [0.7,1,0.31,0.48],
              [1.95,3.1,1,1.49],
              [1.34,1.98,0.64, 1]])

combinations = []
for i in range(0,4):
    for j in range(0,4):
        for k in range(0,4):
            for l in range(0,4):
                pnl = M[3, i] * M[i, j] * M[j, k]* M[k, l] * M[l, 3]
                combinations.append([i, j, k, l, pnl])

combinations.sort(key=lambda x: x[4], reverse=True)
print(combinations[:10])