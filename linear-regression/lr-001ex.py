import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

table = {
    'hours(x)':[2, 3, 4, 5],
    'score(y)':[25,50,42,61]
}
df = pd.DataFrame(table)
print(df)

nptable = np.array([[2, 3, 4, 5], [25, 50, 42, 61]])

plt.title('yee')
plt.xlabel('hours')
plt.ylabel('score')
plt.grid(True)
plt.scatter(nptable[0], nptable[1])
plt.show()

