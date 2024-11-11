import time
import numpy as np
import pandas as pd
import sys  # library that allows into from operating system
from datetime import datetime
from gillespy2 import Model
from gillespy2.core.results import Results

from model import SomeModel, ParameterValues

startTime = time.time()
print(str(datetime.now()))
print(sys.argv)

A1 = float(sys.argv[1])
cellNo = eval(sys.argv[2])
tp = eval(sys.argv[3])

filename = "testData4_cn" + str(cellNo) + ".csv"

if tp == 0:
    header = pd.read_csv("initBook.csv")
    header.to_csv(filename)

initReceptors = [3500, 3500, 7000]

sys.path[:0] = [".."]

model: Model = SomeModel(
    ParameterValues(A1=A1, init=pd.read_csv(filename)),
    timespan=np.linspace(1, 100, 100),
)
results: Results = model.run()

results = results.to_array()
print(results.shape)
results1 = results.reshape(-1, results.shape[2])
df_results = pd.DataFrame(results1)
df_results.to_csv(filename, header=False, mode="a")

print("The script took {0} second !".format(time.time() - startTime))
