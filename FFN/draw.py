import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv('./result/result.csv')




x = [i for i in range(1, 11)]
# 绘制数据表
plt.figure(figsize=(8, 6))

for i in range(6):
    plt.plot(x,df.iloc[:,i],label=i)

plt.legend()
plt.title('multi batchsize')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.show()
