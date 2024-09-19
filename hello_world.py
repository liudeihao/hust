import numpy as np
from matplotlib import pyplot as plt

import matplotlib.pyplot as plt
from matplotlib import font_manager

fonts = [font.name for font in font_manager.fontManager.ttflist]

fonts.sort()

for font in fonts:
    print(font)

import matplotlib
# 使用matplotlib的字体管理器注册字体
plt.rcParams['font.sans-serif'] = ['KaiTi'] # 设置字体为霞鹜文楷
plt.rcParams['font.family']='sans-serif' # 设置字体族
plt.rcParams['axes.unicode_minus']=False # 正确显示负号

t = np.arange(0, 256, 1)

f = np.sin(t)

plt.plot(t, f)
plt.title("正弦函数")

plt.xlabel("时间t")
plt.ylabel("幅度f")
plt.show()