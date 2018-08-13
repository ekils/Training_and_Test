### Training_and_Test
## Training and Test for glucose biocode  algorithm

```

本週是要實現如何使用python的machine leraning，上了coursera的課程主要是了解概念，至於實作，我是參考了莫煩的python教學
以下是網址：
youtube:
https://www.youtube.com/watch?v=YY7-VKXybjc&list=PLXO45tsB95cI7ZleLM5i3XXhhe9YmVrRO
github:
https://github.com/MorvanZhou/tutorials/tree/master/sklearnTUT
受益良多

看了下網路其他教學範例，大多是使用波士頓房地產做解說，然後做線性迴歸。不然就是做分類。
而我的狀況是希望能做多項式的回歸，網路上也是有找到多項式回歸的教學，不過要慢慢自己去建參數，是有點複雜
於是看到了python有： sklearn (science kit learning )的svm(Support Vecotr Machine) SVR（Suppert Vector Regression)演算法
基本上 SVR 就是在針對回歸作演算法處理。

該import的依樣先import：
%matplotlib notebook
from sklearn import datasets 
import matplotlib.pyplot as plt 
import numpy as np 
from sklearn.cross_validation import train_test_split 
from sklearn.svm import SVR 
from sklearn.cross_validation import cross_val_score 
import pickle
from math import*

1.from sklearn import datasets： http://scikit-learn.org/stable/modules/classes.html#module-sklearn.datasets  
                                                 數據庫，內建很多資料可以練習，也有內建很多function，比如datasets.make_regression ....等等
2.from sklearn.cross_validation import train_test_split ： 分類test set 與training set
3.from sklearn.svm import SVR : svr演算法
4.from sklearn.cross_validation import cross_val_score : 交叉驗證，之後有新的data再引入驗證，目前還沒用
5.import pickle:存擋 
6.from math import*：如果要用倒三角函數或是exp要引入

better_score =0
X = np.sort(5 * np.random.rand(500, 1), axis=0)
y = np.exp( -X**2 ) + np.random.normal(0, 0.05, X.shape)
y_= y. ravel()
for i in range(20):
    x_train, x_test, y_train, y_test = train_test_split(X, y_, test_size=0.3) 
    svr_poly = SVR(kernel='rbf') 
    x_test =sorted(x_test)
    y_test= sorted(y_test, reverse=True)
    y_poly= svr_poly.fit(x_train,y_train).predict(x_test) 
    score = svr_poly.score(x_test,y_test) 
    print score
    if better_score>score: 
        better_score = better_score 
    else: 
        better_score = score 
        fitting = svr_poly 
        xx=x_test 
        yy=y_test 

print ('Best R^2 is: {}'.format(better_score)) 


# plt.scatter(X,y,lw =1,marker='.',c= 'cornflowerblue')
# plt.scatter(x_test,y_test,lw= 0.1,marker='*',c ='r')
# plt.plot(x_test,y_poly,c='g',lw =2)
# plt.show()
xx = sorted(xx) 
yy = sorted(yy,reverse=True) 
yyy= fitting.fit(x_train,y_train).predict(xx) 
plt.scatter(X,y,lw =1,marker='.',c= 'cornflowerblue')
plt.scatter(xx,yy,lw= 0.1,marker='*',c ='r')
plt.plot( xx,yyy, color='g', lw=2) 
plt.legend() 
plt.show()

0.981262758171
0.985928238754
0.983921017322
0.978366081228
0.981761295843
0.983258298598
0.984401149668
0.975115055289
0.982420619817
0.984895618793
0.984486517169
0.98190445156
0.980037096507
0.975000835056
0.974193198206
0.986632990775
0.983566759576
0.984506261356
0.981129872633
0.979422542099
Best R^2 is: 0.986632990775
xx=x_test , yy=y_test 
為當我找到較好的 svr_poly 演算法函數時，丟進fitting ，當時所對應的 test set。
當一系列跑完，
fitting為最好的演算法，xx,yy為當時最好的test sample，
我就用演算法fitting去驗證xx 跑出來的結果 yyy 看是否接近yy

藍色.是全部data
紅色＊是最後最好的test set
綠色線是用fitting演算法以xx數據推測出yyy的數值所畫的線

由於x,y是我自己生成的序列，會在正負之間跳來跳去，所以要再用sort去把他們排序。
結果：

```


![GitHub Logo](https://github.com/ekils/Training_and_Test/blob/master/0.png)
