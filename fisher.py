
# 开发环境：win10
# 作者：林自健
# 博客：Lzjusc2017.github.io
# 时间：2019/10/30
# 开发语言：python


from sklearn import datasets
from mxnet import nd

# sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)


'''
    函数功能：加载数据
'''
def get_dataset():
    iris = datasets.load_iris() #显示全部数据，没有显示标签
    return iris


'''
    函数功能：求均值
'''


def get_avg(mavg1,mavg2,iris,class_kind):
    # 40个训练集,10个测试集

    for i in range(4):
        avg1 = 0
        avg2 = 0
        for j in range(class_kind[0]*50,class_kind[0]*50+40):
            avg1 = avg1 + iris.data[j][i]
        for j in range(class_kind[1]*50,class_kind[1]*50+40):
            avg2 = avg2 + iris.data[j][i]
        mavg1.append(avg1/40)
        mavg2.append(avg2/40)
    print("avg1 is " + str(mavg1))
    print("avg2 is " + str(mavg2))


'''
    函数功能：求类内离差阵
'''
def get_sw(avg1,avg2,iris,class_kind):
    # 得到类内离差阵
    feature1 = nd.zeros((4, 4))  # 4个特征
    feature2 = nd.zeros((4, 4))  # 4个特征
    for count in range(2):
        #for i in range(count*50,count*50+40):
        for i in range(class_kind[count]*50,class_kind[count]*50+40):
            temp = nd.zeros((4, 1))
            for j in range(4):
                if count==0:
                    temp[j] = (iris.data[i][j] - avg1[j])
                else:
                    temp[j] = (iris.data[i][j] - avg2[j])
            #print(temp)
            if count==0:
                feature1 = feature1 + nd.dot(temp, temp.T)
            else:
                feature2 = feature2 + nd.dot(temp, temp.T)

    feature1 = feature1/39          # 除以总数-1
    feature2 = feature2 /39
    feature = feature1 + feature2   # 总特征sw
    return feature,feature2,feature


'''
    函数功能：求矩阵的逆
'''
def Gauss_Jordan(feature):
    # 求逆矩阵
    n = 4
    mat = nd.zeros((n,2*n))
    matv = nd.zeros((n,n))

    # 矩阵扩展
    for i in range(n):
        for j in range(n):
            mat[i][j] = feature[i][j]
    for i in range(n):
        mat[i][i+n] = 1.0
    for k in range(n):
        d = abs(mat[k][k].asscalar())
        j = k

        # 按列选择
        for i in range(k+1,n):
            if (abs(mat[i][k].asscalar())>d):
                d = mat[i][k].asscalar()
                j = i
        if (j!=k):
            # 交换
            for l in range(2*n):
                d = mat[j][l].asscalar()
                mat[j][l] = mat[k][l].asscalar()
                mat[k][l] = d
        # 交换好了
        for j in range(k+1,2*n):
            mat[k][j] = (mat[k][j].asscalar())/(mat[k][k].asscalar())
        for i in range(n):
            if i==k:
                continue
            for j in range(k+1,2*n):
                #print("i is "+str(i) + " j is "+ str(j) + " k is "+str(k))
                #sleep(50)
                mat[i][j] = mat[i][j].asscalar() - (mat[i][k].asscalar())*(mat[k][j].asscalar())
    for i in range(n):
        for j in range(n):
            matv[i][j] = mat[i][j+n]
    return matv


'''
    函数功能：求矩阵的u值
'''
def get_u(avg1,avg2,feature):
    n = 4
    m = nd.zeros((n,1))    #
    for i in range(n):
        m[i] = (avg1[i]-avg2[i])
    mat = nd.dot(feature,m)
    return mat


'''
    函数功能：求矩阵逆的测试
'''
def Test():
    test = nd.array([[1, 3, 1], [2, 1, 1], [2, 2, 1]])
    matv = Gauss_Jordan(test)
    print(matv)


'''
    函数功能：测试
'''
def TestSimple(m,iris,mat_u,class_kind):
    C1_Test = iris.data[(class_kind[0]*50+40):(class_kind[0]*50+50)]
    C2_Test = iris.data[(class_kind[1]*50+40):(class_kind[1]*50+50)]
    print(C2_Test)
    acc1 = 0
    acc2 = 0
    for i in range(10):
        Tm1 = 0
        Tm2 = 0
        for j in range(4):
            Tm1 = Tm1 + mat_u[j] * C1_Test[i][j]
            Tm2 = Tm2 + mat_u[j] * C2_Test[i][j]
        Tm1 = Tm1.asscalar()
        Tm2 = Tm2.asscalar()
        print("Tm1 is " + str(Tm1) + ",m is "+ str(m))
        print("Tm2 is " + str(Tm1) + ",m is " + str(m))
        if Tm1>=m:
            acc1 = acc1 +1
        if Tm2<=m:
            acc2 = acc2 + 1
    print("类"+str(class_kind[0]) + "的准确率是" + str((acc1/10)*100) + "%")
    print("类"+str(class_kind[1]) + "的准确率是" + str((acc2 / 10)*100) + "%")
if __name__ == '__main__':
    avg1 = []    #   第一类的均值
    avg2 = []   #   第二类的均值
    iris =get_dataset() #得到鸢尾花的数据

    class_kind = [1,2]
    get_avg(avg1,avg2,iris,class_kind) # 求平均值
    feature1,feature2,feature = get_sw(avg1,avg2, iris,class_kind) #得到sw
    matv = Gauss_Jordan(feature)    #得到矩阵的逆
    mat_u = get_u(avg1,avg2,feature)    # 得到u
    m1 = 0  #标量均值
    m2 = 0
    for i in range(4):
        m1 += mat_u[i].asscalar() * avg1[i]
        m2 += mat_u[i].asscalar() * avg2[i]
    m = (m1+m2)/2
    X = nd.zeros((40,1))
    X2 = nd.zeros((40,1))
    for i in range(40):
        for j in range(4):
            X[i] += mat_u[j].asscalar()*iris.data[i][j]
    for i in range(40):
        for j in range(4):
            X2[i] += mat_u[j].asscalar() * iris.data[i+50][j]
    # 每一个数据的标量值
    TestSimple(m,iris,mat_u,class_kind)