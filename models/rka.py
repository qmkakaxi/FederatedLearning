import six
import torch
from torch.autograd import grad
from  models import utility
import torch.nn.functional as F


def hessian(model,train_set,id,gpu=-1):

    #设置cpu和gpu
    device=torch.device('cuda:{}'.format(gpu) if torch.cuda.is_available() and gpu != -1 else 'cpu')

    # 计算出抽样得到的该元素的具体位置
    id1 = 0
    id2 = 0
    for i in range(len(list(model.parameters()))):
        parameters=list(model.parameters())[i]
        temp = 1
        for j in parameters.size():
            temp *= j
        if id <= temp:
            id1 = i
            id2 = id-1
            break
        else:
            id = id - temp

    second_grad = [torch.zeros(list(model.parameters())[i].size()).to(device) for i in range(len(list(model.parameters())))]
    n=len(train_set.dataset)
    for i in utility.create_progressbar(n, desc='hessian', start=0):

        data, target = train_set.dataset[i]
        data = train_set.collate_fn([data]).to(device)
        target = train_set.collate_fn([target]).to(device)
        output=model(data)
        loss = F.nll_loss(output, target, weight=None, reduction='mean')

        #first_grad为loss对mdele.parameters()的一阶导，type为tuple
        first_grad=grad(loss,list(model.parameters()),create_graph=True)

        """要求hessian矩阵，需要first_grad的每一个元素对model.parameters()求导"""
            #这里为节省内存，每次只求一个first_grad的一个元素对model.parameters()的导数

        parameters=first_grad[id1]
        size=1
        for j in parameters.size():
            size *= j
        # 将参数矩阵转化为向量,x中的元素和i共享内存
        x=parameters.view([size])
        temp=grad(x[id2],list(model.parameters()))
        second_grad=[a+b for a,b in six.moves.zip(second_grad,temp)]

    second_grad=[i/n for i in second_grad]

    return second_grad



def rka(x,a,b):

    stest=[]
    for i,j,k in six.moves.zip(x,a,b):
        temp=k-i*j
        square=torch.sum(j*j)
        stest.append(i+(temp/square)*j)
    return stest