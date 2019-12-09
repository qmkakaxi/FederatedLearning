import six
import torch
import torch.distributed as dist
from models.influence import grad_z,stest
import models.utility as utility
import copy
import numpy as np
from torchvision import datasets, transforms
from utils.options import args_parser
from models.Nets import CNNMnist, CNNCifar

import torch.multiprocessing as mp


class Partition(object):
    """ Dataset-like object, but only access a subset of it. """

    def __init__(self, data, index):
       self.data = data
       # self.index = index
       self.index=list(index)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        data_idx = self.index[index]
        return self.data[data_idx]

def main_worker(gpu,ngpus_per_node, args):
    print("gpu:",gpu)
    args.gpu = gpu
    if args.rank==0:#(第一台服务器只有三台GPU，需要特殊处理)
        newrank=args.rank*ngpus_per_node+gpu
    else:
        newrank=args.rank*ngpus_per_node+gpu-1
    #初始化,使用tcp方式进行通信
    dist.init_process_group(init_method=args.init_method,backend="nccl",world_size=args.world_size,rank=newrank)

    #建立通信group,rank=0作为server，用broadcast模拟send和rec，需要server和每个client建立group
    group=[]
    for i in range(1,args.world_size):
        group.append(dist.new_group([0,i]))
    allgroup=dist.new_group([i for i in range(args.world_size)])


    if newrank==0:
        """ server"""

        print("{}号服务器的第{}块GPU作为server".format(args.rank,gpu))

    #在模型训练期间，server只负责整合参数并分发，不参与任何计算
        #设置cpu
        args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')



        #加载测试数据
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_test = datasets.MNIST('data/', train=False, download=True, transform=trans_mnist)
        test_set= torch.utils.data.DataLoader(dataset_test, batch_size=args.bs)


        """calculate influence function"""
        model=CNNMnist().to(args.device)
        model.load_state_dict(torch.load('w_wag'))

        test_id=0      #选择的test数据id
        data, target = test_set.dataset[test_id]
        data= test_set.collate_fn([data])
        target = test_set.collate_fn([target])
        
        print("begin grad")
        grad_test=grad_z(data,target,model,gpu,create_graph=False)   #v初始值
        print("end grad")
        v=grad_test
        s_test = []


        """server与client交互计算s_test"""
        for i in range(args.world_size-1):
            #id_client=random.randint(1,args.world_size) #选择client
            #向选择的client发送当前v
            print("send v to client:",i+1)
            for j in v:
                temp=j
                dist.broadcast(src=0,tensor=temp,group=group[i])
            #当client计算完成，从client接收v，准备发给下一个client
            print("rec v from client:",i+1)
            v_new=copy.deepcopy(v)
            for j in v_new:
                temp=j
                dist.broadcast(src=i+1,tensor=temp,group=group[i])
            s_test.append(v_new)
	#s_test计算结束，将最终s_test发送给全体client
        e_s_test=s_test[0]
        for i in range(1,args.world_size-1):
            e_s_test = [i + j for i, j in six.moves.zip(e_s_test, s_test[i])]
        for j in e_s_test:
            temp=j
            dist.broadcast(src=0,tensor=temp,group=allgroup)
        """交互结束"""

        #从client接收influence
        print("rec influence")
        allinfluence=[]
        influence=torch.tensor([i for i in range(4285)],dtype=torch.float32)
        influence=influence.to(args.device)

        for i in range(args.world_size-1):
            dist.broadcast(src=i+1,tensor=influence,group=group[i])
            temp=copy.deepcopy(influence)
            allinfluence.append(temp)
        torch.save(allinfluence,'influence')
    else:
        """clents"""

        print("{}号服务器的第{}号GPU作为第{}个client".format(args.rank, gpu,newrank))

        #设置gpu
        args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
        #加载训练数据
        data=torch.load("data/distributed/data_of_client{}".format(newrank))
        bsz=64
        train_set=torch.utils.data.DataLoader(data, batch_size=bsz)
        model=CNNMnist().to(args.device)
        model.load_state_dict(torch.load('w_wag'))     #加载模型
        data, target= train_set.dataset[0]
        data = train_set.collate_fn([data])
        target= train_set.collate_fn([target])
        grad_v = grad_z(data, target, model,gpu=gpu)
        v=grad_v

        """calculate influence function"""
        v_new=[]
        #从server接收v
        """ 和server交互计算s_test，可以循环迭代(当前只进行了一次迭代，没有循环）"""
        for i in v:
            temp=i
            dist.broadcast(src=0,tensor=temp,group=group[newrank-1])
            v_new.append(temp)
        s_test=stest(v_new,model,train_set,gpu,damp=0.01,scale=1000.0,repeat=5)   #计算s_test
        #向server发送s_test,进行下一次迭代
        for i in s_test:
            temp=copy.copy(i)
            dist.broadcast(src=newrank,tensor=temp,group=group[newrank-1])
        #迭代完成后，从server接收最终的s_test，计算influence function
        s_test_fin=[]
        for i in s_test:
            temp=copy.copy(i)
            dist.broadcast(src=0,tensor=temp,group=allgroup)
            s_test_fin.append(temp)
        """s_test计算结束，得到最终的s_test_fin，开始计算influence"""
        print("client:",newrank,"calculate influence")
        n=len(train_set.dataset)
        influence=np.array([i for i in range(n)],dtype='float32')
        for i in utility.create_progressbar(len(train_set.dataset), desc='influence', start=0):

            #计算grad
            data, target= train_set.dataset[i]
            data = train_set.collate_fn([data])
            target= train_set.collate_fn([target])
            grad_z_vec = grad_z(data, target, model,gpu=gpu)
            #计算influence
            inf_tmp = -sum([torch.sum(k * j).data.cpu().numpy() for k, j in six.moves.zip(grad_z_vec, s_test_fin)]) /n
            influence[i]=inf_tmp
        influence=torch.tensor(influence).to(args.device)
        #向服务器发送influence
        print("client:",newrank,"send influence to server")
        dist.broadcast(src=newrank,tensor=influence,group=group[newrank-1])
        print("client:",newrank,"end send influence to server")

if __name__ == '__main__':

    args = args_parser()

    ngpus_per_node = torch.cuda.device_count()
    #args.world_size = ngpus_per_node * args.world_size
    args.world_size=15
    mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
