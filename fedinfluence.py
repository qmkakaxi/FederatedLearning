import six
import torch
import torch.distributed as dist
from models.influence import grad_z,stest
import models.utility as utility
import copy
import numpy as np
from torchvision import datasets, transforms
from utils.sampling import mnist_iid, mnist_noniid, cifar_iid, mnist_remove
from utils.options import args_parser
from models.Nets import CNNMnist, CNNCifar

import torch.multiprocessing as mp


class Partition(object):

    def __init__(self, data, index):
       self.data = data
       # self.index = index
       self.index=list(index)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        data_idx = self.index[index]
        return self.data[data_idx]



def partition_dataset():
    """ 分发数据 """

    trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    dataset_train = datasets.MNIST('data/', train=True, download=True, transform=trans_mnist)
    size = dist.get_world_size()-1
    bsz = int(128 / float(size))
    num_users = size
    dict_users = mnist_iid(dataset_train, 1000)
    print("dict_user:{}".format(len(dict_users)))
    print("world-size:{}".format(dist.get_world_size()))
    print("rank:{}".format(dist.get_rank()))
    partition = Partition(dataset_train, dict_users[dist.get_rank()-1])
    train_set = torch.utils.data.DataLoader(
        partition, batch_size=bsz)
    return train_set, bsz


def main_worker(gpu,ngpus_per_node, args):
    print("gpu:",gpu)
    args.gpu = gpu
    newrank=args.rank*ngpus_per_node+gpu
    ngpus_per_node * args.world_size
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
        grad_test=grad_z(data,target,model,gpu)   #v初始值
        print("end grad")
        #v=torch.tensor(grad_test).to(args.device)
        v=grad_test
        #向client发送v
 
        """server与client交互计算s_test"""
        for i in range(args.world_size-1):
            #id_client=random.randint(1,args.world_size) #选择client
            #向选择的client发送当前v
            print("send v to i:",i)
            for j in v:
                temp=j
                dist.broadcast(src=0,tensor=temp,group=group[i])
            #当client计算完成，从client接收v，准备发给下一个client
            print("rec v from i:",i)
            v_new=[]
            for j in v:
                temp=j
                dist.broadcast(src=i+1,tensor=temp,group=group[i])
                v_new.append(temp)
            v=v_new
	#s_test计算结束，将最终s_test发送给全体client
        for j in v:
            temp=j
            dist.broadcast(src=0,tensor=temp,group=allgroup)
        """交互结束"""

        #从client接收influence
        print("rec influence")
        allinfluence=[]
        influence=torch.tensor([i for i in range(60)],dtype=torch.float32)
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
        train_set, bsz = partition_dataset()
        model=CNNMnist().to(args.device)
        model.load_state_dict(torch.load('w_wag'))     #加载模型
        v=[i.to(args.device) for i in list(model.parameters())]


        """calculate influence function"""
        v_new=[]
        #从server接收v
        """ 和server交互计算s_test，可以循环迭代(当前只进行了一次迭代，没有循环）"""
        for i in v:
            temp=i
            dist.broadcast(src=0,tensor=temp,group=group[newrank-1])
            v_new.append(temp)
        s_test=stest(v_new,model,train_set,gpu,damp=0.01,scale=25.0,repeat=5)   #计算s_test
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
            inf_tmp = -sum([torch.sum(k * j).data.cpu().numpy() for k, j in six.moves.zip(grad_z_vec, s_test_fin)]) 
            influence[i]=inf_tmp
        influence=torch.tensor(influence).to(args.device)
        #向服务器发送influence
        print("client:",newrank,"send influence to server")
        dist.broadcast(src=newrank,tensor=influence,group=group[newrank-1])
        print("client:",newrank,"end send influence to server")

if __name__ == '__main__':

    args = args_parser()

    ngpus_per_node = torch.cuda.device_count()
    args.world_size = ngpus_per_node * args.world_size
  
    mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
