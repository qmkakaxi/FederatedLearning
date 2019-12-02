# FederatedLearning

基于pytorch和nccl实现的简易联邦学习和数据质量评估系统

## 实验环境（USTC linke实验室）：

 1. 四台深度学习服务器，每台服务器配有48个Xeon CPU核，128G 内存和4个Tesla P100 GPU卡。（第一台只有三个GPU卡）
 2. 四台服务器都在同一个局域网内，相互可以ping通。
 3. python 3.5.2 pytorch 1.3.0 
 4. cuda 9.2     nccl 2.3.4
 
 ## 实现：
   本次实验使用gpu作为一个client，每个clent拥有自己数据，每个client在本地进行训练后，将梯度数据传给server，由server整合后分发给client。
 
 ## 运行：
 
 使用第一台服务器的第1号GPU作为server，其余14个GPU作为client
 
 ### 联邦学习：
 
 python fedlearning.py --rank 0 --world-size 4
 
 python fedlearning.py --rank 1 --world-size 4
 
 python fedlearning.py --rank 2 --world-size 4
 
 python fedlearning.py --rank 3 --world-size 4
 
 ### 计算influence：
 
 python fedinfluence.py --rank 0 --world-size 4
 
 python fedinfluence.py --rank 1 --world-size 4
 
 python fedinfluence.py --rank 2 --world-size 4
 
 python fedinfluence.py --rank 3 --world-size 4
