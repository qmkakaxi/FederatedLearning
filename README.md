# FederatedLearning

基于pytorch和nccl实现的简易联邦学习和数据质量评估系统

## 实验环境（USTC linke实验室）：

 1. 四台深度学习服务器，每台服务器配有48个Xeon CPU核，128G 内存和4个Tesla P100 GPU卡。（第一台只有三个GPU卡）
 2. 四台服务器都在同一个局域网内，相互可以ping通。
 3. python 3.5.2   pytorch 1.3.0 
 4. cuda 9.2       nccl 2.3.4
 
 ## 实现：
 
   federatedlearning：本次实验使用gpu作为一个client，每个client拥有自己数据，client在本地进行训练后，将梯度数据传给server，由server整合后分发给所有client
   
   influencefunction：由server和client交互迭代计算stest，计算完成后，server将最终的stest发给所有client，client根据训练数据的grad和stest计算influencefunction，然后发送给server。在计算influencefunction过程中，client只能得到stest，不能获取到其他client和data和grad。(封装在models.influence)
 
   备注：nccl不支持pytorch框架的send和rev方法，实验使用boradcast模拟send和rev，server和每个client建立一个group，保证client向server发送消息时，不会被其他client收到。
 
 ## 运行：
 
 使用第一台服务器的第1号GPU作为server，其余14个GPU作为client
 
 
  ### 数据分割(DataSplit.py):
 
 对已有数据进行分割：
 
 ```
 python DataSplit.py
 ```
 本实验共有14个client，故将数据集划分为14份。
  ### attackdata分割(attackDataSplit):
 
  ```
 python attackDataSplit.py
 ```

 本实验提供两种attackdata方式（封装在models.attackdata.py)：
 1. generate_attack_data1:
    将一部分用户数据中的2与6的数据混合，形成标签为2的新数据
 2. generate_attack_data2:
    将一部分client的数据的标签置换为错误标签
    
  参数 
    1.  dataName：数据集合的名字，本实验提供mnist和cifar数据集，如果您想测试其他数据集，向其中添加即可。
    2.  dict_users：划分给client的数据的index，用来做确定client具体拥有总数据的哪些数据。
    3.  dataset：原始未数据集。
    4.  noisy_client：type：int，选择前noisy_client的client进行attack处理。
    5.  noisy_rate：attackdata的比例。
  ### 联邦学习：
  在第一台服务器
  ```
 python fedlearning.py --rank 0 --world-size 4
  ```
  在第二台服务器
  ```
 python fedlearning.py --rank 1 --world-size 4
  ```
  在第三台服务器
  ```
 python fedlearning.py --rank 2 --world-size 4
 ```
  在第四台服务器
  ```
 python fedlearning.py --rank 3 --world-size 4
 ```
 
  ### 计算influence：
  在第一台服务器
  ```
 python fedinfluence.py --rank 0 --world-size 4
 ```
  在第二台服务器
  ```
 python fedinfluence.py --rank 1 --world-size 4
 ```
  在第三台服务器
  ```
 python fedinfluence.py --rank 2 --world-size 4
 ```
  在第四台服务器
  ```
 python fedinfluence.py --rank 3 --world-size 4
 ```
 
 备注：代码中的数据均为mnist数据，如果您想使用其他数据，只需要修改数据读取方式并且在models.Nets加入您的网络结构。
 
