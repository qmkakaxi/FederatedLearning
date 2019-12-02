import numpy as np


def generate_attack_data(dataName, dict_users, dataset, noisy_client, noisy_rate):
    """
    将一部分用户数据中的2与6的数据混合，形成标签为2的新数据
    noisy_rate是加的噪音比例
    """
    if dataName=='mnist':
        originTargets=dataset.targets.numpy()
    else:
        originTargets=np.array(dataset.targets)

    if noisy_client>len(dict_users.keys()):
        print('too many noisy client')
        raise NameError('noisy_client')
        exit()
    noisyDataList = []
    all6Index = np.where(originTargets==6)[0]
    all2Index = np.where(originTargets==2)[0]
    noisyAddRelation={}
    for userIndex in range(noisy_client):
        for dataIndex in dict_users[userIndex]:
            if dataIndex in all2Index:
                maskDataIndex = np.random.choice(list(all6Index), 1, replace=False)
                dataset.data[dataIndex]=(1-noisy_rate)*dataset.data[dataIndex]+noisy_rate*dataset.data[maskDataIndex]
                noisyDataList.append(dataIndex)
                noisyAddRelation[dataIndex] = maskDataIndex

    return dataset, noisyDataList

