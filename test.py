from options.test_options import Test_Options
from dataset.test_dataset import PSCdataset
from network.psc_model import *
import torch.utils.data

def test(opt):

    opt.shuffle = False
    opt.batch = 1
    opt.nums_works = 0

    # 创建数据集
    psc_dataset=PSCdataset()
    psc_dataset.initialize(opt)

    psc_dataset_loader=torch.utils.data.DataLoader(
        psc_dataset,
        batch_size=opt.batch,
        shuffle=opt.shuffle,
        num_workers=int(opt.nums_works))

    # 创建模型
    psc_model=PSC_model(opt)
    psc_model.print_network()

    print("----- test start -----")

    ite=0
    for index, data in enumerate(psc_dataset_loader):

        print(index)
        psc_model.setinput(data)  # 设置输入
        psc_model.test(ite) # 评估
        ite+=1

if __name__=="__main__" :
    opt=Test_Options().parse()
    test(opt)