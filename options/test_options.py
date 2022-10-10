import argparse

class Test_Options():

    def __init__(self):
        self.parser=argparse.ArgumentParser()

    def initialize(self,parser):

        # 硬件
        parser.add_argument("--gpu_ids",type=str,default='0')
        parser.add_argument('--nums_works', type=int, default=0)
        parser.add_argument('--device', type=str, default='cuda')

        # 数据集
        parser.add_argument('--batch', type=int, default=1)
        parser.add_argument('--img_size', type=int, default=(256,256))
        parser.add_argument('--shuffle', type=bool, default=True)
        parser.add_argument('--joint_num', type=int, default=18, help='joint num')
        parser.add_argument('--sg_num', type=int, default=7, help='parsing num')
        parser.add_argument('--sigma', type=int, default=100)

        # 路径
        parser.add_argument('data_dir', type=str, default='')
        parser.add_argument('--checkpoint_dir', type=str, default='checkpoint\\test')
        parser.add_argument('output_dir', type=str, default='')
        parser.add_argument('process_dir', type=str, default='')

        # 模型
        parser.add_argument('--ngf', type=int, default=16)
        parser.add_argument('--pm_bottle_nc',type=int,default=3)
        parser.add_argument('--aa_bottle_nc', type=int, default=3)
        parser.add_argument('--grid_num', type=int, default=5)
        # parser.add_argument('--t', type=int, default=[0.1*(i+1) for i in range(10)])
        parser.add_argument('--t', type=int, default=[0.2*(i+1) for i in range(5)])

        return parser

    def parse(self):

        parser=self.initialize(self.parser)
        opt,_=parser.parse_known_args()

        self.opt=opt

        return self.opt








