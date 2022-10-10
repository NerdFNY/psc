import torch.utils.data as data
import torchvision.transforms as transforms
import torch

import os
import csv
import numpy as np
from PIL import Image

class PSCdataset(data.Dataset):

    def __init__(self):
        super(PSCdataset,self).__init__()

    def initialize(self,opt):


        # sg
        self.i0_dir=os.path.join(opt.data_dir, "I0")
        self.s0_dir = os.path.join(opt.data_dir, "S0")
        self.p0_dir = os.path.join(opt.data_dir, "P0")
        self.p1_dir = os.path.join(opt.data_dir, "P1")

        # files
        self.i0_files=os.listdir(self.i0_dir)
        self.s0_files = os.listdir(self.s0_dir)
        self.p0_files = os.listdir(self.p0_dir)
        self.p1_files = os.listdir(self.p1_dir)

        # num
        self.sg_num=opt.sg_num
        self.data_size=len(self.i0_files)
        self.img_size=opt.img_size
        self.sigma=opt.sigma
        self.miss_value=-1
        self.joint_num=opt.joint_num

        transform_list = []
        transform_list.append(transforms.Resize(size=self.img_size))
        transform_list.append(transforms.ToTensor())
        transform_list.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        self.trans = transforms.Compose(transform_list)

    def __getitem__(self,index):


        i0_path=os.path.join(self.i0_dir,self.i0_files[index])
        s0_path = os.path.join(self.s0_dir, self.s0_files[index])
        p0_path = os.path.join(self.p0_dir, self.p0_files[index])
        p1_path = os.path.join(self.p1_dir, self.p1_files[index])

        # read s0
        sg_img = Image.open(s0_path)
        sg_img = np.expand_dims(np.array(sg_img)[:, :, 0], 0)

        sg_img_1d= torch.from_numpy(sg_img).view(-1).long()
        ones = torch.sparse.torch.eye(self.sg_num)
        ones = ones.index_select(0, sg_img_1d)
        sg_onehot=ones.view([self.img_size[0],self.img_size[1], self.sg_num])
        s0=sg_onehot.permute(2,0,1)

        # read img
        i0 = Image.open(i0_path)
        i0=self.trans(i0)

        # read joint
        p0_joint = []
        f = open(p0_path, 'r', encoding='utf - 8', newline="")
        csv_reader = csv.reader(f)
        for row in csv_reader:
            p0_joint.append(row)

        p0_j2map = self.joint2map(p0_joint )
        p0_j2map = np.transpose(p0_j2map, (2, 0, 1))  # 18*256*256
        p0= torch.Tensor(p0_j2map)  # tensor

        p1_joint = []
        f = open(p1_path, 'r', encoding='utf - 8', newline="")
        csv_reader = csv.reader(f)
        for row in csv_reader:
            p1_joint.append(row)

        p1_j2map = self.joint2map(p1_joint)
        p1_j2map = np.transpose(p1_j2map, (2, 0, 1))  # 18*256*256
        p1 = torch.Tensor(p1_j2map)

        return i0,s0,p0,p1

    def __len__(self):
        return  self.data_size

    def joint2map(self,J):

        joint=np.zeros(shape=(2,self.joint_num))

        joint[0,:]=np.array(J[0][1:])
        joint[1, :] = np.array(J[1][1:])

        joint=joint.astype(float)
        map=np.zeros(shape=(self.img_size[0],self.img_size[1],self.joint_num),dtype='float32')

        for i in range(joint.shape[1]):

            if joint[0,i]==self.miss_value or joint[1,i]==self.miss_value:
                continue

            joint_x=int(joint[0,i])
            joint_y=int(joint[1,i])

            xx,yy = np.meshgrid(np.arange(self.img_size[0]), np.arange(self.img_size[1]))

            map[:,:,i]=np.exp(-((yy - joint_y) ** 2 + (xx - joint_x) ** 2) / (2 * self.sigma ** 2))
        return map