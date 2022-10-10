import torch.utils.data as data
import torchvision.transforms as transforms
import torch

import os
import csv
import numpy as np
from PIL import Image

class PMdataset(data.Dataset):

    def __init__(self):
        super(PMdataset, self).__init__()

    def initialize(self, opt):

        self.opt = opt
        self.joint_dir = os.path.join(self.opt.data_dir, self.opt.joint_interpolation_dir)
        self.joint_files = os.listdir(self.joint_dir)
        self.joint_num = opt.joint_num
        self.data_size = len(self.joint_files)
        self.img_size = opt.img_size

        self.sigma = opt.sigma
        self.miss_value = -1

        # Totensor
        transform_list = []
        transform_list.append(transforms.Resize(size=self.img_size))
        transform_list.append(transforms.ToTensor())
        transform_list.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        self.trans = transforms.Compose(transform_list)

    def __getitem__(self, index):

        # csv path
        joint_name = os.path.join(self.joint_dir, self.joint_files[index])

        joint = []
        f = open(joint_name, 'r', encoding='utf - 8', newline="")
        csv_reader = csv.reader(f)
        for row in csv_reader:
            joint.append(row)

        j2maps = []
        for n in range(int(len(joint) / 2)):
            j2map = self.joint2map(joint[2 * n:2 * n + 2])
            j2map = np.transpose(j2map, (2, 0, 1))  # 18*256*256
            j2map = torch.Tensor(j2map)  # 转换为tensor
            j2maps.append(j2map)

        # manifold 参数
        t_all = []
        for n in range(int(len(joint) / 2 - 4)):
            t = joint[2 * (n + 4)][0]
            t = torch.Tensor([float(t)])
            t_all.append(t)

        return j2maps, t_all

    def __len__(self):
        return self.data_size

    # joint to map
    def joint2map(self, J):

        joint = np.zeros(shape=(2, self.joint_num))
        joint[0, :] = np.array(J[0][1:])
        joint[1, :] = np.array(J[1][1:])

        joint = joint.astype(float)
        map = np.zeros(shape=(self.img_size[0], self.img_size[1], self.joint_num), dtype='float32')

        for i in range(joint.shape[1]):

            if joint[0, i] == self.miss_value or joint[1, i] == self.miss_value:
                continue

            joint_x = int(joint[0, i])
            joint_y = int(joint[1, i])

            xx, yy = np.meshgrid(np.arange(self.img_size[0]), np.arange(self.img_size[1]))

            map[:, :, i] = np.exp(-((yy - joint_y) ** 2 + (xx - joint_x) ** 2) / (2 * self.sigma ** 2))

        return map


class AAdataset(data.Dataset):

    def __init__(self):
        super(AAdataset,self).__init__()

    def initialize(self,opt):

        # joint
        self.joint_dir=os.path.join(opt.data_dir,opt.joint_dir)

        # sg
        self.front_dir=os.path.join(opt.data_dir,opt.front_sg_dir)
        self.back_dir = os.path.join(opt.data_dir, opt.back_sg_dir)
        self.left_dir = os.path.join(opt.data_dir, opt.left_sg_dir)
        self.right_dir = os.path.join(opt.data_dir, opt.right_sg_dir)
        self.mid0_dir = os.path.join(opt.data_dir, opt.mid0_sg_dir)
        self.mid1_dir = os.path.join(opt.data_dir, opt.mid1_sg_dir)
        self.mid2_dir = os.path.join(opt.data_dir, opt.mid2_sg_dir)
        self.mid3_dir = os.path.join(opt.data_dir, opt.mid3_sg_dir)

        # files
        self.sg_files=os.listdir(self.front_dir)
        self.joint_files = os.listdir( self.joint_dir)

        # num
        self.joint_num=opt.joint_num
        self.sg_num=opt.sg_num
        self.data_size=len(self.joint_files)
        self.img_size=opt.img_size

        self.sigma=opt.sigma
        self.miss_value=-1

        # Totensor
        transform_list=[]
        transform_list.append(transforms.Resize(size=self.img_size))
        transform_list.append(transforms.ToTensor())
        transform_list.append(transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5)))
        self.trans = transforms.Compose(transform_list)

    def __getitem__(self,index):

        ''' joint '''
        joint_path=os.path.join(self.joint_dir,self.joint_files[index])
        joint = []
        f = open(joint_path, 'r', encoding='utf - 8', newline="")
        csv_reader = csv.reader(f)
        for row in csv_reader:
            joint.append(row)


        j2maps = []
        for n in range(int(len(joint) / 2)):
            j2map = self.joint2map(joint[2 * n:2 * n + 2])
            j2map = np.transpose(j2map, (2, 0, 1))  # 18*256*256
            j2map = torch.Tensor(j2map)  # 转换为tensor
            j2maps.append(j2map)


        ''' sg '''
        # sg_path
        sgs=[]
        sgs_label=[]

        front_name=os.path.join(self.front_dir,self.sg_files[index])
        back_name=os.path.join(self.back_dir,self.sg_files[index])
        right_name=os.path.join(self.right_dir,self.sg_files[index])
        left_name = os.path.join(self.left_dir, self.sg_files[index])

        mid0_name = os.path.join(self.mid0_dir, self.sg_files[index])
        mid1_name = os.path.join(self.mid1_dir, self.sg_files[index])
        mid2_name = os.path.join(self.mid2_dir, self.sg_files[index])
        mid3_name = os.path.join(self.mid3_dir, self.sg_files[index])
        sg_path=[right_name,front_name,left_name,back_name,mid0_name, mid1_name, mid2_name, mid3_name]

        for n in range(len(sg_path)):

            sg_img = Image.open(sg_path[n])
            sg_img = np.expand_dims(np.array(sg_img)[:, :, 0], 0)

            sg_img_1d= torch.from_numpy(sg_img).view(-1).long()
            ones = torch.sparse.torch.eye(self.sg_num)
            ones = ones.index_select(0, sg_img_1d)
            sg_onehot=ones.view([self.img_size[0],self.img_size[1], self.sg_num])
            sg_onehot=sg_onehot.permute(2,0,1)
            sgs.append(sg_onehot)

            sg_img_label=torch.from_numpy(sg_img).long()
            sgs_label.append(sg_img_label)

        return j2maps,sgs,sgs_label

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


class CTdataset(data.Dataset):

    def __init__(self):
        super(CTdataset,self).__init__()

    def initialize(self,opt):

        # sg
        self.front_sg_dir=os.path.join(opt.data_dir,"front_sg")
        self.back_sg_dir = os.path.join(opt.data_dir,"back_sg")
        self.left_sg_dir = os.path.join(opt.data_dir,"left_sg")
        self.right_sg_dir = os.path.join(opt.data_dir, "right_sg")
        self.mid0_sg_dir = os.path.join(opt.data_dir, "mid0_sg")
        self.mid1_sg_dir = os.path.join(opt.data_dir, "mid1_sg")
        self.mid2_sg_dir = os.path.join(opt.data_dir, "mid2_sg")
        self.mid3_sg_dir = os.path.join(opt.data_dir, "mid3_sg")
        self.sg_dir=[self.right_sg_dir,self.front_sg_dir,self.left_sg_dir, self.back_sg_dir,
                     self.mid0_sg_dir,self.mid1_sg_dir,self.mid2_sg_dir,self.mid3_sg_dir]

        # img
        self.front_img_dir = os.path.join(opt.data_dir, "front_img")
        self.back_img_dir = os.path.join(opt.data_dir, "back_img")
        self.left_img_dir = os.path.join(opt.data_dir, "left_img")
        self.right_img_dir = os.path.join(opt.data_dir, "right_img")
        self.mid0_img_dir = os.path.join(opt.data_dir, "mid0_img")
        self.mid1_img_dir = os.path.join(opt.data_dir, "mid3_img")
        self.mid2_img_dir = os.path.join(opt.data_dir, "mid2_img")
        self.mid3_img_dir = os.path.join(opt.data_dir, "mid1_img")
        self.img_dir=[self.right_img_dir,self.front_img_dir,self.left_img_dir,self.back_img_dir,
                      self.mid0_img_dir,self.mid1_img_dir,self.mid2_img_dir,self.mid3_img_dir]

        self.sg_files=os.listdir(self.front_sg_dir)
        self.img_files=os.listdir(self.front_img_dir)

        self.sg_num=opt.sg_num
        self.data_size=len(self.sg_files)
        self.img_size=opt.img_size

        transform_list = []
        transform_list.append(transforms.Resize(size=self.img_size))
        transform_list.append(transforms.ToTensor())
        transform_list.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        self.trans = transforms.Compose(transform_list)

    def __getitem__(self,index):

        ''' sg '''
        # sg_path
        sg_path=[]
        for v in range(len(self.sg_dir)):
            file_path=os.path.join(self.sg_dir[v],self.sg_files[index])
            sg_path.append(file_path)

        '''img'''
        img_path=[]
        for v in range(len(self.img_dir)):
            file_path = os.path.join(self.img_dir[v], self.img_files[index])
            img_path.append(file_path)

        # read sg
        sgs=[]
        for n in range(len(sg_path)):

            sg_img = Image.open(sg_path[n])
            sg_img = np.expand_dims(np.array(sg_img)[:, :, 0], 0)

            sg_img_1d= torch.from_numpy(sg_img).view(-1).long()
            ones = torch.sparse.torch.eye(self.sg_num)
            ones = ones.index_select(0, sg_img_1d)
            sg_onehot=ones.view([self.img_size[0],self.img_size[1], self.sg_num])
            sg_onehot=sg_onehot.permute(2,0,1)
            sgs.append(sg_onehot)

        # read img
        imgs = []
        for n in range(len(img_path)):
            sg_img = Image.open(img_path[n])
            sg_img=self.trans(sg_img)
            imgs.append(sg_img)

        return sgs,imgs

    def __len__(self):
        return  self.data_size

