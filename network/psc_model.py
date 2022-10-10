from network.AA_Net import *
from network.CT_Net import *
from network.PM_Net import *
from skimage.draw import circle, line_aa, polygon
import torch
import os
import numpy as np
from torchvision.utils import save_image

class PSC_model():

    def __init__(self,opt):

        super(PSC_model,self).__init__()

        self.gpu_ids=opt.gpu_ids
        self.device = opt.device
        self.ngf=opt.ngf
        self.batch=opt.batch
        self.sg_num=opt.sg_num

        # file
        if not os.path.exists(opt.output_dir):
            os.makedirs(opt.output_dir)
        if not os.path.exists(opt.process_dir):
            os.makedirs(opt.process_dir)

        # componet
        self.pm_net=PMNet(opt,input_nc=opt.joint_num,output_nc=opt.joint_num,ngf=16)
        self.aa_net=AANet(opt,input_nc=[opt.sg_num,opt.joint_num],output_nc=opt.sg_num,ngf= self.ngf)
        self.ct_net=CTNet(opt,input_nc=[opt.sg_num,3,1],output_nc=3,ngf= self.ngf)

        self.pm_net.init_weights()
        self.aa_net.init_weights()
        self.ct_net.init_weights()

        if len(self.gpu_ids)>0 and torch.cuda.is_available():
            self.pm_net.cuda()
            self.aa_net.cuda()
            self.ct_net.cuda()

        # load checkpoint
        self.checkpoint_dir = opt.checkpoint_dir
        self.pm_net.load_state_dict(torch.load(os.path.join(self.checkpoint_dir, "pm.pth"),
                                          map_location=torch.device(self.device)))
        print("load checkpoint: pm_net sucess!")
        self.aa_net.load_state_dict(torch.load(os.path.join(self.checkpoint_dir, 'aa.pth'),
                                          map_location=torch.device(self.device)))
        print("load checkpoint: aa_net sucess!")
        self.ct_net.load_state_dict(torch.load(os.path.join(self.checkpoint_dir, "ct.pth"),
                                          map_location=torch.device(self.device)))
        print("load checkpoint: ct_net sucess!")

        # eval
        self.output_dir=opt.output_dir
        self.process_dir=opt.process_dir

        # interpolate para
        self.t=opt.t
        self.threshold = 0.5
        self.joint_num=opt.joint_num
        self.img_size=opt.img_size
        self.miss_value=-1
        self.sigma=opt.sigma
        self.vis_cir = 2

    # read input
    def setinput(self,input):

        self.input=input

        i0,s0,p0,p1=input[0],input[1],input[2],input[3]

        if len(self.gpu_ids) > 0:
            self.i0=i0.cuda()
            self.s0 = s0.cuda()
            self.p0 = p0.cuda()
            self.p1 = p1.cuda()


    def test(self,ite):

        # pm-net
        for n in range(len(self.t)):
            if self.t[n]<1:
                pt=self.pm_net(self.p0,self.p1,torch.FloatTensor([self.t[n]]).cuda())
            else:
                pt=self.p1.clone().detach()

            pt=pt.detach().cpu().numpy()[0].transpose([1, 2, 0])
            pt_map=torch.FloatTensor(self.joint2transfer(pt)).permute([2,0,1]).unsqueeze(0).cuda()
            pt_vis=torch.FloatTensor(self.map2img(pt)).cuda().permute([2,0,1])/255-1

            # aa-net
            st=self.aa_net(self.s0,pt_map)
            st_map=self.par2transfer(st)
            st_map=st_map.cuda()
            st_vis=self.par2img(st_map).squeeze(0)/255-1

            # ct-net
            ct,_,_=self.ct_net(self.s0,st_map,self.i0)

            # save final result
            save_path=os.path.join(self.output_dir,"%04d_%.2f.jpg"%(ite,self.t[n]))
            ct= Variable(ct)
            save_image(self.de_norm(ct[0].data), save_path, normalize=True)

            # save process result
            p0 = self.p0.detach().cpu().numpy()[0].transpose([1, 2, 0])
            p0_vis = torch.FloatTensor(self.map2img(p0)).cuda().permute([2, 0, 1])/255-1
            p1 = self.p1.detach().cpu().numpy()[0].transpose([1, 2, 0])
            p1_vis = torch.FloatTensor(self.map2img(p1)).cuda().permute([2, 0, 1])/255-1
            s0_vis = self.par2img(self.s0).squeeze(0)/255-1

            save_path = os.path.join(self.process_dir, "%04d_%.2f.jpg" % (ite, self.t[n]))
            output=torch.cat([p0_vis,p1_vis,s0_vis,self.i0.squeeze(0)/255-1,pt_vis,st_vis,ct.squeeze(0)/255-1],dim=-1)
            save_image(self.de_norm(output.data), save_path, normalize=True)


    def joint2transfer(self, J):

        ''' peak value '''
        threshold = self.threshold
        all_peaks = [[] for i in range(18)]
        pose_map = J[..., :18]

        y, x, z = np.where(np.logical_and(pose_map == pose_map.max(axis=(0, 1)), pose_map > threshold))
        for x_i, y_i, z_i in zip(x, y, z):
            all_peaks[z_i].append([x_i, y_i])

        x_values = []
        y_values = []

        for i in range(18):
            if len(all_peaks[i]) != 0:
                x_values.append(all_peaks[i][0][0])
                y_values.append(all_peaks[i][0][1])
            else:
                x_values.append(-1)
                y_values.append(-1)

        J=np.array([x_values,y_values]).transpose()

        joint = np.zeros(shape=(2, self.joint_num))
        joint[0, :] = np.array(J[:, 0])
        joint[1, :] = np.array(J[:, 1])

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

    def par2transfer(self,par_map):

        par_all = torch.cuda.FloatTensor(size=(self.batch, self.sg_num, self.img_size[0], self.img_size[1]))

        for n in range(self.batch):

            vis_color = torch.cuda.FloatTensor(
                [(0, 0, 0), (1, 1, 1), (2, 2, 2), (3, 3, 3), (4, 4, 4), (5, 5, 5), (6, 6, 6),
                 (7, 7, 7), (8, 8, 8), (9, 9, 9)]).long()

            index = torch.argmax(par_map[n], dim=0).view(-1)
            ones = vis_color.index_select(0, index)
            par = ones.view(self.img_size[0], self.img_size[1], 3).long().permute(2, 0, 1)

            sg_img_1d = par[0,:,:].unsqueeze(0).view(-1)  # 变为一维向量
            ones = torch.sparse.torch.eye(self.sg_num).cuda() # onehot
            ones = ones.index_select(0, sg_img_1d)
            sg_onehot = ones.view([self.img_size[0], self.img_size[1], self.sg_num])  # 图像的onehot 编码
            s0_onehot = sg_onehot.permute(2, 0, 1)

            par_all[n] = s0_onehot

        return par_all



    def map2img(self,pose_map):

        threshold=self.threshold

        all_peaks = [[] for i in range(18)]
        pose_map = pose_map[..., :18]

        y, x, z = np.where(np.logical_and(pose_map == pose_map.max(axis=(0,1)),pose_map > threshold))
        for x_i, y_i, z_i in zip(x, y, z):
            all_peaks[z_i].append([x_i, y_i])

        x_values = []
        y_values = []

        for i in range(18):
            if len(all_peaks[i]) != 0:
                x_values.append(all_peaks[i][0][0])
                y_values.append(all_peaks[i][0][1])
            else:
                x_values.append(-1)
                y_values.append(-1)


        # 关节点 18*2
        cords=np.concatenate([np.expand_dims(y_values, -1), np.expand_dims(x_values, -1)], axis=1)

        #
        joint_img=np.zeros(shape=(self.img_size[0],self.img_size[1],3))

        # 运动树依赖
        LIMB_SEQ = [[1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7], [1, 8], [8, 9],
                    [9, 10], [1, 11], [11, 12], [12, 13], [1, 0], [0, 14], [14, 16],
                    [0, 15], [15, 17], [2, 16], [5, 17]]

        # 颜色
        COLORS = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
                  [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
                  [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]

        for f, t in LIMB_SEQ:
            from_missing = cords[f][0] == -1 or cords[f][1] == -1
            to_missing = cords[t][0] == -1 or cords[t][1] == -1
            if from_missing or to_missing:
                continue
            yy, xx, val = line_aa(cords[f][0], cords[f][1], cords[t][0], cords[t][1])
            joint_img[yy, xx] = np.expand_dims(val, 1) * 255

        for i, joint in enumerate(cords):
            if cords[i][0] == -1 or cords[i][1] == -1:
                continue
            yy, xx = circle(joint[0], joint[1], radius=6, shape=(256, 256, 3))
            joint_img[yy, xx] = COLORS[i]

        return joint_img

    def par2img(self,par_map):

        par_all=torch.cuda.FloatTensor(size=(self.batch,3,self.img_size[0],self.img_size[1]))


        for n in range(self.batch):
            vis_color = torch.cuda.FloatTensor(
                [(0, 0, 0), (255, 255, 0), (0, 255, 255), (255, 0, 255), (0, 0, 255), (0, 255, 0), (255, 0, 0),
                 (244, 164, 96), (255, 255, 255), (160, 32, 240)]).long()


            index=torch.argmax(par_map[n],dim=0).view(-1)
            ones = vis_color.index_select(0, index)
            par=ones.view(self.img_size[0],self.img_size[1],3).long().permute(2,0,1)
            par_all[n]=par

        return par_all

    def de_norm(self, x):
        out = (x + 1) / 2
        return out.clamp(0, 1)

    # 打印网络
    def print_network(self):

        model=[self.pm_net,self.aa_net,self.ct_net]

        num_params = 0
        for k in range(len(model)):
            for p in  model[k].parameters():
                num_params += p.numel()
        for k in range(len(model)):
            print(model[k])
        print("The number of parameters: {}".format(num_params))
