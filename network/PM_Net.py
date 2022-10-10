import os
import itertools
from skimage.draw import circle, line_aa, polygon
import cv2 as cv
import time
import datetime
import matplotlib.pyplot as plt


import torch
import torch.nn as nn
from network.function import  *
from torch.nn import init
class PMNet(nn.Module):

    def __init__(self,opt,input_nc,output_nc,ngf,norm_layer=nn.InstanceNorm2d, act=nn.LeakyReLU(0.2), use_spect=False):

        super(PMNet,self).__init__()

        self.opt=opt
        self.gpu_ids=opt.gpu_ids

        # encoder 18 -> 32 -> 64 -> 128 -> 256 -> 512
        self.encoder_conv1 = BlockEncoder(input_nc,ngf*2,ngf,norm_layer,act,use_spect)
        self.encoder_conv2 = BlockEncoder(ngf*2, ngf * 4, ngf*4, norm_layer, act, use_spect)
        self.encoder_conv3 = BlockEncoder(ngf * 4, ngf * 8, ngf * 8, norm_layer, act, use_spect)
        self.encoder_conv4 = BlockEncoder(ngf * 8, ngf * 16, ngf * 16, norm_layer, act, use_spect)

        # bottleneck
        self.bottle_nc=opt.pm_bottle_nc # 瓶颈层的数量
        self.bottle_pose=BlockBottle(ngf * 16,ngf * 16)
        self.bottle_manifold1=BlockBottle(ngf * 16,ngf * 16)
        self.bottle_manifold2 = BlockBottle(ngf * 32, ngf * 16)

        # decoder
        self.decoder_conv1=ResBlockDecoder(ngf * 16,ngf * 8,ngf * 16,norm_layer, act, use_spect)
        self.decoder_conv2 = ResBlockDecoder(ngf * 8, ngf * 4, ngf * 4, norm_layer, act, use_spect)
        self.decoder_conv3 = ResBlockDecoder(ngf * 4, ngf * 2, ngf * 2, norm_layer, act, use_spect)
        self.decoder_conv4 = ResBlockDecoder(ngf * 2,ngf, ngf , norm_layer, act, use_spect)

        # output
        self.output = nn.Sequential(nn.Conv2d(ngf, output_nc, kernel_size=7, stride=1, padding=3, bias=False),
                                    nn.Sigmoid())


    def forward(self,pose_1,pose_2,t):

        # pose_1 -> encoder
        pose1_enc=self.encoder_conv1(pose_1)
        pose1_enc = self.encoder_conv2(pose1_enc)
        pose1_enc = self.encoder_conv3(pose1_enc)
        pose1_enc = self.encoder_conv4(pose1_enc)

        # pose_2 -> encoder
        pose2_enc=self.encoder_conv1(pose_2)
        pose2_enc=self.encoder_conv2(pose2_enc)
        pose2_enc = self.encoder_conv3(pose2_enc)
        pose2_enc = self.encoder_conv4(pose2_enc)

        # Bottleneck
        self.pose_fea_1=pose1_enc
        self.pose_fea_2=pose2_enc

        for i in range(self.bottle_nc):

            # t
            self.t1_map=(1-t).unsqueeze(1).unsqueeze(1).expand(self.pose_fea_1.shape)
            self.t2_map=(t).unsqueeze(1).unsqueeze(1).expand(self.pose_fea_2.shape)

            # map
            self.pose_manifold0=self.feature_inter(self.pose_fea_1,self.pose_fea_2,self.t1_map, self.t2_map)

            # bottleneck
            self.pose_fea_1=self.bottle_pose(self.pose_fea_1)
            self.pose_fea_2=self.bottle_pose(self.pose_fea_2)

            # manifold through bottleneck
            if i ==0:
                self.pose_manifold1=self.bottle_manifold1(self.pose_manifold0)
            else:
                self.pose_manifold1 = torch.cat([self.pose_manifold0, self.pose_manifold1], 1) # 合并
                self.pose_manifold1 = self.bottle_manifold2(self.pose_manifold1)


        # Decoder
        manifold2=self.decoder_conv1(self.pose_manifold1)
        manifold2 = self.decoder_conv2(manifold2)
        manifold2 = self.decoder_conv3(manifold2)
        manifold2 = self.decoder_conv4(manifold2)

        # output norm
        manifold2=self.output(manifold2)

        return manifold2


    def feature_inter(self,feature1,feature2,t1,t2):

        feature1_inter=feature1*t1
        feature2_inter=feature2*t2

        return feature1_inter+feature2_inter

    # init
    def init_weights(self, init_type='normal', gain=0.02):
        def init_func(m):
            classname = m.__class__.__name__
            if classname.find('BatchNorm2d') != -1:
                if hasattr(m, 'weight') and m.weight is not None:
                    init.normal_(m.weight.data, 1.0, gain)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)
            elif hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'xavier_uniform':
                    init.xavier_uniform_(m.weight.data, gain=1.0)
                elif init_type == 'kaiming':
                    init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    init.orthogonal_(m.weight.data, gain=gain)
                elif init_type == 'none':  # uses pytorch's default init method
                    m.reset_parameters()
                else:
                    raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)

        self.apply(init_func)



class PMNet_Train():

    def __init__(self,opt):

        super(PMNet_Train,self).__init__()

        self.gpu_ids=opt.gpu_ids
        self.device = opt.device

        # dir
        self.checkpoint_dir=opt.checkpoint_dir+"\\"+"train"+"\\"+opt.model
        self.vis_dir=opt.vis_dir+"\\"+"train"+"\\"+opt.model
        self.loss_dir=opt.loss_dir+"\\"+"train"+"\\"+opt.model

        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        if not os.path.exists( self.vis_dir):
            os.makedirs( self.vis_dir)
        if not os.path.exists(self.loss_dir):
            os.makedirs(self.loss_dir)

        # generator
        self.g = PMNet(opt,input_nc=opt.joint_num,output_nc=opt.joint_num,ngf=16)
        self.g.init_weights()

        if len(self.gpu_ids)>0 and torch.cuda.is_available():
            self.g.cuda()

        # load checkpoint
        self.inference=opt.pm_inference
        if self.inference:
            print("load checkpoint sucess!")
            self.g.load_state_dict(torch.load(os.path.join(self.checkpoint_dir, "%d_G.pth" % self.inference), map_location=torch.device(self.device)))

        # 优化器
        self.optimizer_g = torch.optim.Adam(itertools.chain(filter(lambda p: p.requires_grad, self.g.parameters())),lr=opt.pm_g_lr, betas=(0.9, 0.999))

        # 损失函数
        self.l1_func=torch.nn.L1Loss()
        self.l2_func=nn.MSELoss()
        self.criterionGAN = GANLoss(use_lsgan=True, tensor=torch.cuda.FloatTensor)
        self.tur_eps=opt.pm_tur_eps
        self.l1_coeff=opt.pm_l1_loss_coeff
        self.l2_coeff = opt.pm_l2_loss_coeff
        self.tur_coeff=opt.pm_tur_loss_coeff
        self.grad_coeff=opt.pm_grad_loss_coeff

        # 可视化
        self.vis_size=opt.img_size
        self.vis_cir=2
        self.threshold=0.1
        self.batch=opt.pm_batch
        self.joint_num = opt.joint_num

        # log
        self.lr=opt.pm_g_lr
        self.lr_reset=opt.pm_lr_reset
        self.epoch=opt.pm_epoch
        self.ite_num=int(len(os.listdir(os.path.join(opt.data_dir,opt.joint_dir)))/opt.pm_batch)
        self.loss_list=[[] for i in range(6)]
        self.index_list=[]


    def setinput(self,input):

        self.input=input

        input_J_basic,input_J_inter=input[0][:4],input[0][4:]
        input_t=input[1]

        if len(self.gpu_ids) > 0:
            self.input_J_basic=[x.cuda() for x in input_J_basic]
            self.input_J_inter=[x.cuda() for x in input_J_inter]
            self.input_t=[x.cuda() for x in input_t]

    def reset_lr(self,epo):
        lr = self.lr * (0.5 ** (epo // self.lr_reset))
        for param_group in self.optimizer_g.param_groups:
            param_group['lr'] = lr

    def forward(self):

        self.pred_J = []

        for n in range(len(self.input_J_basic)):

            pred_J0=self.g(self.input_J_basic[n],self.input_J_basic[(n+1)%len(self.input_J_basic)],self.input_t[2*n])
            pred_J1=self.g(self.input_J_basic[n],self.input_J_basic[(n+1)%len(self.input_J_basic)],self.input_t[2*n+1])

            self.pred_J.append(pred_J0)
            self.pred_J.append(pred_J1)

        ''' L1 and L2 loss'''
        self.l1_loss=0
        self.l2_loss = 0
        for n in range(len(self.input_J_inter)):
            l1=self.l1_func(self.pred_J[n],self.input_J_inter[n])
            l2=self.l2_func(self.pred_J[n],self.input_J_inter[n])
            self.l1_loss=self.l1_loss+l1
            self.l2_loss = self.l2_loss + l2

        self.l1_loss=self.l1_loss/8.
        self.l2_loss = self.l2_loss / 8.

        ''' dis Loss '''
        self.t_tur=[]
        for n in range(len(self.input_t)):

            coeff=torch.rand(1)*2.-1.
            tur=torch.tensor(coeff*self.tur_eps).cuda()
            t_tur=self.input_t[n] +  tur

            t_tur = torch.where(t_tur< 0.0, torch.tensor(0.0).cuda(),t_tur) # min
            t_tur = torch.where(t_tur> 1.0, torch.tensor(1.0).cuda(), t_tur)  # max

            self.t_tur.append(t_tur)

        self.pred_J_tur = []

        # pred
        for n in range(len(self.input_J_basic)):

            pred_J0_tur=self.g(self.input_J_basic[n],self.input_J_basic[(n+1)%len(self.input_J_basic)], self.t_tur[2*n])
            pred_J1_tur=self.g(self.input_J_basic[n],self.input_J_basic[(n+1)%len(self.input_J_basic)], self.t_tur[2*n+1])

            self.pred_J_tur.append(pred_J0_tur)
            self.pred_J_tur.append(pred_J1_tur)

        # diff
        self.tur_loss=0
        for n in range(len(self.input_J_inter)):
            tur_loss = self.l1_func(self.pred_J[n], self.pred_J_tur[n])
            self.tur_loss = self.tur_loss + tur_loss

        self.tur_loss = self.tur_loss / 8.

        ''' grad loss '''
        self.grad_loss=0
        for n in range(len(self.input_J_basic)):

            step1=(1-self.input_t[2*n]).unsqueeze(1).unsqueeze(1).expand(-1,self.joint_num,self.vis_size[0],self.vis_size[1])
            grad1=(self.input_J_basic[n]-self.pred_J[2*n])/step1

            step2=(self.input_t[2*n]+1-self.input_t[2*n+1]).unsqueeze(1).unsqueeze(1).expand(-1,self.joint_num,self.vis_size[0],self.vis_size[1])
            grad2=(self.pred_J[2*n]-self.pred_J[2*n+1])/step2

            step3=(self.input_t[2*n+1]).unsqueeze(1).unsqueeze(1).expand(-1,self.joint_num,self.vis_size[0],self.vis_size[1])
            grad3=(self.pred_J[2*n+1]-self.input_J_basic[(n+1)%len(self.input_J_basic)])/step3

            grad_diff1=self.l1_func(grad2,grad1)
            grad_diff2=self.l1_func(grad3,grad2)

            self.grad_loss=self.grad_loss+grad_diff1+grad_diff2

        self.grad_loss=self.grad_loss/8.

        ''' all '''
        self.loss=self.l1_loss*self.l1_coeff+self.l2_loss*self.l2_coeff+self.grad_loss*self.grad_coeff+self.tur_loss*self.tur_coeff

        self.optimizer_g.zero_grad()
        self.loss.backward(retain_graph=False)
        self.optimizer_g.step()

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


        # joinr 18*2
        cords=np.concatenate([np.expand_dims(y_values, -1), np.expand_dims(x_values, -1)], axis=1)

        #
        joint_img=np.zeros(shape=(self.vis_size[0],self.vis_size[0],3))

        # joint tree
        LIMB_SEQ = [[1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7], [1, 8], [8, 9],
                    [9, 10], [1, 11], [11, 12], [12, 13], [1, 0], [0, 14], [14, 16],
                    [0, 15], [15, 17], [2, 16], [5, 17]]

        # color
        COLORS = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
                  [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
                  [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]

        for f, t in LIMB_SEQ:
            from_missing = cords[f][0] == -1 or cords[f][1] ==-1
            to_missing = cords[t][0] == -1 or cords[t][1] == -1
            if from_missing or to_missing:
                continue
            yy, xx, val = line_aa(cords[f][0], cords[f][1], cords[t][0], cords[t][1])
            joint_img[yy, xx] = np.expand_dims(val, 1) * 255

        for i, joint in enumerate(cords):
            if cords[i][0] == -1 or cords[i][1] == -1:
                continue
            yy, xx = circle(joint[0], joint[1], radius=self.vis_cir, shape=(self.vis_size[0],self.vis_size[1],3))
            joint_img[yy, xx] = COLORS[i]

        return joint_img


    def vis_result(self,epo,ite):

        # input
        vis_inp_path=[]
        for n in range(len(self.input_J_inter)):
            for m in range(self.batch):
                vis_inp_path.append(os.path.join(self.vis_dir,'E{}_I{}_B{}_No{}_Inp.jpg'.format(epo, ite, m,n)))
        # pred
        vis_pre_path=[]
        for n in range(len(self.pred_J)):
            for m in range(self.batch):
                vis_pre_path.append(os.path.join(self.vis_dir,'E{}_I{}_B{}_No{}_Pre.jpg'.format(epo, ite,m, n)))

        for n in range(len(self.input_J_inter)):

            # input
            joint_map=self.input_J_inter[n].detach().cpu().numpy()
            for m in range(self.batch):
                joint_img=self.map2img((joint_map[m].transpose([1,2,0])))
                cv.imwrite(vis_inp_path[n*self.batch+m],joint_img)

            # pred
            joint_map2=self.pred_J[n].detach().cpu().numpy()
            for m in range(self.batch):
                joint_img = self.map2img(joint_map2[m].transpose([1,2,0]))
                cv.imwrite(vis_pre_path[n*self.batch+m], joint_img)

    def log_print(self,epo,ite,time_bench):

        # 耗时
        elapsed = time .time() - time_bench
        elapsed = str(datetime.timedelta(seconds=elapsed))
        log = "Elapsed [{}], Epoch [{}/{}], Iter [{}/{}]".format(elapsed, epo+1,self.epoch, ite + 1, self.ite_num)

        # loss
        log += ",g_Loss: {:.4f}".format(self.loss.item())
        log += ", L1_loss: {:.4f}".format(self.l1_loss.item()*self.l1_coeff)
        log += ", L2_loss: {:.4f}".format(self.l2_loss.item() * self.l2_coeff)
        log += ", Grad_loss: {:.4f}".format(self.grad_loss.item()*self.grad_coeff)
        log += ", Tur_loss: {:.4f}".format(self.tur_loss.item() * self.tur_coeff)
        self.tur_loss * self.tur_coeff

        print(log)

    def plot_loss(self,epo,ite):

        ite_sum=epo* self.ite_num+ite
        self.index_list.append(ite_sum)

        loss_list=[self.loss.item(),self.l1_loss.item()*self.l1_coeff,self.l2_loss.item() * self.l2_coeff,self.grad_loss.item()*self.grad_coeff,self.tur_loss.item() * self.tur_coeff]
        loss_name=["g_loss","l1_loss","l2_loss","grad_loss","tur_loss"]

        for m in range(len(loss_list)):

            self.loss_list[m].append(loss_list[m])
            plt.figure()
            plt.plot(self.index_list,self.loss_list[m], 'b', label=loss_name[m])
            plt.ylabel(loss_name[m])
            plt.xlabel('iter_num')
            plt.legend()
            plt.savefig(os.path.join(self.loss_dir, "{}.jpg".format(loss_name[m])))

            plt.cla()
            plt.close("all")

    # 保存网络
    def save_network(self,epo):

        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        torch.save(self.g.state_dict(),os.path.join(self.checkpoint_dir, '{}_G.pth'.format(epo+1)))

    # 打印网络
    def print_network(self):

        model=self.g
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(model)
        print("The number of parameters: {}".format(num_params))







