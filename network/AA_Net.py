import os
import itertools
from network.function import  *
from torch.nn import init
import cv2 as cv
import datetime
import time
import matplotlib.pyplot as plt



class AANet(nn.Module):

    def __init__(self,opt,input_nc,output_nc,ngf,norm_layer=nn.BatchNorm2d, use_spect=False):

        super(AANet,self).__init__()

        self.opt=opt
        self.gpu_ids=opt.gpu_ids

        # (Sg0+P0) encoder 18 -> 32 -> 64 -> 128 -> 256 -> 512
        self.encoder1_conv1=nn.Sequential(nn.Conv2d(input_nc[0], ngf*2, kernel_size=3,stride=2, padding=1, bias=False),
                                          nn.ReLU(True),
                                          ResidualBlock(ngf*2, norm_layer),
                                          ResidualBlock(ngf*2, norm_layer))
        self.encoder1_conv2=nn.Sequential(nn.Conv2d(ngf*2, ngf*4, kernel_size=3,stride=2, padding=1, bias=False),
                                          norm_layer(ngf*4),
                                          nn.ReLU(True),
                                          ResidualBlock(ngf*4, norm_layer),
                                          ResidualBlock(ngf*4, norm_layer))
        self.encoder1_conv3=nn.Sequential(nn.Conv2d(ngf*4, ngf*8, kernel_size=3,stride=2, padding=1, bias=False),
                                          norm_layer(ngf*8),
                                          nn.ReLU(True),
                                          ResidualBlock(ngf*8, norm_layer),
                                          ResidualBlock(ngf*8, norm_layer))
        self.encoder1_conv4=nn.Sequential(nn.Conv2d(ngf*8, ngf*16, kernel_size=3,stride=2, padding=1, bias=False),
                                          norm_layer(ngf*16),
                                          nn.ReLU(True),
                                          ResidualBlock(ngf*16, norm_layer),
                                          ResidualBlock(ngf*16, norm_layer))

        # Pt encoder
        self.encoder2_conv1=nn.Sequential(nn.Conv2d(input_nc[1], ngf*2, kernel_size=3,stride=2, padding=1, bias=False),
                                          nn.ReLU(True),
                                          ResidualBlock(ngf*2, norm_layer),
                                          ResidualBlock(ngf*2, norm_layer))
        self.encoder2_conv2=nn.Sequential(nn.Conv2d(ngf*2, ngf*4, kernel_size=3,stride=2, padding=1, bias=False),
                                          norm_layer(ngf*4),
                                          nn.ReLU(True),
                                          ResidualBlock(ngf*4, norm_layer),
                                          ResidualBlock(ngf*4, norm_layer))
        self.encoder2_conv3=nn.Sequential(nn.Conv2d(ngf*4, ngf*8, kernel_size=3,stride=2, padding=1, bias=False),
                                          norm_layer(ngf*8),
                                          nn.ReLU(True),
                                          ResidualBlock(ngf*8, norm_layer),
                                          ResidualBlock(ngf*8, norm_layer))
        self.encoder2_conv4=nn.Sequential(nn.Conv2d(ngf*8, ngf*16, kernel_size=3,stride=2, padding=1, bias=False),
                                          norm_layer(ngf*16),
                                          nn.ReLU(True),
                                          ResidualBlock(ngf*16, norm_layer),
                                          ResidualBlock(ngf*16, norm_layer))

        # bottleneck
        self.bottle_nc=opt.aa_bottle_nc # 瓶颈层的数量

        self.bottle1_neck1=BlockBottle(ngf * 16,ngf * 16)
        self.bottle1_neck2 = BlockBottle(ngf * 16, ngf * 16)
        self.bottle1_neck3 = BlockBottle(ngf * 16, ngf * 16)
        self.bottle_neck_sg=[self.bottle1_neck1,self.bottle1_neck2,self.bottle1_neck3]

        self.bottle2_neck1=BlockBottle(ngf * 16,ngf * 16)
        self.bottle2_neck2 = BlockBottle(ngf * 16, ngf * 16)
        self.bottle2_neck3 = BlockBottle(ngf * 16, ngf * 16)
        self.bottle_neck_pt=[self.bottle2_neck1,self.bottle2_neck2,self.bottle2_neck3]

        # norm
        self.img_size=opt.img_size
        self.adain1,self.adain2,self.adain3=[GetMatrix(ngf * 16,1),GetMatrix(ngf * 16,1),GetMatrix(ngf * 16,1)]
        self.AdaIN=[self.adain1,self.adain2,self.adain3] # beta gama
        self.feature_normal=feature_normalize

        # attention
        self.se_att=SEAttention(256)

        # decoder
        self.decoder_conv1=nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'),
                                         nn.Conv2d(ngf * 16, ngf*8 , kernel_size=3, stride=1, padding=1, bias=False),
                                         norm_layer(ngf * 8),
                                         nn.ReLU(True),
                                         ResidualBlock(ngf*8, norm_layer),
                                         ResidualBlock(ngf*8, norm_layer))
        self.decoder_conv2=nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'),
                                         nn.Conv2d(ngf * 8, ngf*4 , kernel_size=3, stride=1, padding=1, bias=False),
                                         norm_layer(ngf * 4),
                                         nn.ReLU(True),
                                         ResidualBlock(ngf*4, norm_layer),
                                         ResidualBlock(ngf*4, norm_layer))
        self.decoder_conv3=nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'),
                                         nn.Conv2d(ngf * 4, ngf*2 , kernel_size=3, stride=1, padding=1, bias=False),
                                         norm_layer(ngf * 2),
                                         nn.ReLU(True),
                                         ResidualBlock(ngf*2, norm_layer),
                                         ResidualBlock(ngf*2, norm_layer))
        self.decoder_conv4=nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'),
                                         nn.Conv2d(ngf*2, 7, kernel_size=3, stride=1, padding=1, bias=False),
                                         nn.Softmax(dim=1))


    def forward(self,sg_1,p_t):

        # sg_1+p_1 -> encoder
        sg1_enc=self.encoder1_conv1(sg_1) # B*64*128*128
        sg2_enc = self.encoder1_conv2(sg1_enc) # B*128*64*64
        sg3_enc = self.encoder1_conv3(sg2_enc) # B*256*32*32
        sg4_enc = self.encoder1_conv4(sg3_enc) # B*512*16*16

        # pt -> encoder
        pt1_enc=self.encoder2_conv1(p_t)
        pt2_enc = self.encoder2_conv2(pt1_enc)
        pt3_enc = self.encoder2_conv3(pt2_enc)
        pt4_enc= self.encoder2_conv4(pt3_enc)

        # bottleneck
        sg1_bot,pt1_bot=sg4_enc,pt4_enc

        beta1, gama1 = self.AdaIN[0](pt1_bot.clone())
        sg1_norm = self.feature_normal(sg1_bot.clone())
        sg1_adain = sg1_norm * (1 + beta1) + gama1
        sg2_bot = self.bottle_neck_sg[0](sg1_adain)
        pt2_bot = self.bottle_neck_pt[0](pt1_bot)

        beta2, gama2 = self.AdaIN[1](pt2_bot.clone())
        sg2_norm = self.feature_normal(sg2_bot)
        sg2_adain = sg2_norm * (1 + beta2) + gama2
        sg3_bot = self.bottle_neck_sg[1](sg2_adain)
        pt3_bot = self.bottle_neck_pt[1](pt2_bot.clone())

        beta3, gama3 = self.AdaIN[0](pt3_bot.clone())
        sg3_norm = self.feature_normal(sg3_bot)
        sg3_adain = sg3_norm * (1 + beta3) + gama3
        sg4_bot = self.bottle_neck_sg[2](sg3_adain)
        pt4_bot = self.bottle_neck_pt[2](pt3_bot.clone())

        # sg1_enc <-> sg1_bot attention
        fea_att=self.se_att(sg4_bot)

        # decoder
        sg1_dec=self.decoder_conv1(fea_att)
        sg2_dec = self.decoder_conv2(sg1_dec)
        sg3_dec = self.decoder_conv3(sg2_dec)
        output_par = self.decoder_conv4(sg3_dec)

        return output_par


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


class AANet_Train():

    def __init__(self,opt):

        super(AANet_Train,self).__init__()

        self.gpu_ids=opt.gpu_ids
        self.device = opt.device
        self.ngf=opt.aa_ngf
        self.img_size=opt.img_size

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
        self.g = AANet(opt,input_nc=[opt.sg_num,opt.joint_num],output_nc=opt.sg_num,ngf= self.ngf)
        self.g.init_weights()

        if len(self.gpu_ids)>0 and torch.cuda.is_available():
            self.g.cuda()

        # load checkpoint
        self.inference=opt.aa_inference
        if self.inference:
            print("load checkpoint sucess!")
            self.g.load_state_dict(torch.load(os.path.join(self.checkpoint_dir,"%d_G.pth"%self.inference), map_location=torch.device(self.device)))

        # 优化器
        self.lr=opt.aa_g_lr
        self.lr_reset=opt.aa_lr_reset
        self.optimizer_g = torch.optim.Adam(itertools.chain(filter(lambda p: p.requires_grad, self.g.parameters())),lr=opt.aa_g_lr, betas=(0.9, 0.999))

        # 损失函数
        self.l1_func=torch.nn.L1Loss()
        self.l2_func=torch.nn.MSELoss()

        self.ce_weight=torch.tensor([1.0,10.0,20.0,10.0,1.0,1.0,10.0]).cuda()
        self.ce_func=torch.nn.CrossEntropyLoss(weight=self.ce_weight)

        self.criterionGAN = GANLoss(use_lsgan=True, tensor=torch.cuda.FloatTensor)
        self.l1_coeff=opt.aa_l1_loss_coeff
        self.ce_coeff = opt.aa_ce_loss_coeff

        # 可视化
        self.batch=opt.aa_batch
        self.vis_size=opt.img_size
        self.sg_num = opt.sg_num

        # log
        self.epoch=opt.aa_epoch
        self.ite_num=int(len(os.listdir(os.path.join(opt.data_dir,opt.front_sg_dir)))/opt.aa_batch)
        self.loss_list=[[] for i in range(10)]
        self.index_list=[]

    def setinput(self,input):

        self.input=input

        input_joint,input_joint_mid,\
        input_sg,input_sg_mid,\
        input_sg_label,input_sg_mid_label=input[0][:4],input[0][4:],input[1][:4],input[1][4:],input[2][:4],input[2][4:]

        if len(self.gpu_ids) > 0:
            self.input_joint=[x.cuda() for x in input_joint]
            self.input_sg=[x.cuda() for x in input_sg]
            self.input_sg_mid=[x.cuda() for x in input_sg_mid]
            self.input_joint_mid=[x.cuda() for x in input_joint_mid]
            self.input_sg_label=[x.cuda() for x in input_sg_label]
            self.input_sg_mid_label = [x.cuda() for x in input_sg_mid_label]

    def reset_lr(self,epo):
        lr = self.lr * (0.5 ** (epo // self.lr_reset))
        for param_group in self.optimizer_g.param_groups:
            param_group['lr'] = lr

    def forward(self):

        # stard -> stard
        self.pred_sg=[]
        for v in range(4):
            pred_sg = self.g(self.input_sg[v],  self.input_joint[(v+1)%4])
            self.pred_sg.append(pred_sg)

        # anti
        self.pred_sg_anti = []
        for v in range(4):
            pred_sg_anti = self.g(self.input_sg[(v + 1) % 4], self.input_joint[v])
            self.pred_sg_anti.append(pred_sg_anti)

        # stard -> mid -> stard
        self.pred_sg_mid = []
        for v in range(4):
            pred_sg_mid = self.g(self.input_sg[v], self.input_joint_mid[v])  # stard-mid
            self.pred_sg_mid.append(pred_sg_mid)

        # anti
        self.pred_sg_mid_anti = []
        for v in range(4):
            pred_sg_mid_anti = self.g(self.input_sg[(v + 1) % 4], self.input_joint_mid[v])  # stard-mid
            self.pred_sg_mid_anti.append(pred_sg_mid_anti)

        ''' L1 loss'''
        self.l1_loss=0.0
        for n in range(len(self.pred_sg)):

            l1_1=self.l1_func(self.pred_sg[n],self.input_sg[(n+1)%4])
            l1_2=self.l1_func(self.pred_sg_anti[(n+1)%4],self.input_sg[(n+1)%4])

            l1_3 = self.l1_func(self.pred_sg_mid[n], self.input_sg_mid[n])
            l1_4 = self.l1_func(self.pred_sg_mid_anti[n], self.input_sg_mid[n])

            self.l1_loss=self.l1_loss+l1_1+l1_2+l1_3+l1_4

        self.l1_loss=self.l1_loss/4.

        ''' ce loss '''
        self.ce_loss=0.0
        for n in range(len(self.pred_sg)):

            sg_label=self.input_sg_label[(n+1)%4].squeeze(1).long()
            ce_1=self.ce_func(self.pred_sg[n],sg_label)
            ce_2 = self.ce_func(self.pred_sg_anti[(n+1)%4], sg_label)

            sg_mid_label=self.input_sg_mid_label[n].squeeze(1).long()
            ce_3 = self.ce_func(self.pred_sg_mid[n], sg_mid_label)
            ce_4 = self.ce_func(self.pred_sg_mid_anti[n], sg_mid_label)

            self.ce_loss=self.ce_loss+ce_1+ce_2+ce_3+ce_4

        self.ce_loss=self.ce_loss/4.


        ''' all '''
        self.loss=self.l1_loss*self.l1_coeff+self.ce_loss*self.ce_coeff

        self.optimizer_g.zero_grad()
        self.loss.backward(retain_graph=False)
        self.optimizer_g.step()

    def par2img(self,par_map,vis=True):

        par_all=torch.cuda.FloatTensor(size=(self.batch,3,self.img_size[0],self.img_size[1]))


        for n in range(self.batch):

            if vis:
                vis_color=torch.cuda.FloatTensor([(0,0,0),(255,255,0),(0,255,255),(255,0,255),(0,0,255),(0,255,0),(255,0,0),(244,164,96),(255,255,255),(160,32,240)]).long()
            else:
                vis_color = torch.cuda.FloatTensor(
                    [(0, 0, 0), (1, 1, 1), (2, 2, 2), (3, 3, 3), (4, 4, 4), (5, 5, 5), (6, 6, 6),
                     (7, 7, 7), (8, 8, 8), (9, 9, 9)]).long()


            index=torch.argmax(par_map[n],dim=0).view(-1)
            ones = vis_color.index_select(0, index)
            par=ones.view(self.img_size[0],self.img_size[1],3).long().permute(2,0,1)
            par_all[n]=par

        return par_all

    def joint_vis(self,joint):
        joint=torch.sum(joint,dim=1)*255
        joint=joint.unsqueeze(1).repeat(1,3,1,1)
        return joint

    def vis_result(self,epo,ite):

        self.input_sg_vis=torch.cuda.FloatTensor(size=(self.batch,4,3,self.img_size[0],self.img_size[1]))
        self.input_sg_mid_vis = torch.cuda.FloatTensor(size=(self.batch,4,3,self.img_size[0],self.img_size[1]))

        self.pred_sg_vis=torch.cuda.FloatTensor(size=(self.batch,4,3,self.img_size[0],self.img_size[1]))
        self.pred_sg_mid_vis=torch.cuda.FloatTensor(size=(self.batch,4,3,self.img_size[0],self.img_size[1]))

        self.input_joint_vis=torch.cuda.FloatTensor(size=(self.batch,4,3,self.img_size[0],self.img_size[1]))
        self.input_joint_mid_vis=torch.cuda.FloatTensor(size=(self.batch,4,3,self.img_size[0],self.img_size[1]))

        for n in range(len(self.input_sg)):

            self.input_sg_vis[:,n,:,:,:]=self.par2img(self.input_sg[n])
            self.input_sg_mid_vis[:,n,:,:,:]=self.par2img(self.input_sg_mid[n])

            self.pred_sg_vis[:,n,:,:,:]=self.par2img(self.pred_sg[n])
            self.pred_sg_mid_vis[:,n,:,:,:]=self.par2img(self.pred_sg_mid[n])

            self.input_joint_vis[:,n,:,:,:]=self.joint_vis(self.input_joint[n])
            self.input_joint_mid_vis[:,n,:,:,:]=self.joint_vis(self.input_joint_mid[n])

        for m in range(self.batch):

            vis_path = os.path.join(self.vis_dir, 'E{}_I{}_B{}.jpg'.format(epo, ite, m))

            sg=torch.cat([x for x in self.input_sg_vis[m]],dim=-1)
            sg_mid=torch.cat([x for x in self.input_sg_mid_vis[m]],dim=-1)
            joint = torch.cat([x for x in self.input_joint_vis[m]], dim=-1)
            joint_mid= torch.cat([x for x in self.input_joint_mid_vis[m]], dim=-1)
            pred=torch.cat([x for x in self.pred_sg_vis[m]],dim=-1)
            pred_vis=torch.cat([x for x in self.pred_sg_mid_vis[m]],dim=-1)


            all=torch.cat((sg,sg_mid,joint,joint_mid,pred,pred_vis),dim=-2).permute(1,2,0)
            cv.imwrite(vis_path, all.cpu().numpy())

    def log_print(self,epo,ite,time_bench):

        # 耗时
        elapsed = time .time() - time_bench
        elapsed = str(datetime.timedelta(seconds=elapsed))
        log = "Elapsed [{}], Epoch [{}/{}], Iter [{}/{}]".format(elapsed, epo+1,self.epoch, ite + 1, self.ite_num)

        # loss
        log += ",g_Loss: {:.4f}".format(self.loss.item())
        log+=", L1_loss: {:.4f}".format(self.l1_loss.item()*self.l1_coeff)
        log += ", CE_loss: {:.4f}".format(self.ce_loss.item() * self.ce_coeff)

        print(log)

    # 画损失函数
    def plot_loss(self,epo,ite):

        ite_sum=epo* self.ite_num+ite
        self.index_list.append(ite_sum)

        loss_list=[self.loss.item(),self.l1_loss.item()*self.l1_coeff,
                   self.ce_loss.item() * self.ce_coeff]

        loss_name=["g_loss","l1_loss","ce_loss"]

        for m in range(len(loss_list)):

            self.loss_list[m].append(loss_list[m])
            plt.figure()
            plt.plot(self.index_list,self.loss_list[m], 'b', label=loss_name[m])
            plt.ylabel(loss_name[m])
            plt.xlabel('iter_num')
            plt.legend()
            plt.savefig(os.path.join(self.loss_dir, "{}.jpg".format(loss_name[m])))

    # 保存网络
    def save_network(self,epo):

        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        torch.save(self.g.state_dict(),os.path.join(self.checkpoint_dir, '{}_G.pth'.format(epo+1)))

    # 打印网络
    def print_network(self):

        model1=self.g
        num_params = 0
        for p in model1.parameters():
            num_params += p.numel()
        print(model1)
        print("The number of parameters: {}".format(num_params))






