import os
import itertools
from network.function import  *
from torch.nn import init
from torchvision.utils import save_image
import datetime
import time
import matplotlib.pyplot as plt

class CTNet(nn.Module):

    def __init__(self,opt,input_nc,output_nc,ngf,norm_layer=nn.InstanceNorm2d, act=nn.LeakyReLU(0.2), use_spect=False):

        super(CTNet,self).__init__()

        self.opt=opt
        self.gpu_ids=opt.gpu_ids

        '''Sg'''
        # sg encoder
        self.encoder1_conv1 = BlockEncoder(input_nc[0],ngf*2,ngf,norm_layer,act,use_spect)
        self.encoder1_conv2 = BlockEncoder(ngf*2, ngf * 4, ngf*4, norm_layer, act, use_spect)
        self.encoder1_conv3 = BlockEncoder(ngf * 4, ngf * 8, ngf * 8, norm_layer, act, use_spect)
        self.encoder1_conv4 = BlockEncoder(ngf * 8, ngf * 16, ngf * 16, norm_layer, act, use_spect)

        # sg reduce channel
        self.sg_out_2 = nn.Sequential(nn.Conv2d(ngf * 8, ngf, kernel_size=3, stride=1, padding=1))  # 64-> 16
        self.sg_out_3 = nn.Sequential(nn.Conv2d(ngf * 16, ngf, kernel_size=3, stride=1, padding=1)) # 128-> 16
        self.sg_out_4 = nn.Sequential(nn.Conv2d(ngf * 32, ngf, kernel_size=3, stride=1, padding=1))  # 256 -> 16

        # flow
        self.flow_out = nn.Sequential(nn.Conv2d(ngf , 2, kernel_size=3, stride=1, padding=1),nn.Tanh())
        self.occ_mask = nn.Sequential(nn.Conv2d(ngf, 1, kernel_size=3, stride=1, padding=1, bias=True), nn.Sigmoid())

        ''' Body '''
        # Ib-> encoder 3 -> 32 -> 64 -> 128 -> 256
        self.encoder2_conv1 = BlockEncoder(input_nc[1], ngf * 2, ngf, norm_layer, act, use_spect)
        self.encoder2_conv2 = BlockEncoder(ngf * 2, ngf * 4, ngf * 4, norm_layer, act, use_spect)
        self.encoder2_conv3 = BlockEncoder(ngf * 4, ngf * 8, ngf * 8, norm_layer, act, use_spect)
        self.encoder2_conv4 = BlockEncoder(ngf * 8, ngf * 16, ngf * 16, norm_layer, act, use_spect)

        # warp tool
        self.warp=bilinear_warp

        # Img bottleneck
        self.bottle_neck1=BlockBottle(ngf * (16+4),ngf * 16)
        self.bottle_neck2 = BlockBottle(ngf * (16+4), ngf * 16)

        # Ib -> decoder
        self.decoder2_conv1 = ResBlockDecoder(ngf * 16, ngf * 8, ngf * 16, norm_layer, act, use_spect)
        self.decoder2_conv2 = ResBlockDecoder(ngf * 16, ngf * 4, ngf * 8, norm_layer, act, use_spect)
        self.decoder2_conv3 = ResBlockDecoder(ngf * 8, ngf * 2, ngf * 4, norm_layer, act, use_spect)
        self.decoder2_conv4 = ResBlockDecoder(ngf * 4, ngf, ngf*2, norm_layer, act, use_spect)

        # ouput
        self.output = Output(ngf, output_nc, 3, norm_layer, act, None)

        '''cloth'''
        # cloth encoder 3 -> 32 -> 64 -> 128 -> 256
        self.encoder3_conv1 = BlockEncoder(input_nc[1], ngf * 2, ngf, norm_layer, act, use_spect)
        self.encoder3_conv2 = BlockEncoder(ngf * 2, ngf * 4, ngf * 4, norm_layer, act, use_spect)
        self.encoder3_conv3 = BlockEncoder(ngf * 4, ngf * 8, ngf * 8, norm_layer, act, use_spect)
        self.encoder3_conv4 = BlockEncoder(ngf * 8, ngf * 16, ngf * 16, norm_layer, act, use_spect)

        # Ic -> decoder
        self.decoder3_conv1 = ResBlockDecoder(ngf * 16, ngf * 8, ngf * 16, norm_layer, act, use_spect)
        self.decoder3_conv2 = ResBlockDecoder(ngf * 16, ngf * 4, ngf * 8, norm_layer, act, use_spect)
        self.decoder3_conv3 = ResBlockDecoder(ngf * 8, ngf * 2, ngf * 4, norm_layer, act, use_spect)

        ''' cloth grid '''
        self.sober_conv=Sober_conv2d(dim_in=1,dim_out=1)
        self.encoder4_conv1 = BlockEncoder(input_nc[2], ngf * 2, ngf, norm_layer, act, use_spect)
        self.encoder4_conv2 = BlockEncoder(ngf * 2, ngf * 4, ngf * 4, norm_layer, act, use_spect)
        self.encoder4_conv3 = BlockEncoder(ngf * 4, ngf * 8, ngf * 8, norm_layer, act, use_spect)
        self.encoder4_conv4 = BlockEncoder(ngf * 8, ngf * 16, ngf * 16, norm_layer, act, use_spect)

        ''' style code '''
        self.style_conv1 = BlockEncoder(input_nc[0], ngf * 2, ngf, norm_layer, act, use_spect)  # 7*64*64 ->32*32*32
        self.style_conv2 = BlockEncoder(ngf * 2, ngf * 4, ngf * 4, norm_layer, act, use_spect)  # 32*32*32 -> 64*16*16

    def forward(self,sg0,sg1,img1):

        ''' s0+s1 -> flow '''
        # s0-> Enc
        s0_enc_1 = self.encoder1_conv1(sg0)  # B*32*128*128
        s0_enc_2 = self.encoder1_conv2(s0_enc_1)  # B*64*64*64
        s0_enc_3 = self.encoder1_conv3(s0_enc_2)  # B*128*32*32
        s0_enc_4 = self.encoder1_conv4(s0_enc_3)  # B*256*16*16

        # s1-> Enc
        s1_enc_1 = self.encoder1_conv1(sg1)  # B*32*128*128
        s1_enc_2 = self.encoder1_conv2(s1_enc_1)  # B*64*64*64
        s1_enc_3 = self.encoder1_conv3(s1_enc_2)  # B*128*32*32
        s1_enc_4 = self.encoder1_conv4(s1_enc_3)  # B*256*16*16

        # enc_2 enc_3 enc_4 output flow
        flow_enc2=self.sg_out_2(torch.cat((s0_enc_2,s1_enc_2),1))
        sg_out_2=flow_enc2
        flow_enc2=self.flow_out(flow_enc2)
        occ_mask = self.occ_mask(sg_out_2)

        flow_enc3 = self.sg_out_3(torch.cat((s0_enc_3, s1_enc_3), 1))
        flow_enc3 = self.flow_out(flow_enc3)

        flow_enc4 = self.sg_out_4(torch.cat((s0_enc_4, s1_enc_4), 1))
        flow_enc4 = self.flow_out(flow_enc4)

        '''cloth gradient'''
        # cloth_img
        cloth_mask = (sg0[:, 4, :, :] + sg0[:, 5, :, :]).unsqueeze(1)
        cloth_img=cloth_mask*img1

        cloth_gradient=self.sober_conv(cloth_img)
        cloth_gradient_enc_1=self.encoder4_conv1(cloth_gradient)
        cloth_gradient_enc_2=self.encoder4_conv2(cloth_gradient_enc_1)
        cloth_gradient_enc_2_warp=self.warp(cloth_gradient_enc_2,flow_enc2)
        cloth_gradient_enc_3 = self.encoder4_conv3(cloth_gradient_enc_2_warp)
        cloth_gradient_enc_3_warp = self.warp(cloth_gradient_enc_3, flow_enc3)

        ''' cloth '''
        # cloth_img -> enc
        cloth_enc_1 = self.encoder3_conv1(cloth_img)  # B*32*128*128
        cloth_enc_2 = self.encoder3_conv2(cloth_enc_1)  # B*64*64*64
        cloth_enc_2_warp=self.warp(cloth_enc_2,flow_enc2)

        cloth_enc_3 = self.encoder3_conv3(cloth_enc_2_warp)  # B*128*32*32
        cloth_enc_3_warp=self.warp(cloth_enc_3,flow_enc3)

        cloth_enc_4 = self.encoder3_conv4(cloth_enc_3_warp)  # B*256*16*16
        cloth_enc_4_warp = self.warp(cloth_enc_4, flow_enc4)

        # cloth_img -> dec
        cloth_dec_1=self.decoder3_conv1(cloth_enc_4_warp)
        cloth_dec_1_cat=torch.cat((cloth_dec_1+cloth_gradient_enc_3_warp,cloth_enc_3_warp+cloth_gradient_enc_3_warp),1)
        cloth_dec_2=self.decoder3_conv2(cloth_dec_1_cat)
        cloth_dec_2_cat=torch.cat((cloth_dec_2+cloth_gradient_enc_2_warp,cloth_enc_2_warp+cloth_gradient_enc_2_warp),1)
        cloth_dec_3=self.decoder3_conv3(cloth_dec_2_cat)

        # cloth_enc_2 -> style code
        b, c, h, w = cloth_enc_2.shape[0], cloth_enc_2.shape[1], cloth_enc_2.shape[2], cloth_enc_2.shape[3]
        sg0_interpolation = torch.nn.functional.interpolate(sg0, size=(h, w), scale_factor=None, mode='nearest',align_corners=None)
        cloth_style_code = []
        layer_list=[4,5]
        for l in range(len(layer_list)):
            fea_layer = sg0_interpolation[:, layer_list[l], :, :] * cloth_enc_2
            fea_code = torch.nn.functional.adaptive_avg_pool2d(fea_layer, (1, 1))
            fea_code = torch.squeeze(fea_code, -1)  # B*256*1
            cloth_style_code.append(fea_code)
        Cloth_style_code = torch.cat([fea for fea in cloth_style_code], dim=-1)  # B*256*2

        ''' Body '''
        # img1 -> enc
        non_cloth_mask = (sg0[:, 0, :, :] + sg0[:, 1, :, :]+sg0[:, 2, :, :]+sg0[:, 3, :, :]+sg0[:, 6,:, :]).unsqueeze(1)
        non_cloth_img = non_cloth_mask * img1
        body_enc_1=self.encoder2_conv1(non_cloth_img) # B*64*128*128
        body_enc_2 = self.encoder2_conv2(body_enc_1)# B*128*64*64
        body_enc_2_warp=self.warp(body_enc_2,flow_enc2)

        body_enc_3 = self.encoder2_conv3(body_enc_2_warp) # B*256*32*32
        body_enc3_warp=self.warp(body_enc_3,flow_enc3)

        body_enc_4 = self.encoder2_conv4(body_enc3_warp) # B*512*16*16
        body_enc4_warp = self.warp(body_enc_4, flow_enc4)

        # body_enc_2 -> style code
        b, c, h, w = body_enc_2.shape[0], body_enc_2.shape[1], body_enc_2.shape[2], body_enc_2.shape[3]
        sg0_interpolation = torch.nn.functional.interpolate(sg0, size=(h, w), scale_factor=None, mode='nearest',
                                                            align_corners=None)
        body_style_code = []
        layer_list = [0, 1, 2, 3, 6]
        for l in range(len(layer_list)):
            fea_layer = sg0_interpolation[:, layer_list[l], :, :] * body_enc_2
            fea_code = torch.nn.functional.adaptive_avg_pool2d(fea_layer, (1, 1))
            fea_code = torch.squeeze(fea_code, -1)  # B*256*1
            body_style_code.append(fea_code)
        Body_style_code = torch.cat([fea for fea in body_style_code], dim=-1)

        # reestablish sptial structure
        style_code = torch.cat((Body_style_code, Cloth_style_code), 2).permute(0, 2, 1)  # B*7*64
        b, c, h, w = s1_enc_2.shape[0], s1_enc_2.shape[1], s1_enc_2.shape[2], s1_enc_2.shape[3]  # B*64*64*64
        s1_enc_2_spatial = s1_enc_2.view(b, c, -1)  # B * 64* (64*64)
        spatial_style = torch.matmul(style_code, s1_enc_2_spatial)  # B * 7 * (64*64)
        spatial_style = spatial_style.view(b, -1, h, w)  # B * 7 * 64*64
        spatial_style = spatial_style * (1 - occ_mask)

        # reduce resize
        style_end = self.style_conv1(spatial_style)
        style_end = self.style_conv2(style_end)

        # body -> bot
        body_bot = torch.cat((style_end, body_enc4_warp), 1)
        body_bot = self.bottle_neck1(body_bot)
        body_bot = torch.cat((style_end, body_bot), 1)
        body_bot = self.bottle_neck2(body_bot)

        # img -> Dec
        body_dec_1 = self.decoder2_conv1(body_bot)  # B*256*32*32
        body_dec_1_cat=torch.cat((body_dec_1, cloth_dec_1),1)

        body_dec_2 = self.decoder2_conv2(body_dec_1_cat)  # B*128*16*16
        body_dec_2_cat = torch.cat((body_dec_2, cloth_dec_2), 1)

        body_dec_3 = self.decoder2_conv3(body_dec_2_cat)
        body_dec_3_cat=torch.cat((body_dec_3, cloth_dec_3), 1)

        body_dec_4 = self.decoder2_conv4(body_dec_3_cat)
        img_out = self.output(body_dec_4)

        s0_enc2_flow = self.warp(s0_enc_2, flow_enc2)
        s0_enc3_flow = self.warp(s0_enc_3, flow_enc3)
        s0_enc4_flow = self.warp(s0_enc_4, flow_enc4)

        s1_feature=[s0_enc_2,s1_enc_3,s1_enc_4]
        s0_flow=[s0_enc2_flow,s0_enc3_flow,s0_enc4_flow]

        return img_out,s1_feature,s0_flow

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



class CTNet_Train():

    def __init__(self,opt):

        super(CTNet_Train,self).__init__()

        self.gpu_ids=opt.gpu_ids
        self.device = opt.device
        self.ngf=opt.ct_ngf

        # dir
        self.checkpoint_dir = opt.checkpoint_dir + "\\" + "train" + "\\" + opt.model
        self.vis_dir = opt.vis_dir + "\\" + "train" + "\\" + opt.model
        self.loss_dir = opt.loss_dir + "\\" + "train" + "\\" + opt.model

        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        if not os.path.exists(self.vis_dir):
            os.makedirs(self.vis_dir)
        if not os.path.exists(self.loss_dir):
            os.makedirs(self.loss_dir)

        # generator
        self.g = CTNet(opt,input_nc=[opt.sg_num,3,1],output_nc=3,ngf= self.ngf)
        self.g.init_weights()

        if len(self.gpu_ids)>0 and torch.cuda.is_available():
            self.g.cuda()

        # load checkpoint
        self.inference=opt.ct_inference
        if self.inference:
            print("load checkpoint sucess!")
            self.g.load_state_dict(torch.load(os.path.join(self.checkpoint_dir,"%d_G.pth"%self.inference), map_location=torch.device(self.device)))

        # 优化器
        self.lr=opt.ct_g_lr
        self.lr_reset = opt.ct_lr_reset
        self.optimizer_g = torch.optim.Adam(itertools.chain(filter(lambda p: p.requires_grad, self.g.parameters())),lr=opt.ct_g_lr, betas=(0.9, 0.999))

        # 损失函数
        self.l1_func=torch.nn.L1Loss()
        self.l2_func=torch.nn.MSELoss()
        self.criterionGAN = GANLoss(use_lsgan=True, tensor=torch.cuda.FloatTensor)
        self.sober_conv=Sober_conv2d(dim_in=1,dim_out=1)
        self.vgg_func=VGG19().cuda()
        self.vgg_layer=['relu1_1','relu2_1','relu3_1','relu4_1']

        # 超参数
        self.l1_coeff=opt.ct_l1_loss_coeff
        self.content_coeff=opt.ct_content_loss_coeff
        self.style_coeff=opt.ct_style_loss_coeff
        self.face_coeff= opt.ct_face_loss_coeff
        self.cloth_coeff=opt.ct_cloth_loss_coeff
        self.cloth_grad_coeff=opt.ct_cloth_grad_loss_coeff
        self.flow_coeff = opt.ct_flow_loss_coeff

        # 可视化
        self.vis_size=opt.img_size
        self.batch=opt.ct_batch
        self.sg_num = opt.sg_num

        # log
        self.epoch=opt.ct_epoch
        self.ite_num=int(len(os.listdir(os.path.join(opt.data_dir,"front_img")))/opt.ct_batch)
        self.losses_list=[[] for i in range(10)]
        self.index_list=[]

    # 动态调整学习率
    def reset_lr(self, epo):
        lr = self.lr * (0.5 ** (epo // self.lr_reset))
        for param_group in self.optimizer_g.param_groups:
            param_group['lr'] = lr

    # 读入输入
    def setinput(self,input):

        self.input=input

        sg_sta_inp,sg_mid_inp,img_sta_inp,img_mid_inp=input[0][:4],input[0][4:],input[1][:4],input[1][4:]

        if len(self.gpu_ids) > 0:
            self.sg_sta_inp=[x.cuda() for x in sg_sta_inp]
            self.sg_mid_inp=[x.cuda() for x in sg_mid_inp]
            self.img_sta_inp=[x.cuda() for x in img_sta_inp]
            self.img_mid_inp=[x.cuda() for x in img_mid_inp]

    def forward(self):

        ''' 预测 '''

        # stard -> stard
        self.pred_img,self.s1_feature,self.s0_flow=[],[],[]
        for v in range(4):
            pred_img,s1_feature,s0_flow = self.g(self.sg_sta_inp[v],self.sg_sta_inp[(v+1)%4],self.img_sta_inp[v])
            self.pred_img.append(pred_img)
            self.s1_feature.append(s1_feature)
            self.s0_flow.append(s0_flow)

        # stard -> mid -> stard
        self.pred_img_mid,self.s1_feature_mid,self.s0_flow_mid = [],[],[]
        for v in range(4):
            pred_img_mid,s1_feature_mid,s0_flow_mid= self.g(self.sg_sta_inp[v],self.sg_mid_inp[v],self.img_sta_inp[v])
            self.pred_img_mid.append(pred_img_mid)
            self.s1_feature_mid.append(s1_feature_mid)
            self.s0_flow_mid.append(s0_flow_mid)

        ''' flow loss'''
        self.flow_loss=0.0
        for n in range(len(self.s1_feature)):
            flow_1,flow_2=0.0,0.0
            for m in range(len(self.s1_feature[n])):
                flow_1 += self.l1_func(self.s1_feature[n][m],self.s0_flow[n][m])
                flow_2 += self.l1_func(self.s1_feature_mid[n][m], self.s0_flow_mid[n][m])
            self.flow_loss+=flow_1+flow_2
        self.flow_loss=self.flow_loss/4.

        ''' L1 loss'''
        self.l1_loss=0.0
        for n in range(len(self.pred_img)):

            l1_1=self.l1_func(self.pred_img[n],self.img_sta_inp[(n+1)%4])
            l1_2= self.l1_func(self.pred_img_mid[n], self.img_mid_inp[n])

            self.l1_loss=self.l1_loss+l1_1+l1_2

        self.l1_loss=self.l1_loss/4.

        ''' content and style loss'''
        self.content_loss=0.0
        self.style_loss=0.0
        for n in range(len(self.pred_img)):

            content_1=self.vgg_cal(self.pred_img[n],self.img_sta_inp[(n+1)%4])
            content_2=self.vgg_cal(self.pred_img_mid[n], self.img_mid_inp[n])

            style_1=self.vgg_cal(self.pred_img[n],self.img_sta_inp[(n+1)%4],style=True)
            style_2 = self.vgg_cal(self.pred_img_mid[n], self.img_mid_inp[n],style=True)

            self.content_loss=self.content_loss+content_1+content_2
            self.style_loss=self.style_loss+style_1+style_2

        self.content_loss=self.content_loss/4.
        self.style_loss=self.style_loss/4.

        '''face loss'''
        self.face_loss = 0.0
        for n in range(len(self.pred_img)):
            sg_face_stard = self.sg_sta_inp[(n + 1) % 4][:, 2, :, :]
            sg_face_mid = self.sg_mid_inp[n][:, 2, :, :]

            sg_face_stard = sg_face_stard.unsqueeze(1)
            sg_face_mid = sg_face_mid.unsqueeze(1)

            face_1 = self.l2_func(self.pred_img[n] * sg_face_stard, self.img_sta_inp[(n + 1) % 4] * sg_face_stard)
            face_2 = self.l2_func(self.pred_img_mid[n] * sg_face_mid, self.img_mid_inp[n] * sg_face_mid)

            self.face_loss = self.face_loss + face_1 + face_2

        self.face_loss = self.face_loss / 4.

        '''cloth loss'''
        self.cloth_loss = 0.0
        self.cloth_grad_loss=0.0
        for n in range(len(self.pred_img)):
            sg_cloth_stard = self.sg_sta_inp[(n + 1) % 4][:, 4, :, :] + self.sg_sta_inp[(n + 1) % 4][:, 5, :, :]
            sg_cloth_mid = self.sg_mid_inp[n][:, 4, :, :] + self.sg_mid_inp[n][:, 5, :, :]

            sg_cloth_stard = sg_cloth_stard.unsqueeze(1)
            sg_cloth_mid = sg_cloth_mid.unsqueeze(1)

            cloth_1 = self.l2_func(self.pred_img[n] * sg_cloth_stard, self.img_sta_inp[(n + 1) % 4] * sg_cloth_stard)
            cloth_2 = self.l2_func(self.pred_img_mid[n] * sg_cloth_mid, self.img_mid_inp[n] * sg_cloth_mid)

            pre_grad=self.sober_conv(self.pred_img[n] * sg_cloth_stard)
            inp_grad=self.sober_conv(self.img_sta_inp[(n + 1) % 4] * sg_cloth_stard)
            pred_grad_mid=self.sober_conv(self.pred_img_mid[n] * sg_cloth_mid)
            inp_grad_mid=self.sober_conv(self.img_mid_inp[n] * sg_cloth_mid)

            cloth_grad_1=self.l1_func(pre_grad,inp_grad)
            cloth_grad_2=self.l1_func( pred_grad_mid,inp_grad_mid)

            self.cloth_loss = self.cloth_loss + cloth_1 + cloth_2
            self.cloth_grad_loss=self.cloth_grad_loss+cloth_grad_1+cloth_grad_2

        self.cloth_loss = self.cloth_loss / 4.
        self.cloth_grad_loss=self.cloth_grad_loss/4.

        ''' all '''
        self.loss=self.l1_loss*self.l1_coeff+self.face_loss*self.face_coeff+ self.cloth_loss *self.cloth_coeff+ \
                  self.content_loss*self.content_coeff+self.style_loss*self.style_coeff+self.flow_loss*self.flow_coeff

        self.loss_name=['g_loss',"l1_loss","face_loss","cloth_loss","content_loss","style_loss","flow_loss"]
        self.loss_list=[self.loss,self.l1_loss*self.l1_coeff,self.face_loss*self.face_coeff,self.cloth_loss *self.cloth_coeff,
                         self.content_loss*self.content_coeff,self.style_loss*self.style_coeff,self.flow_loss*self.flow_coeff]

        self.optimizer_g.zero_grad()
        self.loss.backward(retain_graph=False)
        self.optimizer_g.step()

    def de_norm(self, x):
        out = (x + 1) / 2
        return out.clamp(0, 1)

    # vgg_loss
    def vgg_cal(self,x,y,style=False):

        x_vgg = self.vgg_func(x)
        y_vgg = self.vgg_func(y)

        loss=0.0
        for l in range(len(self.vgg_layer)):
            if style:
                x_gram=compute_gram(x_vgg[self.vgg_layer[l]])
                y_gram=compute_gram(y_vgg[self.vgg_layer[l]])
                loss+=self.l1_func(x_gram,y_gram)
            else:
                loss+=self.l1_func(x_vgg[self.vgg_layer[l]],y_vgg[self.vgg_layer[l]])
        return loss

    def vis_result(self,epo,ite):

        view_list=['f','l','b',"r"]
        mid_list=['0','1','2','3']

        vis_inp_path=[]
        vis_inpm_path = []
        for n in range(len(self.pred_img)):
            for m in range(self.batch):
                vis_inp_path.append(os.path.join(self.vis_dir,'E{}_I{}_B{}_Inp_V{}.jpg'.format(epo, ite, m, view_list[n])))
                vis_inpm_path.append(os.path.join(self.vis_dir, 'E{}_I{}_B{}_Inp_M{}.jpg'.format(epo, ite, m, mid_list[n])))

        vis_pre_path=[]
        vis_prem_path = []
        for n in range(len(self.pred_img)):
            for m in range(self.batch):
                vis_pre_path.append( os.path.join(self.vis_dir, 'E{}_I{}_B{}_Pre_V{}.jpg'.format(epo, ite, m, view_list[n])))
                vis_prem_path.append(os.path.join(self.vis_dir, 'E{}_I{}_B{}_Pre_M{}.jpg'.format(epo, ite, m, mid_list[n])))

        for n in range(len(self.img_sta_inp)):


            img=Variable(self.img_sta_inp[(n+1)%4])
            img_mid=Variable(self.img_mid_inp[n])
            for m in range(self.batch):

                save_image(self.de_norm(img[m].data), vis_inp_path[n*self.batch+m], normalize=True)
                save_image(self.de_norm(img_mid[m].data), vis_inpm_path[n * self.batch + m], normalize=True)

            img2=Variable(self.pred_img[n])
            img_mid2 =Variable( self.pred_img_mid[n])
            for m in range(self.batch):
                save_image(self.de_norm(img2[m].data), vis_pre_path[n * self.batch + m], normalize=True)
                save_image(self.de_norm(img_mid2[m].data), vis_prem_path[n * self.batch + m], normalize=True)

    def log_print(self, epo, ite, time_bench):

        # 耗时
        elapsed = time.time() - time_bench
        elapsed = str(datetime.timedelta(seconds=elapsed))
        log = "Elapsed [{}], Epoch [{}/{}], Iter [{}/{}]".format(elapsed, epo + 1, self.epoch, ite + 1, self.ite_num)

        # loss
        for k in range(len(self.loss_list)):
            log += ", %s: %0.4f" % (self.loss_name[k], self.loss_list[k])

        print(log)

    def plot_loss(self, epo, ite):

        ite_sum = epo * self.ite_num + ite
        self.index_list.append(ite_sum)

        for m in range(len(self.loss_list)):
            self.losses_list[m].append(self.loss_list[m].item())
            plt.figure()
            plt.plot(self.index_list, self.losses_list[m], 'b', label=self.loss_name[m])
            plt.ylabel(self.loss_name[m])
            plt.xlabel('iter_num')
            plt.legend()
            plt.savefig(os.path.join(self.loss_dir, "{}.jpg".format(self.loss_name[m])))
            plt.cla()
            plt.close("all")
    # 保存网络
    def save_network(self,epo):

        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        torch.save(self.g.state_dict(),os.path.join(self.checkpoint_dir, '{}_G.pth'.format(epo+1)))

    # 打印网络
    def print_network(self):

        model=[self.g]

        num_params = 0
        for k in range(len(model)):
            for p in  model[k].parameters():
                num_params += p.numel()
        for k in range(len(model)):
            print(model[k])
        print("The number of parameters: {}".format(num_params))

