import argparse


class Train_Options():

    def __init__(self):
        self.parser=argparse.ArgumentParser()

    def initialize(self,parser):

        # device
        parser.add_argument("--gpu_ids",type=str,default='0')
        parser.add_argument('--nums_works', type=int, default=1)
        parser.add_argument('--device', type=str, default='cuda')

        # dataset
        parser.add_argument('--img_size', type=int, default=(256,256))
        parser.add_argument('--shuffle', type=bool, default=True)
        parser.add_argument('--joint_num', type=int, default=18, help='joint num')
        parser.add_argument('--sg_num', type=int, default=7, help='parsing num')
        parser.add_argument('--sigma', type=int, default=100)

        # dir
        parser.add_argument('data_dir', type=str, default='data/train')
        parser.add_argument('--joint_interpolation_dir', type=str, default='joint_interpolation')
        parser.add_argument('--joint_dir', type=str, default='joint')
        parser.add_argument('--front_sg_dir', type=str, default='front_sg')
        parser.add_argument('--back_sg_dir', type=str, default='back_sg')
        parser.add_argument('--right_sg_dir', type=str, default='right_sg')
        parser.add_argument('--left_sg_dir', type=str, default='left_sg')
        parser.add_argument('--mid0_sg_dir', type=str, default='mid0_sg')
        parser.add_argument('--mid1_sg_dir', type=str, default='mid1_sg')
        parser.add_argument('--mid2_sg_dir', type=str, default='mid2_sg')
        parser.add_argument('--mid3_sg_dir', type=str, default='mid3_sg')
        parser.add_argument('--checkpoint_dir', type=str, default='checkpoint')
        parser.add_argument('--vis_dir', type=str, default='vis')
        parser.add_argument('--loss_dir', type=str, default='loss')

        # model
        parser.add_argument('model', type=str, default='',help='pm,aa,ct')
        parser.add_argument('--ngf', type=int, default=16)
        parser.add_argument('--grid_num', type=int, default=5)

        # training
        parser.add_argument('--log_print_ite', type=int, default=50)
        parser.add_argument('--log_vis_ite', type=int, default=200)
        parser.add_argument('--vis_ite', type=int, default=500)
        parser.add_argument('--save_epo', type=int, default=2)

        # pm
        parser.add_argument('--pm_batch', type=int, default=2)
        parser.add_argument('--pm_epoch', type=int, default=50)
        parser.add_argument('--pm_inference', type=str, default=None, help='checkpoint loading')
        parser.add_argument('--pm_g_lr', type=float, default=4e-4)
        parser.add_argument('--pm_lr_reset', type=int, default=12)

        parser.add_argument('--pm_bottle_nc', type=int, default=3)
        parser.add_argument('--pm_ngf', type=int, default=16)

        parser.add_argument('--pm_tur_eps', type=float, default=1e-2)
        parser.add_argument('--pm_l1_loss_coeff', type=float, default=10.0)  # 10.0
        parser.add_argument('--pm_l2_loss_coeff', type=float, default=30.0)  # 30.0
        parser.add_argument('--pm_tur_loss_coeff', type=float, default=1.5)  # 1.5
        parser.add_argument('--pm_grad_loss_coeff', type=float, default=0.35)   # 0.35

        # aa
        parser.add_argument('--aa_batch', type=int, default=2)
        parser.add_argument('--aa_epoch', type=int, default=100)
        parser.add_argument('--aa_inference', type=str, default=None, help='checkpoint loading')
        parser.add_argument('--aa_g_lr', type=float, default=4e-4)
        parser.add_argument('--aa_lr_reset', type=int, default=25)

        parser.add_argument('--aa_bottle_nc', type=int, default=3)
        parser.add_argument('--aa_ngf', type=int, default=16)

        parser.add_argument('--aa_l1_loss_coeff', type=float, default=8.0)
        parser.add_argument('--aa_ce_loss_coeff', type=float, default=0.5)


        # ct
        parser.add_argument('--ct_batch', type=int, default=1)
        parser.add_argument('--ct_epoch', type=int, default=150)
        parser.add_argument('--ct_inference', type=str, default=None, help='checkpoint loading')
        parser.add_argument('--ct_g_lr', type=float, default=4e-4)
        parser.add_argument('--ct_lr_reset', type=int, default=37)

        parser.add_argument('--ct_ngf', type=int, default=16)

        parser.add_argument('--ct_l1_loss_coeff', type=float, default=2.0)
        parser.add_argument('--ct_content_loss_coeff', type=float, default=0.3)
        parser.add_argument('--ct_style_loss_coeff', type=float, default=100)
        parser.add_argument('--ct_face_loss_coeff', type=float, default=300.0)
        parser.add_argument('--ct_cloth_loss_coeff', type=float, default=30.0)
        parser.add_argument('--ct_cloth_grad_loss_coeff', type=float, default=10.0)
        parser.add_argument('--ct_flow_loss_coeff', type=float, default=5.0)

        return parser

    def parse(self):

        parser=self.initialize(self.parser)
        opt,_=parser.parse_known_args()

        self.opt=opt

        return self.opt








