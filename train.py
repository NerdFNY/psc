import torch
import time
from options.train_options import Train_Options
from dataset.train_dataset import *
from network.psc_model import *
import torch.utils.data

def train(opt):

    if opt.model=="pm":

        pm_dataset = PMdataset()
        pm_dataset.initialize(opt)

        pm_dataset_loader = torch.utils.data.DataLoader(
            pm_dataset,
            batch_size=opt.pm_batch,
            shuffle=opt.shuffle,
            num_workers=int(opt.nums_works))


        pm_model=PMNet_Train(opt)
        pm_model.print_network()

        if opt.pm_inference==None:
            epoch_start=0
        else:
            epoch_start = opt.pm_inference
        time_bench = time.time()

        for i in range(epoch_start, opt.pm_epoch):
            iter = 0
            for index, data in enumerate(pm_dataset_loader):
                iter += 1

                pm_model.setinput(data)
                pm_model.reset_lr(i)
                pm_model.forward()

                # vis result
                if iter % opt.vis_ite == 0:
                    pm_model.vis_result(i, iter)

                # print loss
                if iter % opt.log_print_ite == 0:
                    pm_model.log_print(i, iter, time_bench)

                # vis loss
                if iter % opt.log_vis_ite == 0:
                    pm_model.plot_loss(i, iter)

            # save network
            if i % opt.save_epo == 0:
                pm_model.save_network(i)

    elif opt.model == "aa":

        aa_dataset = AAdataset()
        aa_dataset.initialize(opt)
        aa_dataset_loader = torch.utils.data.DataLoader(
            aa_dataset,
            batch_size=opt.aa_batch,
            shuffle=opt.shuffle,
            num_workers=int(opt.nums_works))

        aa_model = AANet_Train(opt)
        aa_model.print_network()

        if opt.pm_inference == None:
            epoch_start = 0
        else:
            epoch_start = opt.aa_inference
        time_bench = time.time()

        for i in range(epoch_start, opt.aa_epoch):
            iter = 0
            for index, data in enumerate(aa_dataset_loader):
                iter += 1

                aa_model.setinput(data)
                aa_model.reset_lr(i)
                aa_model.forward()

                # vis result
                if iter % opt.vis_ite == 0:
                    aa_model.vis_result(i, iter)

                # print loss
                if iter % opt.log_print_ite == 0:
                    aa_model.log_print(i, iter, time_bench)

                # vis loss
                if iter % opt.log_vis_ite == 0:
                    aa_model.plot_loss(i, iter)

            # save network
            if i % opt.save_epo == 0:
                aa_model.save_network(i)

    elif opt.model == "ct":

        ct_dataset = CTdataset()
        ct_dataset.initialize(opt)
        ct_dataset_loader = torch.utils.data.DataLoader(
            ct_dataset,
            batch_size=opt.ct_batch,
            shuffle=opt.shuffle,
            num_workers=int(opt.nums_works))

        ct_model = CTNet_Train(opt)
        ct_model.print_network()

        if opt.pm_inference == None:
            epoch_start = 0
        else:
            epoch_start = opt.ct_inference
        time_bench = time.time()

        for i in range(epoch_start, opt.ct_epoch):
            iter = 0
            for index, data in enumerate(ct_dataset_loader):
                iter += 1

                ct_model.setinput(data)
                ct_model.reset_lr(i)
                ct_model.forward()

                # vis result
                if iter % opt.vis_ite == 0:
                    ct_model.vis_result(i, iter)

                # print loss
                if iter % opt.log_print_ite == 0:
                    ct_model.log_print(i, iter, time_bench)

                # vis loss
                if iter % opt.log_vis_ite == 0:
                    ct_model.plot_loss(i, iter)

            # save network
            if i % opt.save_epo == 0:
                ct_model.save_network(i)


if __name__=="__main__" :
    opt=Train_Options().parse()
    train(opt)