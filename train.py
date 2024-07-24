from random import seed
import time

import torch
import os
import json
import matplotlib.pyplot as plt
import numpy as np
from torch import optim
from torch.utils.data import DataLoader
from models.Recon_subnetwork import UNetModel, update_ema_params
from models.Seg_subnetwork import SegmentationSubNetwork
from tqdm import tqdm
import torch.nn as nn
from data.dataset_beta_thresh import MVTecTrainDataset,MVTecTestDataset,MVTec3DTrainDataset,MVTec3DTestDataset
from math import exp
import torch.nn.functional as F
from models.DDPM import GaussianDiffusionModel, get_beta_schedule
from scipy.ndimage import gaussian_filter
from skimage.measure import label, regionprops
from sklearn.metrics import roc_auc_score,auc,average_precision_score
import pandas as pd
from collections import defaultdict

def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)    

def defaultdict_from_json(jsonDict):
    func = lambda: defaultdict(str)
    dd = func()
    dd.update(jsonDict)
    return dd

class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=4, logits=False, reduce=True): 
        super(BinaryFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss

def mean_flat(tensor):
    return torch.mean(tensor, dim=list(range(1, len(tensor.shape))))

def train(training_dataset_loader, testing_dataset_loader, args, data_len,sub_class,class_type,device ):
    # 设置图片通道，默认是3，即RGB
    in_channels = args["channels"]
    # 设置Unet模型，用于预测异常掩膜

    unet_model = UNetModel(args['img_size'][0], args['base_channels'], channel_mults=args['channel_mults'], dropout=args[
                "dropout"], n_heads=args["num_heads"], n_head_channels=args["num_head_channels"],
            in_channels=in_channels
            ).to(device)

    seg_model = SegmentationSubNetwork(in_channels=8, out_channels=1).to(device)

    if args['get_features']:
        unet_model_t = UNetModel(args['img_size'][0] / 4, args['base_channels'], channel_mults=args['channel_mults'], dropout=args[
                "dropout"], n_heads=args["num_heads"], n_head_channels=args["num_head_channels"],
            in_channels=(args["f_channels"]  + args["down_sample_scale"] ** 2)
            ).to(device)

        seg_model_t = SegmentationSubNetwork(in_channels=2 * (args["f_channels"]  + args["down_sample_scale"] ** 2), out_channels=1).to(device)


    # 这个beta就是扩散过程前需要指定的那个值，一般从0.01-0.2线性增加，也可以用余弦函数来生成
    betas = get_beta_schedule(args['T'], args['beta_schedule'])

    # 生成ddpm采样模型
    # "loss_type": "l2",
    # "noise_fn":"gauss",
    # “in_channels”: 3
    ddpm_sample =  GaussianDiffusionModel(
            args['img_size'], betas, loss_weight=args['loss_weight'],
            loss_type=args['loss-type'], noise=args["noise_fn"], img_channels=in_channels,
            )

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPU(s)")
        unet_model = nn.DataParallel(unet_model)
        seg_model = nn.DataParallel(seg_model)
        if args["get_features"]:
            unet_model_t = nn.DataParallel(unet_model_t)
            seg_model_t = nn.DataParallel(seg_model_t)

    # 因为unet就是用来去噪的模型，ddpm训练的参数也就是去噪模型Unet的参数
    optimizer_ddpm = optim.Adam( unet_model.parameters(), lr=args['diffusion_lr'],weight_decay=args['weight_decay'])
    
    optimizer_seg = optim.Adam(seg_model.parameters(),lr=args['seg_lr'],weight_decay=args['weight_decay'])
    optimizer_ddpm_t = optim.Adam(unet_model_t.parameters(), lr=args['diffusion_lr'],
                                  weight_decay=args['weight_decay'])
    optimizer_seg_t = optim.Adam(seg_model_t.parameters(), lr=args['seg_lr'], weight_decay=args['weight_decay'])

    # 计算预测掩膜和真实掩膜的距离
    loss_focal = BinaryFocalLoss().to(device)
    loss_smL1= nn.SmoothL1Loss().to(device)

    

    tqdm_epoch = range(0, args['EPOCHS'] )
    # PyTorch 中的一个学习率调度器类,它会根据余弦函数的曲线逐步降低学习率。
    # 就是动态调整optimizer_seg这个学习器的学习速率
    scheduler_seg =optim.lr_scheduler.CosineAnnealingLR(optimizer_seg, T_max=10, eta_min=0, last_epoch=- 1, verbose=False)
    scheduler_seg_t =optim.lr_scheduler.CosineAnnealingLR(optimizer_seg_t, T_max=10, eta_min=0, last_epoch=- 1, verbose=False)


    # dataset loop
    # 每10步记录一次各种损失
    train_loss_list=[]
    train_noise_loss_list=[]
    train_focal_loss_list=[]
    train_smL1_loss_list=[]

    # 记录loss对应的epoch
    loss_x_list=[]

    # 记录最好的结果以及对应的epoch
    best_image_auroc=0.0
    best_pixel_auroc=0.0
    best_epoch=0

    # 每50步记录一下auroc的值，并存入记录auroc时相应的epoch到performance_x_list`
    image_auroc_list=[]
    pixel_auroc_list=[]
    performance_x_list=[]
    
    for epoch in tqdm_epoch:
        unet_model.train()
        seg_model.train()
        train_loss = 0.0
        train_focal_loss=0.0
        train_smL1_loss = 0.0
        train_noise_loss = 0.0
        tbar = tqdm(training_dataset_loader)
        for i, sample in enumerate(tbar):
            # 获取训练图片、掩膜、标签
            # 返回增强后的图像、掩膜矩阵、重新标签的异常标签（因为训练时候只有正常图像，而我们随机对其了异常增强）
            # augmented_image, msk, np.array([has_anomaly], dtype=np.float32)
            # (2 ,3 ,256, 256)
            image_original = sample['image'].to(device)
            # (2 ,3 ,256, 256)
            aug_image=sample['aug_img'].to(device)
            # (2, 1, 256, 256)
            depth = sample["depth"].to(device)
            # (2, 1, 256, 256)
            aug_depth = sample["aug_depth"].to(device)
            # (2, 1, 256, 256)
            anomaly_mask = sample["anomaly_msk"].to(device)
            # (2, 1, 256, 256)
            normal_mask = sample["normal_msk"].to(device)
            # (2,)
            anomaly_label_aug = sample["is_anomaly"].to(device).squeeze()

            anomaly_label_original = 0

            if args["get_features"]:
                features_t = sample["features_t"].to(device)
                depth_t = sample["depth_t"].to(device)
                features_aug_t = sample["features_aug_t"].to(device)
                depth_aug_t = sample["depth_aug_t"].to(device)

                feature_t_original = torch.cat((features_t, depth_t), dim=1)
                aug_feature_t = torch.cat((features_aug_t, depth_aug_t), dim=1)



            image_original = torch.cat((image_original, depth), dim=1)
            aug_image = torch.cat((aug_image, aug_depth), dim=1)


            # noise_loss对应公式（9）
            # normal_t是从[1,,τ]里随机出的时间步
            # start = time.time()
            # (),(2,3,256,256),(2,),(2,3,256,256),(2,3,256,256) 去噪过程花了11秒
            noise_loss_aug, pred_x0_aug,t_aug,x_t_aug = ddpm_sample.one_step_denoising(unet_model, aug_image, anomaly_label_aug,args)
            noise_loss_original, pred_x0_original, t_original, x_t_original = ddpm_sample.one_step_denoising(unet_model, image_original, anomaly_label_original, args)
            if args["get_features"]:
                noise_loss_f_aug, pred_f0_aug, t_aug, f_t_aug = ddpm_sample.one_step_denoising(unet_model_t, aug_feature_t,
                                                                                             anomaly_label_aug, args)
                noise_loss_f_original, pred_f0_original, t_original, f_t_original = ddpm_sample.one_step_denoising(
                    unet_model_t, feature_t_original, anomaly_label_original, args)
                f_noise_loss = noise_loss_f_aug + noise_loss_f_original

            # stop1 = time.time()

            # (2,1,256,256) 分割图预测不到0.1秒
            pred_mask_ano = seg_model(torch.cat((aug_image, pred_x0_aug), dim=1))
            pred_mask_nor = seg_model(torch.cat((image_original, pred_x0_original), dim=1))
            if args["get_features"]:
                pred_fmask_ano = seg_model_t(torch.cat((aug_feature_t, pred_f0_aug), dim=1))
                pred_fmask_nor = seg_model_t(torch.cat((feature_t_original, pred_f0_original), dim=1))
                pred_fmask_ano = F.interpolate(pred_fmask_ano, size=args["img_size"], mode='bilinear', align_corners=False)
                pred_fmask_nor = F.interpolate(pred_fmask_nor, size=args["img_size"], mode='bilinear',
                                               align_corners=False)
                pred_mask_nor = (pred_mask_nor + pred_fmask_nor) / 2
                pred_mask_ano = (pred_mask_ano + pred_fmask_ano) / 2
            # stop = time.time()
            # print("\n运行时间：************************：" + str(stop1 - start))
            # print("\n运行时间：************************：" + str(stop - stop1))

            #loss
            # Lmask损失
            # 标量（）
            noise_loss = noise_loss_aug + noise_loss_original + f_noise_loss
            l2_originalAndAug_loss = mean_flat(pred_x0_original-pred_x0_aug).square().mean()

            focal_mask_loss = loss_focal(pred_mask_ano,anomaly_mask) + loss_focal(pred_mask_nor,normal_mask)
            # 标量()
            smL1_mask_loss = loss_smL1(pred_mask_ano, anomaly_mask) + loss_smL1(pred_mask_nor, normal_mask)
            # 标量()
            loss = noise_loss + 5*focal_mask_loss + smL1_mask_loss + l2_originalAndAug_loss
            
            optimizer_ddpm.zero_grad()
            optimizer_seg.zero_grad()
            optimizer_ddpm_t.zero_grad()
            optimizer_seg_t.zero_grad()
            loss.backward()

            optimizer_ddpm.step()
            optimizer_seg.step()
            optimizer_ddpm_t.step()
            optimizer_seg_t.step()
            scheduler_seg.step()
            scheduler_seg_t.step()

            # train loss是每一个sample都会叠加的
            train_loss += loss.item()
            tbar.set_description('Epoch:%d, Train loss: %.3f' % (epoch, train_loss))

            train_smL1_loss += smL1_mask_loss.item() #+ smL1_originalAndAug_loss
            train_focal_loss+=5*focal_mask_loss.item() #+ 0.5*focal_originalAndAug_loss
            train_noise_loss+=noise_loss.item()
            

        if epoch % 10 ==0  and epoch > 0:
            train_loss_list.append(round(train_loss,3))
            train_smL1_loss_list.append(round(train_smL1_loss,3))
            train_focal_loss_list.append(round(train_focal_loss,3))
            train_noise_loss_list.append(round(train_noise_loss,3))
            loss_x_list.append(int(epoch))


        if (epoch+1) % 50==0 and epoch > 0:
            temp_image_auroc,temp_pixel_auroc= eval(testing_dataset_loader,args,unet_model,seg_model,unet_model_t,seg_model_t,data_len,sub_class,device)
            image_auroc_list.append(temp_image_auroc)
            pixel_auroc_list.append(temp_pixel_auroc)
            performance_x_list.append(int(epoch))
            if(temp_image_auroc+temp_pixel_auroc>=best_image_auroc+best_pixel_auroc):
                if temp_image_auroc>=best_image_auroc:
                    save(unet_model,seg_model, args=args,final='best',epoch=epoch,sub_class=sub_class)
                    best_image_auroc = temp_image_auroc
                    best_pixel_auroc = temp_pixel_auroc
                    best_epoch = epoch
                
            
    save(unet_model,seg_model, args=args,final='last',epoch=args['EPOCHS'],sub_class=sub_class)



    temp = {"classname":[sub_class],"Image-AUROC": [best_image_auroc],"Pixel-AUROC":[best_pixel_auroc],"epoch":best_epoch}
    df_class = pd.DataFrame(temp)
    df_class.to_csv(f"{args['output_path']}/metrics/ARGS={args['arg_num']}/{args['eval_normal_t']}_{args['eval_noisier_t']}t_{args['condition_w']}_{class_type}_image_pixel_auroc_train.csv", mode='a',header=False,index=False)
   
    

def eval(testing_dataset_loader,args,unet_model,seg_model,unet_model_t,seg_model_t,data_len,sub_class,device):
    unet_model.eval()
    seg_model.eval()
    os.makedirs(f'{args["output_path"]}/metrics/ARGS={args["arg_num"]}/{sub_class}/', exist_ok=True)
    in_channels = args["channels"]
    betas = get_beta_schedule(args['T'], args['beta_schedule'])

    ddpm_sample =  GaussianDiffusionModel(
            args['img_size'], betas, loss_weight=args['loss_weight'],
            loss_type=args['loss-type'], noise=args["noise_fn"], img_channels=in_channels
            )
    
    print("data_len",data_len)
    # 计算每个图像像素级异常分数
    total_image_pred = np.array([])
    # 记录图像真实标签
    total_image_gt =np.array([])
    # 记录图像像素级异常掩膜
    total_pixel_gt=np.array([])
    # 记录像素级预测异常掩膜
    total_pixel_pred = np.array([])
    tbar = tqdm(testing_dataset_loader)
    for i, sample in enumerate(tbar):
        image = sample["image"].to(device)
        depth = sample["depth"].to(device)
        target = sample['is_anomaly'].to(device)
        gt_mask = sample["img_gt"].to(device)
        if args["get_features"]:
            features_t = sample["features_t"].to(device)
            depth_t = sample["depth_t"].to(device)
            features_t = torch.cat((features_t, depth_t), dim=1)


        image = torch.cat((image, depth), dim=1)


        t_tensor = torch.tensor([args["eval_t"]], device=image.device).repeat(image.shape[0])
        # 就只有pred_x_0_condition用到了
        loss, pred_x_0, x_t = ddpm_sample.one_step_denoising_eval(unet_model, image,t_tensor, args)
        pred_mask = seg_model(torch.cat((image, pred_x_0), dim=1))
        out_mask = pred_mask
        if args["get_features"]:
            loss_f, pred_f_0, f_t = ddpm_sample.one_step_denoising_eval(unet_model_t, features_t, t_tensor, args)
            pred_fmask_ano = seg_model_t(torch.cat((features_t, pred_f_0), dim=1))
            pred_fmask_ano = F.interpolate(pred_fmask_ano, size=args["img_size"], mode='bilinear', align_corners=False)
            out_mask = (pred_fmask_ano + out_mask) / 2




        topk_out_mask = torch.flatten(out_mask[0], start_dim=1)
        topk_out_mask = torch.topk(topk_out_mask, 30, dim=1, largest=True)[0]
        # 取前30个异常分数最大像素做平均作为像素级别异常分数
        image_score = torch.mean(topk_out_mask)

        total_image_pred = np.append(total_image_pred, image_score.detach().cpu().numpy())
        total_image_gt = np.append(total_image_gt, target[0].detach().cpu().numpy())

        flatten_pred_mask = out_mask[0].flatten().detach().cpu().numpy()
        flatten_gt_mask = gt_mask[0].flatten().detach().cpu().numpy().astype(int)

        total_pixel_gt = np.append(total_pixel_gt, flatten_gt_mask)
        total_pixel_pred = np.append(total_pixel_pred, flatten_pred_mask)
        
        
    print(sub_class)
    # round函数用来确定保留几位小数
    auroc_image = round(roc_auc_score(total_image_gt,total_image_pred),3)*100
    print("Image AUC-ROC: " ,auroc_image)
    
    auroc_pixel =  round(roc_auc_score(total_pixel_gt, total_pixel_pred),3)*100
    print("Pixel AUC-ROC:" ,auroc_pixel)
   
    return auroc_image,auroc_pixel


def save(unet_model,seg_model, args,final,epoch,sub_class):
    
    if final=='last':
        torch.save(
            {
                'n_epoch':              epoch,
                'unet_model_state_dict': unet_model.state_dict(),
                'seg_model_state_dict':  seg_model.state_dict(),
                "args":                 args
                }, f'{args["output_path"]}/model/diff-params-ARGS={args["arg_num"]}/{sub_class}/params-{final}.pt'
            )
    
    else:
        torch.save(
                {
                    'n_epoch':              epoch,
                    'unet_model_state_dict': unet_model.state_dict(),
                    'seg_model_state_dict':  seg_model.state_dict(),
                    "args":                 args
                    }, f'{args["output_path"]}/model/diff-params-ARGS={args["arg_num"]}/{sub_class}/params-{final}.pt'
                )
    
    

def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,2,1,3,4,5,6,7"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # read file from argument
    file = "args1.json"
    # load the json args
    with open(f'./args/{file}', 'r') as f:
        args = json.load(f)
    args['arg_num'] = file[4:-5]
    args = defaultdict_from_json(args)


    # 评测了很多不同的数据集
    mvtec_classes = ['carpet', 'grid', 'leather', 'tile', 'wood', 'bottle', 'cable', 'capsule', 'hazelnut', 'metal_nut', 'pill', 'screw',
            'toothbrush', 'transistor', 'zipper']

    mvtec_3d_classes = ['dowel', 'potato']


    # 默认用MVTec-AD数据集进行测评
    current_classes = mvtec_3d_classes

    class_type = ''
    for sub_class in current_classes:    
        print("class",sub_class)
        if sub_class in mvtec_3d_classes:
            training_dataset = MVTec3DTrainDataset(
                classname=sub_class,args=args
                )
            testing_dataset = MVTec3DTestDataset(
                classname=sub_class,args=args
                )
            class_type='VisA'
        elif sub_class in mvtec_classes:
            # 加载MVTec-2D数据
            subclass_path = os.path.join(args["mvtec_root_path"],sub_class)
            # img_size用来统一训练数据的尺寸，怕会有不同尺寸的图片
            training_dataset = MVTecTrainDataset(
                subclass_path,sub_class,img_size=args["img_size"],args=args
                )
            testing_dataset = MVTecTestDataset(
                subclass_path,sub_class,img_size=args["img_size"],
                )
            class_type='MVTec'

        

        print(file, args)     

        data_len = len(testing_dataset)
        print(data_len)
        # 基于数据集创建数据加载器，需要打乱数据、设置batch
        # pin_memory=True: 这会将加载的数据保存在锁定的内存中,可以加快 GPU 数据传输的速度。
        # drop_last=True: 如果最后一个 batch 的样本数量小于指定的 batch_size,这将丢弃该 batch。这是为了保证每个 batch 的大小一致。
        # shuffle=True会报错，详细查看：https://blog.csdn.net/qq_38247544/article/details/106651404
        training_dataset_loader = DataLoader(training_dataset, batch_size=args['Batch_Size'],shuffle=False,num_workers=8,pin_memory=True,drop_last=True)
        test_loader = DataLoader(testing_dataset, batch_size=1,shuffle=False, num_workers=4)

        # make arg specific directories
        # 创建中间结果输出文件夹
        for i in [f'{args["output_path"]}/model/diff-params-ARGS={args["arg_num"]}/{sub_class}',
                f'{args["output_path"]}/diffusion-training-images/ARGS={args["arg_num"]}/{sub_class}',
                 f'{args["output_path"]}/metrics/ARGS={args["arg_num"]}/{sub_class}']:
            try:
                os.makedirs(i)
            except OSError:
                pass

        # train之前把数据集相关设置好
        train(training_dataset_loader, test_loader, args, data_len,sub_class,class_type,device )

if __name__ == '__main__':
    
    seed(42)
    main()
