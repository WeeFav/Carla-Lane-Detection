import torch
import os
import numpy as np
import yaml
import time
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from model.model import UFLDNet
from data.dataloader import get_train_loader
from utils.factory import get_metric_dict, get_loss_dict, get_optimizer, get_scheduler
from utils.metrics import update_metrics, reset_metrics

def inference(net, data_label, use_aux):
    if use_aux:
        img, cls_label, seg_label = data_label
        img, cls_label, seg_label = img.cuda(), cls_label.long().cuda(), seg_label.long().cuda()
        cls_out, seg_out = net(img)
        return {'cls_out': cls_out, 'cls_label': cls_label, 'seg_out':seg_out, 'seg_label': seg_label}
    else:
        img, cls_label = data_label
        img, cls_label = img.cuda(), cls_label.long().cuda()
        cls_out = net(img)
        return {'cls_out': cls_out, 'cls_label': cls_label}

def resolve_val_data(results, use_aux):
    results['cls_out'] = torch.argmax(results['cls_out'], dim=1)
    if use_aux:
        results['seg_out'] = torch.argmax(results['seg_out'], dim=1)
    return results

def calc_loss(loss_dict, results, logger, global_step):
    loss = 0

    for i in range(len(loss_dict['name'])):

        data_src = loss_dict['data_src'][i]

        datas = [results[src] for src in data_src]

        loss_cur = loss_dict['op'][i](*datas)

        if global_step % 20 == 0:
            logger.add_scalar('loss/'+loss_dict['name'][i], loss_cur, global_step)

        loss += loss_cur * loss_dict['weight'][i]
    return loss

def train(net, data_loader, loss_dict, optimizer, scheduler, logger, epoch, metric_dict, use_aux):
    net.train()

    progress_bar = tqdm(train_loader)
    t_data_0 = time.time()

    for b_idx, data_label in enumerate(progress_bar):
        t_data_1 = time.time()
        reset_metrics(metric_dict)
        global_step = epoch * len(data_loader) + b_idx

        t_net_0 = time.time()
        results = inference(net, data_label, use_aux)

        loss = calc_loss(loss_dict, results, logger, global_step)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step(global_step)
        t_net_1 = time.time()

        results = resolve_val_data(results, use_aux)

        update_metrics(metric_dict, results)
        if global_step % 20 == 0:
            for me_name, me_op in zip(metric_dict['name'], metric_dict['op']):
                logger.add_scalar('metric/' + me_name, me_op.get(), global_step=global_step)
        logger.add_scalar('meta/lr', optimizer.param_groups[0]['lr'], global_step=global_step)

        kwargs = {me_name: '%.3f' % me_op.get() for me_name, me_op in zip(metric_dict['name'], metric_dict['op'])}
        progress_bar.set_postfix(loss = '%.3f' % float(loss), 
                                data_time = '%.3f' % float(t_data_1 - t_data_0), 
                                net_time = '%.3f' % float(t_net_1 - t_net_0), 
                                **kwargs)
        t_data_0 = time.time()

def save_model(net, optimizer, epoch, save_path):
    model_state_dict = net.state_dict()
    state = {'model': model_state_dict, 'optimizer': optimizer.state_dict()}
    assert os.path.exists(save_path)
    model_path = os.path.join(save_path, 'ep%03d.pth' % epoch)
    torch.save(state, model_path)
        
if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True

    with open("demo.yaml", 'r') as f:
        cfg = yaml.safe_load(f)
        
    train_loader, cls_num_per_lane = get_train_loader(cfg['batch_size'], cfg['data_root'], cfg['griding_num'], cfg['dataset'], cfg['use_aux'], cfg['num_lanes'])
    print(len(train_loader))

    net = UFLDNet(pretrained = True, backbone=cfg.backbone,cls_dim = (cfg.griding_num+1,cls_num_per_lane, cfg.num_lanes),use_aux=cfg.use_aux).cuda()

    optimizer = get_optimizer(net, cfg)

    if cfg['finetune'] is not None:
        print('finetune from ', cfg['finetune'])
        state_all = torch.load(cfg['finetune'])['model']
        state_clip = {}  # only use backbone parameters
        for k,v in state_all.items():
            if 'model' in k:
                state_clip[k] = v
        net.load_state_dict(state_clip, strict=False)

    if cfg['resume'] is not None:
        print('==> Resume model from ' + cfg['resume'])
        resume_dict = torch.load(cfg['resume'], map_location='cpu')
        net.load_state_dict(resume_dict['model'])
        if 'optimizer' in resume_dict.keys():
            optimizer.load_state_dict(resume_dict['optimizer'])
        resume_epoch = int(os.path.split(cfg['resume'])[1][2:5]) + 1
    else:
        resume_epoch = 0

    scheduler = get_scheduler(optimizer, cfg, len(train_loader))
    metric_dict = get_metric_dict(cfg)
    loss_dict = get_loss_dict(cfg)

    logger = SummaryWriter()

    for epoch in range(resume_epoch, cfg['epoch']):
        train(net, train_loader, loss_dict, optimizer, scheduler, logger, epoch, metric_dict, cfg['use_aux'])
        save_model(net, optimizer, epoch , cfg['save_path'])
    
    logger.close()