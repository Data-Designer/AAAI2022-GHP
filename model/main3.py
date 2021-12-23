# 这个model会在每个节点上训练一次，然后循环20次

import itertools
import os
import torch.nn.functional as F
from random import shuffle
from AuxTarget import *
from AutomaticWeightedLoss import *
from data.Data_copy import *

from pytorch_lightning.metrics import F1,Accuracy,Recall,Precision
from pytorch_lightning.metrics.functional import auc
print(torch.__version__)
print('device',device)





# 对网络参数先初始化一下
# 其实如果能够使用Blog数据判断社区的下一个时期一些指标的情况就好了
def get_net(wide_dim,dense_dim,embed_dim,input_fec_embed, hidden_size, out_embedding):
    net = AuxTarget8(wide_dim,dense_dim,embed_dim,input_fec_embed, hidden_size, out_embedding) # 修改变种
    for param in net.parameters():
        nn.init.normal_(param, mean=0, std=0.01)
    return net



# 这个给train 用
def lossbatch(model, x_catgs, x_conts, Gs, inputss, data_id_alls, vocab_to_ints, step, label_stab, label_repu, label_act,
              label_commu, label_reso, label_ability, awl,data_guild):
    '''
    func: 接收model和batch数据，返回total loss
    '''
    start = step * BATCHSIZE
    end = start + BATCHSIZE
    x_catgs_bs = [x_catg[start:end].to(device) for x_catg in x_catgs]
    x_conts_bs = [x_cont[start:end].to(device) for x_cont in x_conts]
    stab, repu, act, commu, reso, ability = model(x_catgs_bs, x_conts_bs, Gs, inputss, data_id_alls, vocab_to_ints,
                                            step,data_guild)  # 使用当前的step更新guild id
    loss1 = criterion_stab(stab,label_stab[start:end]) # True,1,cri,sql
    loss2 = criterion_repu(repu,label_repu[start:end]) #  need accuray,F1-score
    loss3 = criterion_act(label_act[start:end], act)
    loss4 = criterion_commu(label_commu[start:end], commu)
    loss5 = criterion_reso(reso,label_reso[start:end])
    loss6 = criterion_ability(label_ability[start:end], ability)
    losses = torch.stack((loss6,)) # perception在这里改
    # losses = torch.stack((loss2, loss3, loss4, loss5, loss6))  # perception在这里改
    total_loss = awl(losses)
    return total_loss, loss1, loss2, loss3, loss4, loss5, loss6

# 这个给test用，免得修改代码
def lossbatchtest(model, x_catgs, x_conts, Gs, inputss, data_id_alls, vocab_to_ints, step, label_stab, label_repu, label_act,
              label_commu, label_reso, label_ability, awl,data_guild):
    '''
    func: 接收model和batch数据，返回total loss
    '''
    start = step * BATCHSIZE
    end = start + BATCHSIZE
    x_catgs_bs = [x_catg[start:end].to(device) for x_catg in x_catgs]
    x_conts_bs = [x_cont[start:end].to(device) for x_cont in x_conts]
    stab, repu, act, commu, reso, ability = model(x_catgs_bs, x_conts_bs, Gs, inputss, data_id_alls, vocab_to_ints,
                                            step,data_guild)  # 使用当前的step更新guild id
    loss1 = criterion_stab(stab,label_stab[start:end]) # True,1,cri,sql
    loss2 = criterion_repu(repu,label_repu[start:end]) #  need accuray,F1-score
    loss3 = criterion_act(label_act[start:end], act)
    loss4 = criterion_commu(label_commu[start:end], commu)
    loss5 = criterion_reso(reso,label_reso[start:end])
    loss6 = criterion_ability(label_ability[start:end], ability)
    losses = torch.stack((loss6,)) # 这里进行单一目标和多目标的修改
    total_loss = awl(losses)

    # metric
    # predict_stab = stab.max(1)[1].detach().cpu().numpy().tolist()
    # label_stab = label_stab[start:end].detach().cpu().numpy().tolist()
    # predict_repu = repu.max(1)[1].detach().cpu().numpy().tolist()
    # print('repu',predict_repu)
    # label_repu = label_repu[start:end].detach().cpu().numpy().tolist() # classifier
    # print('label_repu', label_repu)
    loss1_mae = criterion_stab_mae(stab,label_stab[start:end])
    loss2_mae = criterion_repu_mae(repu, label_repu[start:end])
    loss3_mae = criterion_act_mae(label_act[start:end], act) # regression
    loss4_mae = criterion_commu_mae(label_commu[start:end], commu)
    loss5_mae = criterion_reso_mae(reso,label_reso[start:end])
    loss6_mae = criterion_ability_mae(label_ability[start:end], ability)
    return total_loss, loss1, loss2, loss3, loss4, loss5, loss6,\
           loss1_mae,loss2_mae,\
           loss3_mae,loss4_mae,loss5_mae,loss6_mae



# 训练
def train(epoch, model, optimizer, x_catgs, x_conts, Gs, inputss, data_id_alls, vocab_to_ints, label_stab, label_repu,
          label_act, label_commu, label_reso, label_ability, awl,data_guild,epoch_flag):
    model.train()
    train_loss = 0
    loss1_train, loss2_train, loss3_train, loss4_train, loss5_train, loss6_train = 0,0,0,0,0,0
    print('Epoch {} '.format(epoch))
    # batch
    for step in range((data_guild['guild_id'].shape[0] - 1) // BATCHSIZE + 1):
        total_loss, loss1, loss2, loss3, loss4, loss5, loss6 = lossbatch(model, x_catgs, x_conts, Gs, inputss, data_id_alls, vocab_to_ints, step, label_stab,
                               label_repu, label_act, label_commu, label_reso, label_ability, awl,data_guild)
        optimizer.zero_grad()
        total_loss.backward(retain_graph=True)  # 所以只需要改total loss就好了。
        train_loss = train_loss + total_loss.item()  # 累计每个step的loss值
        loss1_train = loss1_train + loss1.item()
        loss2_train = loss2_train + loss2.item()
        loss3_train = loss3_train + loss3.item()
        loss4_train = loss4_train + loss4.item()
        loss5_train = loss5_train + loss5.item()
        loss6_train = loss6_train + loss6.item()
        if step % 10 == 0:  # 每 10次step打印一次 total loss
            print('STEP {} , {}'.format(step, total_loss))
        optimizer.step()
    train_losses.append(train_loss)
    loss1_traines.append(loss1_train)
    loss2_traines.append(loss2_train)
    loss3_traines.append(loss3_train)
    loss4_traines.append(loss4_train)
    loss5_traines.append(loss5_train)
    loss6_traines.append(loss6_train)
    train_epoch_size.append(data_guild['guild_id'].shape[0])


# 验证测试
def test(model, x_catgs_valids, x_conts_valids, Gs_valids, inputss_valids, data_id_alls_valids,
         vocab_to_ints_valids,label_stab_valids, label_repu_valids, label_act_valids, label_commu_valids, label_reso_valids, label_ability_valids,awl,data_guild_valids,epoch,epoch_flag):
    model.eval()  # test验证集合应该是一组独立的数据,但是二者分布不一致咋办
    eval_loss = 0
    loss1_eval, loss2_eval, loss3_eval, loss4_eval, loss5_eval, loss6_eval = 0,0,0,0,0,0
    # regression metrics
    loss1_mae,loss2_mae, loss3_mae,loss4_mae,loss5_mae,loss6_mae = 0,0,0,0,0,0

    with torch.no_grad():
        for step in range((data_guild_valids['guild_id'].shape[0] - 1) // BATCHSIZE + 1): # 为哈在这里没帮其切分呢。
            eval_loss, loss1_eval, loss2_eval, loss3_eval, loss4_eval, loss5_eval, loss6_eval,\
            loss1_mae,loss2_mae,\
           loss3_mae,loss4_mae,loss5_mae,loss6_mae = lossbatchtest(model, x_catgs_valids, x_conts_valids, Gs_valids, inputss_valids, data_id_alls_valids,
                                  vocab_to_ints_valids, step,
                                  label_stab_valids, label_repu_valids, label_act_valids, label_commu_valids,
                                  label_reso_valids, label_ability_valids,awl,data_guild_valids)
            eval_loss = eval_loss + eval_loss.item()
            loss1_eval = loss1_eval + loss1_eval.item()
            loss2_eval = loss2_eval + loss2_eval.item()
            loss3_eval = loss3_eval + loss3_eval.item()
            loss4_eval = loss4_eval + loss4_eval.item()
            loss5_eval = loss5_eval + loss5_eval.item()
            loss6_eval = loss6_eval + loss6_eval.item()
            # metric
            # predict_stabs.extend(predict_stab) # 必须要转成列表就很迷
            # label_stabs.extend(label_stab)
            loss1_mae = loss1_mae + loss1_mae.item()
            loss2_mae = loss2_mae + loss2_mae.item()
            # predict_repus.extend(predict_repu)
            # label_repus.extend(label_repu)
            loss3_mae = loss3_mae + loss3_mae.item()
            loss4_mae = loss4_mae + loss4_mae.item()
            loss5_mae = loss5_mae + loss5_mae.item()
            loss6_mae = loss6_mae + loss6_mae.item()
            if step %10==0:
                print('STEP {} , {}'.format(step, eval_loss))
        print('Eval_losses: ', eval_loss)
        eval_losses.append(eval_loss) # 这里其实记录的就是一部分帮会的值
        loss1_evals.append(loss1_eval)
        loss2_evals.append(loss2_eval)
        loss3_evals.append(loss3_eval)
        loss4_evals.append(loss4_eval)
        loss5_evals.append(loss5_eval)
        loss6_evals.append(loss6_eval)
        eval_epoch_size.append(data_guild_valids['guild_id'].shape[0])

        # metrics
        loss1_evals_maes.append(loss1_mae)
        loss2_evals_maes.append(loss2_mae)
        loss3_evals_maes.append(loss3_mae)
        loss4_evals_maes.append(loss4_mae)
        loss5_evals_maes.append(loss5_mae)
        loss6_evals_maes.append(loss6_mae)



def main():
    start_epoch = 0
    for epoch in range(start_epoch + 1, EPOCHS):
        train(epoch, model, optimizer, x_catgs, x_conts, Gs, inputss, data_id_alls, vocab_to_ints, label_stab,
              label_repu, label_act, label_commu, label_reso, label_ability, awl,data_guild,epoch_flag)
        print('testing.............')
        test(model, x_catgs_valids, x_conts_valids, Gs_valids, inputss_valids, data_id_alls_valids,vocab_to_ints_valids,
             label_stab_valids, label_repu_valids, label_act_valids, label_commu_valids, label_reso_valids, label_ability_valids,awl,data_guild_valids,
             epoch,epoch_flag)
        state = {'model': model.state_dict(),
                 'optimizer': optimizer.state_dict(),
                 'epoch': epoch
                 }
        torch.save(state, log_dir)  # 一个epoch存储一次

    print('Finish Training')
    writer.close()



if __name__ == '__main__':
    # 设置随机数种子，保证结果可以复现
    setup_seed(20)
    # model = get_net(wide_dim, dense_dim, embed_dim, input_fec_embed, hidden_size, out_embedding).to(
    #     device) # 这里的wide_dim和dense_dim给它换成常数，原来是为了可扩展性
    # awl = AutomaticWeightedLoss(torch.Tensor([True, True, True, True, True, True]))  # 必须存为Tensor
    awl = AutomaticWeightedLoss(torch.Tensor([True,]))
    model = get_net(36, 32, 8, input_fec_embed, hidden_size, out_embedding).to(
        device) # 这里的wide_dim和dense_dim,embedding dim给它换成常数，原来是为了可扩展性
    # 定义一些loss值和优化器
    # criterion_stab = torch.nn.MSELoss()
    criterion_stab = torch.nn.MSELoss()
    criterion_repu =  torch.nn.MSELoss()
    criterion_act = torch.nn.MSELoss()
    criterion_commu = torch.nn.MSELoss()
    criterion_reso = torch.nn.MSELoss()
    criterion_ability = torch.nn.MSELoss()
    torch.autograd.set_detect_anomaly(True)  # 可以查看error回调栈


    # 指标计算所用
    criterion_stab_mae= torch.nn.L1Loss()
    criterion_repu_mae = torch.nn.L1Loss()
    criterion_act_mae = torch.nn.L1Loss()
    criterion_commu_mae = torch.nn.L1Loss()
    criterion_reso_mae =  torch.nn.L1Loss()
    criterion_ability_mae = torch.nn.L1Loss()

    # 读取数据
    # ds_end = '2020-10-17'  # 这个输入的一般都是截止日期,往前推14天
    optimizer = torch.optim.Adam(itertools.chain(model.parameters(), awl.parameters()), lr=1e-5,
                                 weight_decay=1e-5)  # 放入学习的是embeding而不是玩家的feature
    ds_ends = get_date_lista('2020-11-01','2021-01-17') + get_date_lista('2021-01-31','2021-02-08')
    shuffle(ds_ends) # 不打乱会出现20次为周期的下降
    ds_ends.remove('2020-12-20') # 本来在error里的，减少IO
    ds_ends.remove('2020-12-06') # 本来在error里的，减少IO
    ds_ends.remove('2020-12-13')  # 本来在error里的，减少IO
    ds_ends.remove('2020-12-27') # 本来在error里的，减少IO
    ds_ends.remove('2020-11-22') # 本来在error里的，减少IO
    ds_ends.remove('2020-11-01') # 本来在error里的，减少IO
    ds_ends_test = get_date_lista('2021-01-24','2021-01-25') # 用一个周期的数据进行test
    # 就选一组数据进行测试
    data_guild_valids, x_catgs_valids, x_conts_valids, wide_dim_valids, dense_dim_valids, embed_dim_valids, \
    Gs_valids, inputss_valids, data_id_alls_valids, vocab_to_ints_valids, \
    label_stab_valids, label_repu_valids, label_act_valids, label_commu_valids, label_reso_valids, label_ability_valids,embed_alls_valids = getdata(ds_ends_test[0])
    epoch_flag = 0

    # 减少IO操作,在第一个epoch中存储全量数据
    store_data = {}
    for i in range(12): # 这里才是真正的8个epoch
        EPOCHS = 2
        # metric
        # acc = Accuracy().to(device)
        # recall = Recall(num_classes=2).cuda()
        # precision = Precision().cuda()
        # f1 = F1(num_classes=2).to(device)
        # auc直接对矩阵进行计算
        train_losses = [] # 这里存的是分批数据的训练结果
        loss1_traines = []
        loss2_traines = []
        loss3_traines = []
        loss4_traines = []
        loss5_traines = []
        loss6_traines = []
        train_epoch_size = [] # 最后sum
        eval_losses = []
        loss1_evals = []
        loss2_evals = []
        loss3_evals = []
        loss4_evals = []
        loss5_evals = []
        loss6_evals = []
        eval_epoch_size = []
        # metric calculation
        # predict_stabs = []
        # label_stabs = []
        loss1_evals_maes = []
        # predict_repus = []
        # label_repus = [] # 为啥这里定义不行啊,显示referenced before assignment
        loss2_evals_maes = []
        loss3_evals_maes = []
        loss4_evals_maes = []
        loss5_evals_maes = []
        loss6_evals_maes = []

        for ds_end in ds_ends:
            # try:
            if i ==0:
                store_data[ds_end] = getdata(ds_end)
                data_guild, x_catgs, x_conts, wide_dim, dense_dim, embed_dim, \
                Gs, inputss, data_id_alls, vocab_to_ints, \
                label_stab, label_repu, label_act, label_commu, label_reso, label_ability,embed_alls = store_data[ds_end] # 14天的都全了

            else:
                data_guild, x_catgs, x_conts, wide_dim, dense_dim, embed_dim, \
                Gs, inputss, data_id_alls, vocab_to_ints, \
                label_stab, label_repu, label_act, label_commu, label_reso, label_ability,embed_alls = store_data[ds_end]
                # 图已经包含在里面了，可以model.parameters查看一下

            main()
            epoch_flag  = epoch_flag+1 # 用于标记数据段
            # except ValueError:
            #     print('该14天内有train数据缺失，跳过该数据段')
            #     continue
            # except RuntimeError:
            #     print('该时间段内有帮会信息丢失，创建超过14天的帮会数量大于某天查询的帮会数量') # label数量》guild_info数量
        print('************************************************************************',i)
        writer.add_scalar('training loss', sum(train_losses) / sum(train_epoch_size), i)
        writer.add_scalar('loss1_train', sum(loss1_traines) / sum(train_epoch_size), i)
        print('loss1_train', sum(loss1_traines) / sum(train_epoch_size), i)
        writer.add_scalar('loss2_train', sum(loss2_traines) / sum(train_epoch_size), i)
        writer.add_scalar('loss3_train', sum(loss3_traines) / sum(train_epoch_size), i)
        writer.add_scalar('loss4_train', sum(loss4_traines) / sum(train_epoch_size), i)
        writer.add_scalar('loss5_train', sum(loss5_traines) / sum(train_epoch_size), i)
        writer.add_scalar('loss6_train', sum(loss6_traines) / sum(train_epoch_size), i)
        writer.add_scalar('eval loss', sum(eval_losses) / sum(eval_epoch_size), i)
        writer.add_scalar('loss1_eval', sum(loss1_evals) / sum(eval_epoch_size), i)
        print('loss1_eval', sum(loss1_evals) / sum(eval_epoch_size), i)
        writer.add_scalar('loss2_eval', sum(loss2_evals) / sum(eval_epoch_size), i)
        writer.add_scalar('loss3_eval', sum(loss3_evals) / sum(eval_epoch_size), i)
        writer.add_scalar('loss4_eval', sum(loss4_evals) / sum(eval_epoch_size), i)
        writer.add_scalar('loss5_eval', sum(loss5_evals) / sum(eval_epoch_size), i)
        writer.add_scalar('loss6_eval', sum(loss6_evals) / sum(eval_epoch_size), i)
        print('下面开始计算metric')
        # merics
        writer.add_scalar('loss1_eval_mae', sum(loss1_evals_maes) / sum(eval_epoch_size), i)
        print('loss1_eval_mae', sum(loss1_evals_maes) / sum(eval_epoch_size),i)
        writer.add_scalar('loss2_eval_mae', sum(loss2_evals_maes) / sum(eval_epoch_size), i)
        writer.add_scalar('loss3_eval_mae', sum(loss3_evals_maes)/ sum(eval_epoch_size),i)
        writer.add_scalar('loss4_eval_mae', sum(loss4_evals_maes) / sum(eval_epoch_size), i)
        writer.add_scalar('loss5_eval_mae', sum(loss5_evals_maes) / sum(eval_epoch_size), i)
        writer.add_scalar('loss6_eval_mae', sum(loss6_evals_maes) / sum(eval_epoch_size), i)
        # writer.add_scalar('Stab ACC',acc(torch.Tensor(predict_stabs).long().to(device),torch.Tensor(label_stabs).long().to(device)),i)
        # writer.add_scalar('Repus ACC', acc(torch.Tensor(predict_repus).long().to(device), torch.Tensor(label_repus).long().to(device)),i)
        # writer.add_scalar('Stab F1', f1(torch.Tensor(predict_stabs).long().to(device), torch.Tensor(label_stabs).long().to(device)),i)
        # writer.add_scalar('Repu F1', f1(torch.Tensor(predict_repus).long().to(device), torch.Tensor(label_repus).long().to(device)),i)
        # writer.add_scalar('Stab AUC', auc(torch.Tensor(predict_stabs).long().to(device),torch.Tensor(label_stabs).long().to(device)),i)
        # writer.add_scalar('Repu AUC', auc(torch.Tensor(predict_repus).long().to(device), torch.Tensor(label_repus).long().to(device)), i)

    writer.close()
