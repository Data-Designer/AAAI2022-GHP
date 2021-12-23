import itertools
import os
import torch.nn.functional as F
import warnings
from random import shuffle
from AuxTarget import *
from AutomaticWeightedLoss import *
from data.Data_copy import *
print(torch.__version__)
print('device',device)
warnings.filterwarnings('ignore')




# 对网络参数先初始化一下
def get_net(wide_dim,dense_dim,embed_dim,input_fec_embed, hidden_size, out_embedding):
    net = AuxTarget2(wide_dim,dense_dim,embed_dim,input_fec_embed, hidden_size, out_embedding) # 修改变种
    for param in net.parameters():
        nn.init.normal_(param, mean=0, std=0.01)
    return net




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
    # print(stab)
    # print(stab.shape)
    # print(stab.dtype)
    # print(label_stab[start:end])
    # print(label_stab[start:end].shape)
    # print(label_stab[start:end].dtype)
    loss1 = criterion_stab(stab,label_stab[start:end]) # True,aux,cri,sql,
    loss2 = criterion_repu(repu,label_repu[start:end]) # 不收敛
    loss3 = criterion_act(label_act[start:end], act)
    loss4 = criterion_commu(label_commu[start:end], commu)
    loss5 = criterion_reso(reso,label_reso[start:end])
    loss6 = criterion_ability(label_ability[start:end], ability)
    losses = torch.stack((loss1, loss2, loss3, loss4, loss5, loss6))
    total_loss = awl(losses)
    return total_loss, loss1, loss2, loss3, loss4, loss5, loss6 # 已修改


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
        total_loss.backward(retain_graph=True)
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
    train_losses.append(train_loss / data_guild['guild_id'].shape[0])  # 一个epoch存储一次平均total loss值 ，这个为train loss
    writer.add_scalar('training loss', train_loss/ data_guild['guild_id'].shape[0], epoch + epoch_flag * EPOCHS) # 这里改成Epoch
    writer.add_scalar('loss1_train', loss1_train / data_guild['guild_id'].shape[0], epoch + epoch_flag * EPOCHS)
    writer.add_scalar('loss2_train', loss2_train / data_guild['guild_id'].shape[0], epoch + epoch_flag * EPOCHS)
    writer.add_scalar('loss3_train', loss3_train / data_guild['guild_id'].shape[0], epoch + epoch_flag * EPOCHS)
    writer.add_scalar('loss4_train', loss4_train / data_guild['guild_id'].shape[0], epoch + epoch_flag * EPOCHS)
    writer.add_scalar('loss5_train', loss5_train / data_guild['guild_id'].shape[0], epoch + epoch_flag * EPOCHS)
    writer.add_scalar('loss6_train', loss6_train / data_guild['guild_id'].shape[0], epoch + epoch_flag * EPOCHS)

# 验证测试
def test(model, x_catgs_valids, x_conts_valids, Gs_valids, inputss_valids, data_id_alls_valids,
         vocab_to_ints_valids,label_stab_valids, label_repu_valids, label_act_valids, label_commu_valids, label_reso_valids, label_ability_valids,awl,data_guild_valids,epoch,epoch_flag):
    model.eval()  # test验证集合应该是一组独立的数据,但是二者分布不一致咋办
    eval_loss = 0
    loss1_eval, loss2_eval, loss3_eval, loss4_eval, loss5_eval, loss6_eval = 0,0,0,0,0,0
    with torch.no_grad():
        for step in range((data_guild_valids['guild_id'].shape[0] - 1) // BATCHSIZE + 1): # 为哈在这里没帮其切分呢。
            eval_loss, loss1_eval, loss2_eval, loss3_eval, loss4_eval, loss5_eval, loss6_eval = lossbatch(model, x_catgs_valids, x_conts_valids, Gs_valids, inputss_valids, data_id_alls_valids,
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
            if step %10==0:
                print('STEP {} , {}'.format(step, eval_loss))
        print('Eval_losses: ', eval_loss)
        eval_losses.append(eval_loss / data_guild_valids['guild_id'].shape[0])
        writer.add_scalar('eval loss', eval_loss/ data_guild_valids['guild_id'].shape[0], epoch + epoch_flag * EPOCHS)
        writer.add_scalar('loss1_eval', loss1_eval / data_guild_valids['guild_id'].shape[0], epoch + epoch_flag * EPOCHS)
        writer.add_scalar('loss2_eval', loss2_eval / data_guild_valids['guild_id'].shape[0], epoch + epoch_flag * EPOCHS)
        writer.add_scalar('loss3_eval', loss3_eval / data_guild_valids['guild_id'].shape[0], epoch + epoch_flag * EPOCHS)
        writer.add_scalar('loss4_eval', loss4_eval / data_guild_valids['guild_id'].shape[0], epoch + epoch_flag * EPOCHS)
        writer.add_scalar('loss5_eval', loss5_eval / data_guild_valids['guild_id'].shape[0], epoch + epoch_flag * EPOCHS)
        writer.add_scalar('loss6_eval', loss6_eval / data_guild_valids['guild_id'].shape[0], epoch + epoch_flag * EPOCHS)




def main():
    #     awl = AutomaticWeightedLoss(6) # 这里增加一下权重设计

    # if test_flag:
    #     # 这里是最终验证
    #     checkpoint = torch.load(log_dir)
    #     model.load_state_dict(checkpoint['model'])
    #     optimizer.load_state_dict(checkpoint['optimizer'])
    #     start_epoch = checkpoint['epoch']
    #     test(model, x_catgs_valids, x_conts_valids, Gs_valids, inputss_valids, data_id_alls_valids,
    #          vocab_to_ints_valids,label_stab_valids, label_repu_valids, label_act_valids, label_commu_valids, label_reso_valids,
    #          label_ability_valids,data_guild_valid,epoch)
    #     return

    # if os.path.exists(log_dir):
    #     checkpoint = torch.load(log_dir)
    #     model.load_state_dict(checkpoint['model'])
    #     optimizer.load_state_dict(checkpoint['optimizer'])
    #     start_epoch = checkpoint['epoch']
    #     print('加载epoch {} 成功'.format(start_epoch))
    # else:
    #
    #     print('无保存model，将从头开始训练！')
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
    awl = AutomaticWeightedLoss(torch.Tensor([False, False, True, True, True, True]))  # 必须存为Tensor

    model = get_net(36, 32, 8, input_fec_embed, hidden_size, out_embedding).to(
        device) # 这里的wide_dim和dense_dim,embedding dim给它换成常数，原来是为了可扩展性
    # 定义一些loss值和优化器
    # criterion_stab = torch.nn.MSELoss()
    criterion_stab = torch.nn.CrossEntropyLoss()
    criterion_repu = torch.nn.CrossEntropyLoss()
    criterion_act = torch.nn.MSELoss()
    criterion_commu = torch.nn.MSELoss()
    criterion_reso = torch.nn.MSELoss()
    criterion_ability = torch.nn.MSELoss()
    torch.autograd.set_detect_anomaly(True)  # 可以查看error回调栈
    # 图已经包含在里面了，可以model.parameters查看一下
    optimizer = torch.optim.Adam(itertools.chain(model.parameters(),awl.parameters()), lr=1e-4, weight_decay=1e-5)
    # 读取数据
    # ds_end = '2020-10-17'  # 这个输入的一般都是截止日期,往前推14天
    epoch_flag = 0
    ds_ends = get_date_lista('2020-11-01','2021-01-31')
    ds_ends.remove('2020-12-20') # 本来在error里的，减少IO
    ds_ends.remove('2020-12-06') # 这一段数据有缺失，会影响结果
    ds_ends.remove('2020-11-01') # 这一阶段也有数据缺失，即14天内有数据缺失
    shuffle(ds_ends) # 不打乱会出现20次为周期的下降
    # 这里可以直接ds_end*2
    ds_ends = ds_ends + ds_ends  # 补充数据进行训练，我不关心train_loss，我只关心eval loss收敛即可。
    ds_ends_test = get_date_lista('2021-02-07','2021-02-08') # 用一个周期的数据进行test
    # 就选一组数据进行测试
    data_guild_valids, x_catgs_valids, x_conts_valids, wide_dim_valids, dense_dim_valids, embed_dim_valids, \
    Gs_valids, inputss_valids, data_id_alls_valids, vocab_to_ints_valids, \
    label_stab_valids, label_repu_valids, label_act_valids, label_commu_valids, label_reso_valids, label_ability_valids = getdata(ds_ends_test[0])
    for ds_end in ds_ends:
        try:
            data_guild, x_catgs, x_conts, wide_dim, dense_dim, embed_dim, \
            Gs, inputss, data_id_alls, vocab_to_ints, \
            label_stab, label_repu, label_act, label_commu, label_reso, label_ability = getdata(ds_end)
            main()
            epoch_flag  = epoch_flag+1 # 用于标记数据段
        except ValueError:
            print('该14天内有train数据缺失，跳过该数据段')
            continue
        except RuntimeError:
            print('该时间段内有帮会信息丢失，创建超过14天的帮会数量大于某天查询的帮会数量') # label数量》guild_info数量

    writer.close()
