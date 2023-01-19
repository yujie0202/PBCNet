import argparse
import os
import random
import time
from collections import defaultdict

code_path = os.path.dirname(os.path.abspath(__file__))
import sys
sys.path.append(code_path + '/model_code')

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from dgl.dataloading import GraphDataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error

from Dataloader.dataloader import collate_fn, LeadOptDataset,LeadOptDataset_test
from ReadoutModel.readout_bind import DMPNN
# from ReadoutModel.readout_bind_deltadelta import DMPNN
from utilis.function import get_loss_func
from utilis.initial import initialize_weights
from utilis.scalar import StandardScaler
from utilis.scheduler import NoamLR_shan
from utilis.trick import Writer
from utilis.utilis import gm_process


def softmax(x):
    x_exp = np.exp(x)
    # 如果是列向量，则axis=0
    x_sum = np.sum(x_exp, axis=0)
    s = x_exp / x_sum
    return s


def setup_cpu(cpu_num):
    os.environ['OMP_NUM_THREADS'] = str(cpu_num)
    os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
    os.environ['MKL_NUM_THREADS'] = str(cpu_num)
    os.environ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
    os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
    torch.set_num_threads(cpu_num)


def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def predict(args, model, loader, device):
    model.eval()

    # if args.loss_function == 'mve':
    #     uncertainty = []
    # elif args.loss_function == "evidential":
    #     uncertainty = []

    valid_prediction = []
    valid_labels = []
    valid_1_labels = []
    ref_2_labels = []
    rank = []
    file = []

    att__1 = []
    att__2 = []

    for batch_data in loader:
        # 0 表示关闭
        if args.two_task == 0:
            if args.GCN_ == 0:
                graph1, graph2, _, label, label1, label2, rank1, file_name = batch_data
                # to cuda
                graph1, graph2, label, label1, label2= graph1.to(device), graph2.to(device),label.to(device), label1.to(device), label2.to(device)

                logits,att1,att2 = model(graph1,
                               graph2)
            elif args.GCN_ == 1:
                graph1, graph2, pock, label, label1, label2, rank1, file_name = batch_data
                # to cuda
                graph1, graph2, pock, label, label1, label2 = graph1.to(device), graph2.to(device), pock.to(device), label.to(device), label1.to(
                    device), label2.to(device)

                logits,att1,att2 = model(graph1,
                               graph2, pock)
        if args.two_task == 1:
            if args.GCN_ == 0:
                graph1, graph2, _, label, label1, label2, rank1, file_name = batch_data
                # to cuda
                graph1, graph2, label, label1, label2= graph1.to(device), graph2.to(device),label.to(device), label1.to(device), label2.to(device)

                logits,_,att1,att2 = model(graph1,
                               graph2)
            elif args.GCN_ == 1:
                graph1, graph2, pock, label, label1, label2, rank1, file_name = batch_data
                # to cuda
                graph1, graph2, pock, label, label1, label2 = graph1.to(device), graph2.to(device), pock.to(device), label.to(device), label1.to(
                    device), label2.to(device)

                logits,_, att1,att2 = model(graph1,
                               graph2, pock)




        if args.loss_function == 'mve':
            valid_prediction += logits[:, 0].unsqueeze(-1).tolist()
        elif args.loss_function == "evidential":
            valid_prediction += logits[:, 0].unsqueeze(-1).tolist()
        else:
            valid_prediction += logits.tolist()

        att__1 += att1.tolist()
        att__2 += att2.tolist()

        valid_labels += label.tolist()
        valid_1_labels += label1.tolist()
        ref_2_labels += label2.tolist()
        rank += rank1.tolist()
        file += file_name

    mae = mean_absolute_error(valid_labels, valid_prediction)
    rmse = mean_squared_error(valid_labels, valid_prediction) ** 0.5

    valid_labels_G = np.log(np.power(10, -np.array(valid_labels).astype(float)))*297*1.9872*1e-3
    valid_prediction_G = np.log(np.power(10, -np.array(valid_prediction).astype(float)))*297*1.9872*1e-3

    mae_g = mean_absolute_error(valid_labels_G, valid_prediction_G)
    rmse_g = mean_squared_error(valid_labels_G, valid_prediction_G) ** 0.5

    valid_prediction = np.array(valid_prediction).flatten()
    valid_prediction_G = np.array(valid_prediction_G).flatten()

    return mae, rmse, mae_g, rmse_g, valid_prediction, valid_prediction_G, np.array(valid_labels), att__1, att__2


@torch.no_grad()
def evaluate(args, model, loader, device, label_scalar):
    model.eval()

    # if args.loss_function == 'mve':
    #     uncertainty = []
    # elif args.loss_function == "evidential":
    #     uncertainty = []

    valid_prediction = []
    valid_labels = []
    valid_1_labels = []
    ref_2_labels = []
    rank = []
    file = []

    for batch_data in loader:
        if args.two_task == 0:
            if args.GCN_ == 0:
                graph1, graph2, _, label, label1, label2, rank1, file_name = batch_data
                # to cuda
                graph1, graph2, label, label1, label2= graph1.to(device), graph2.to(device),label.to(device), label1.to(device), label2.to(device)

                logits,att1,att2 = model(graph1,
                               graph2)
            elif args.GCN_ == 1:
                graph1, graph2, pock, label, label1, label2, rank1, file_name = batch_data
                # to cuda
                graph1, graph2, pock, label, label1, label2 = graph1.to(device), graph2.to(device), pock.to(device), label.to(device), label1.to(
                    device), label2.to(device)

                logits,att1,att2 = model(graph1,
                               graph2, pock)
        if args.two_task == 1:
            if args.GCN_ == 0:
                graph1, graph2, _, label, label1, label2, rank1, file_name = batch_data
                # to cuda
                graph1, graph2, label, label1, label2= graph1.to(device), graph2.to(device),label.to(device), label1.to(device), label2.to(device)

                logits,_,att1,att2 = model(graph1,
                               graph2)
            elif args.GCN_ == 1:
                graph1, graph2, pock, label, label1, label2, rank1, file_name = batch_data
                # to cuda
                graph1, graph2, pock, label, label1, label2 = graph1.to(device), graph2.to(device), pock.to(device), label.to(device), label1.to(
                    device), label2.to(device)

                logits,_, att1,att2 = model(graph1,
                               graph2, pock)

        if args.loss_function == 'mve':
            valid_prediction += logits[:, 0].unsqueeze(-1).tolist()
        elif args.loss_function == "evidential":
            valid_prediction += logits[:, 0].unsqueeze(-1).tolist()
        else:
            valid_prediction += logits.tolist()

        valid_labels += label.tolist()
        valid_1_labels += label1.tolist()
        ref_2_labels += label2.tolist()
        rank += rank1.tolist()
        file += file_name

    if label_scalar is None:
        pass
    else:
        valid_prediction = label_scalar.inverse_transform(valid_prediction)

    mae = mean_absolute_error(valid_labels, valid_prediction)
    rmse = mean_squared_error(valid_labels, valid_prediction) ** 0.5

    pre_abs_pic50 = np.array(valid_prediction).flatten() + np.array(ref_2_labels)

    file_to_p = defaultdict(list)
    for pre, lab, r, f in zip(pre_abs_pic50, valid_1_labels, rank, file):
        file_to_p[f].append([pre, lab, r])

    spearman = []
    pearson = []
    length = []
    files_ = []
    pre_abs_pic50_mean = []
    valid_1_labels_mean = []

    for f in file_to_p.keys():
        _df = pd.DataFrame(file_to_p[f], columns=['a', 'b', 'c'])
        _df = _df.groupby('c')[['a', 'b']].mean().reset_index()

        pre_abs_pic50_mean.extend(list(_df['a'].values))
        valid_1_labels_mean.extend(list(_df['b'].values))

        if len(_df) >= 5:
            files_.append(f)
            length.append(len(_df))

            spear = _df[["a", "b"]].corr(method='spearman')
            spearman.append(spear.iloc[0, 1])

            pear = _df[["a", "b"]].corr(method='pearson')
            pearson.append(pear.iloc[0, 1])

    abs_rmse = mean_squared_error(pre_abs_pic50_mean, valid_1_labels_mean) ** 0.5
    abs_mae = mean_absolute_error(pre_abs_pic50_mean, valid_1_labels_mean)

    spearman_ = np.nanmean(spearman)
    pearson_ = np.nanmean(pearson)

    corr_df = pd.DataFrame({'file_name': files_,
                            'spearman': spearman,
                            'pearson': pearson,
                            'num_of_val_data': length})

    return mae, rmse, abs_mae, abs_rmse, spearman_, pearson_, corr_df


def train(args, model, train_loader, val_loader, test_rmse_loader_FEP2, test_corr_loader_FEP2, test_file_names_FEP2,
          test_rmse_loader_FEP1, test_corr_loader_FEP1, test_file_names_FEP1,label_scalar, device, num_training_samples,file_names):
    global es, prediction_of_FEP, weights_loss, weight_file, pearson, file_name_all, corr_df, prediction_of_FEP1, prediction_of_FEP2, corr_of_FEP1
    train_start = time.time()
    # learning rate decay and optimizer
    opt = torch.optim.Adam(model.parameters(), lr=0.000001)
    epoch_step = len(train_loader)
    scheduler = NoamLR_shan(opt,
                            warmup_epochs=[1],
                            decay_epochs=[2],
                            final_epochs=[3],
                            steps_per_epoch=epoch_step,
                            init_lr=[args.init_lr],
                            max_lr=[args.max_lr],
                            final_lr=[args.final_lr])

    if args.loss_function == "mse_weight":
        file_name_all = pd.DataFrame({'file_name': file_names})
        corr_df = pd.DataFrame({'file_name': file_names,"pearson":[-1.0 for _ in range(len(file_names))]})


    # training loss
    loss_func = get_loss_func(args.loss_function)
    if args.two_task == 1:
        loss_func_class = get_loss_func("entropy")

    # for computing training mae and rmse
    mae_func = torch.nn.L1Loss(reduction='sum')
    mse_func = torch.nn.MSELoss(reduction='sum')

    # for log
    save_dir = code_path
    # save_dir = save_dir + f"_res_{args.readout_type}_{args.loss_function}_GCN{args.GCN_}_device{args.device}_hidden{args.hidden_dim}_cs{args.cs}_fold{args.fold}_seed{args.seed}"

    logger_writer = Writer(f"{save_dir}/record.txt")

    # for early stoping
    # stop_metric = 0
    # not_change_epoch = 0
    # Path = os.path.join(save_dir, "model.pth")

    # for log in a batch defined step
    batch_all = 0
    loss_for_log = 0
    
    model.train()
    for epoch in range(5):

        loss_mse = []
        loss_entropy = []
        start = time.time()

        if args.loss_function == "mse_weight":
            weight_file = pd.merge(file_name_all, corr_df, how="outer")
            if weight_file["pearson"].isnull().any() == True:
                weight_file= weight_file.fillna(pearson)
            weight_file["weight"] =  (1-(weight_file["pearson"]*0.5 +0.5))**3


        training_mae = 0
        training_mse = 0

        for batch_data in train_loader:
            if args.two_task == 0:
                if args.GCN_ == 0:
                    graph1, graph2, _, label, label1, label2, rank1, file_name = batch_data
                    # to cuda
                    graph1, graph2, label, label1, label2 = graph1.to(device), graph2.to(device), label.to(
                        device), label1.to(device), label2.to(device)

                    logits,_,_ = model(graph1,
                                   graph2)
                elif args.GCN_ == 1:
                    graph1, graph2, pock, label, label1, label2, rank1, file_name = batch_data
                    # to cuda
                    graph1, graph2, pock, label, label1, label2 = graph1.to(device), graph2.to(device), pock.to(
                        device), label.to(device), label1.to(
                        device), label2.to(device)

                    logits,_,_ = model(graph1,
                                   graph2, pock)
            else:
                if args.GCN_ == 0:
                    graph1, graph2, _, label, label1, label2, rank1, file_name = batch_data
                    # to cuda
                    graph1, graph2, label, label1, label2 = graph1.to(device), graph2.to(device), label.to(
                        device), label1.to(device), label2.to(device)

                    logits,logits_class,_,_ = model(graph1,
                                   graph2)
                elif args.GCN_ == 1:
                    graph1, graph2, pock, label, label1, label2, rank1, file_name = batch_data
                    # to cuda
                    graph1, graph2, pock, label, label1, label2 = graph1.to(device), graph2.to(device), pock.to(
                        device), label.to(device), label1.to(
                        device), label2.to(device)

                    logits,logits_class,_,_ = model(graph1,
                                   graph2, pock)


            if args.loss_function == "mse_weight":
                file_name_batch = pd.DataFrame({'file_name': file_name})
                weights_loss = torch.tensor(softmax(pd.merge(file_name_batch,weight_file).weight.values), device=device)


            if args.loss_function == 'mve':
                loss = loss_func(logits[:, 0].float(), label.float(), torch.exp(logits[:, 1]).float())
                train_mae_ = mae_func(logits[:, 0].float(), label.float())
                train_mse_ = mse_func(logits[:, 0].float(), label.float())
            elif args.loss_function == "evidential":
                loss = loss_func(logits[:, 0].float(),
                                 F.softplus(logits[:, 1].float()) + 1.0,
                                 F.softplus(logits[:, 2].float()) + 1.0,
                                 F.softplus(logits[:, 3].float()) + 0.1,
                                 label.float())
                train_mae_ = mae_func(logits[:, 0].float(), label.float())
                train_mse_ = mse_func(logits[:, 0].float(), label.float())
            
            elif args.loss_function == "mse_weight":
                loss = loss_func(logits.squeeze().float(), label.float(), weights_loss)
                train_mae_ = mae_func(logits.squeeze().float(), label.float())
                train_mse_ = mse_func(logits.squeeze().float(), label.float())

            else:
                loss = loss_func(logits.squeeze().float(), label.float())
                train_mae_ = mae_func(logits.squeeze().float(), label.float())
                train_mse_ = mse_func(logits.squeeze().float(), label.float())


            loss_mse.append(loss.item())

            if args.two_task == 1:

                loss_class = loss_func_class(logits_class.float(), label.float())

                loss_entropy.append(loss_class.item())

                loss = loss + loss_class

            opt.zero_grad()
            loss.backward()
            opt.step()
            scheduler.step()
            batch_all += 1
            loss_for_log += loss.item()
            training_mae += train_mae_.item()
            training_mse += train_mse_.item()

            if batch_all % args.log_frequency == 0:
                _loss = loss_for_log / args.log_frequency  # mean loss for each batch with size of log_frequency.
                print(f"Epoch {epoch}  Batch {batch_all}  Loss {_loss}")
                logger_writer(f"Epoch {epoch}  Batch {batch_all}  Loss {_loss}")
                print(f"Epoch {epoch}  Batch {batch_all}  Learning rate {scheduler.get_lr()[0]}")
                logger_writer(f"Epoch {epoch}  Batch {batch_all}  Learning rate {scheduler.get_lr()[0]}")
                loss_for_log = 0


            # 47500

            if batch_all % 47500 == 0:
                train_time = time.time()


                training_rmse = (training_mse / num_training_samples) ** 0.5
                training_mae = training_mae / num_training_samples

                logger_writer("  ")
                logger_writer(f"Epoch {epoch}_{batch_all}")
                logger_writer(f"Training time {train_time - start}")
                logger_writer(f"Training Set mae {training_mae}")
                logger_writer(f"Training Set rmse {training_rmse}")

                    #  =================== FEP+ =================
                # rmse
                rmse_gs = []

                file_name = []
                for num_loader in range(len(test_rmse_loader_FEP2)):
                    loader = test_rmse_loader_FEP2[num_loader]
                    file_nm = test_file_names_FEP2[num_loader]
                    file_name.append(file_nm)

                    # if num_loader == 0:
                    mae, rmse, mae_g, rmse_g, valid_prediction, valid_prediction_G, valid_labels,att__1,att__2 = predict(args, model, loader, device)



                    #     logger_writer(f"fep+ attention1 {att__1}")
                    #     logger_writer(f"fep+ attention2 {att__2}")


                    # else:
                    #     mae, rmse, mae_g, rmse_g, valid_prediction, valid_prediction_G, valid_labels,_,_ = predict(args, model, loader, device)


                    rmse_gs.append(rmse_g)

                    if num_loader == 0:
                        prediction_of_FEP2 = pd.DataFrame({f'prediction_ic50_{file_nm}': valid_prediction,
                                                           f'prediction_G_{file_nm}': valid_prediction_G,
                                                           f"label_ic50_{file_nm}": valid_labels})
                    else:
                        prediction_of_FEP2_ = pd.DataFrame({f'prediction_ic50_{file_nm}': valid_prediction,
                                                            f'prediction_G_{file_nm}': valid_prediction_G,
                                                            f"label_ic50_{file_nm}": valid_labels})
                        prediction_of_FEP2 = pd.merge(prediction_of_FEP2, prediction_of_FEP2_, how="outer",right_index=True,left_index=True)



                # corr
                spearmans = []
                pearsons = []
                kendalls = []

                spearmans_var = []
                pearsons_var = []
                kendalls_var = []

                spearmans_min = []
                pearsons_min = []
                kendalls_min = []

                spearmans_max = []
                pearsons_max = []
                kendalls_max = []

                for num_loader in range(len(test_corr_loader_FEP2)):
                    loader = test_corr_loader_FEP2[num_loader]
                    file_nm = test_file_names_FEP2[num_loader]
                    mae, rmse, mae_g, rmse_g, valid_prediction, valid_prediction_G, _ ,_,_= predict(args, model, loader, device)

                    df = pd.read_csv(f"{save_dir}/data/test_set_fep+_graph_rmH_I/input_files/1_reference/train_files/{file_nm}.csv")
                    df["predict_dmpnnmve_pic50"] = valid_prediction

                    abs_label = df.Lable1.values
                    abs_predict = np.array(df.predict_dmpnnmve_pic50.values).astype(float) + np.array(df.Lable2.values).astype(
                        float)

                    df["abs_label_p"] = abs_label
                    df["abs_predict_p"] = abs_predict

                    # =================以PIC50为单位====================
                    reference_num = df.reference_num.values
                    ligand1_num = df.Ligand1_num.values
                    _df = pd.DataFrame({"reference_num": reference_num, f"abs_label_{file_nm}": abs_label, f"abs_predict_{file_nm}": abs_predict, f"ligand1_num_{file_nm}": ligand1_num})

                    # ================ 用来画散点图的 ==============
                    if num_loader == 0:
                        corr_of_FEP2 = _df.groupby(f'ligand1_num_{file_nm}')[[f'abs_label_{file_nm}', f'abs_predict_{file_nm}']].mean().reset_index()
                    else:
                        corr_of_FEP2_ = _df.groupby(f'ligand1_num_{file_nm}')[[f'abs_label_{file_nm}', f'abs_predict_{file_nm}']].mean().reset_index()
                        corr_of_FEP2 = pd.merge(corr_of_FEP2, corr_of_FEP2_, how="outer",right_index=True,left_index=True)


                    _df_group = _df.groupby('reference_num')  # [['abs_label', 'abs_predict']].mean().reset_index()

                    spearman_ = []
                    pearson_ = []
                    kendall_ = []
                    for _, _df_onegroup in _df_group:
                        spearman = _df_onegroup[[f"abs_label_{file_nm}", f"abs_predict_{file_nm}"]].corr(method='spearman').iloc[0, 1]
                        pearson = _df_onegroup[[f"abs_label_{file_nm}", f"abs_predict_{file_nm}"]].corr(method='pearson').iloc[0, 1]
                        kendall = _df_onegroup[[f"abs_label_{file_nm}", f"abs_predict_{file_nm}"]].corr(method='kendall').iloc[0, 1]
                        spearman_.append(spearman)
                        pearson_.append(pearson)
                        kendall_.append(kendall)
                    spearmans.append(np.mean(spearman_))
                    pearsons.append(np.mean(pearson_))
                    kendalls.append(np.mean(kendall_))

                    spearmans_var.append(np.var(spearman_))
                    pearsons_var.append(np.var(pearson_))
                    kendalls_var.append(np.var(kendall_))

                    spearmans_min.append(np.min(spearman_))
                    pearsons_min.append(np.min(pearson_))
                    kendalls_min.append(np.min(kendall_))

                    spearmans_max.append(np.max(spearman_))
                    pearsons_max.append(np.max(pearson_))
                    kendalls_max.append(np.max(kendall_))

                for m_ in range(len(test_file_names_FEP2)):
                    file_nm = test_file_names_FEP2[m_]
                    rmse__ = rmse_gs[m_]
                    s_ = spearmans[m_]
                    p_ = pearsons[m_]
                    k_ = kendalls[m_]
                    s_var_ = spearmans_var[m_]
                    p_var_ = pearsons_var[m_]
                    k_var_ = kendalls_var[m_]

                    s_max_ = spearmans_max[m_]
                    p_max_ = pearsons_max[m_]
                    k_max_ = kendalls_max[m_]

                    s_min_ = spearmans_min[m_]
                    p_min_ = pearsons_min[m_]
                    k_min_ = kendalls_min[m_]

                    logger_writer(f"{file_nm},RMSE:{rmse__},spearman:{s_},spearman_var:{s_var_},spearmans_min:{s_min_},spearmans_max:{s_max_},\
                                    pearson:{p_}, pearsons_var:{p_var_},pearson_min:{p_min_},pearsons_max:{p_max_},kendall:{k_},kendall_var:{k_var_},\
                                    kendall_min:{k_min_},kendalls_max:{k_max_}")

                logger_writer(f"FEP,RMSE:{np.mean(rmse_gs)},spearman:{np.mean(spearmans)},spearman_var:{np.mean(spearmans_var)},spearmans_min:{np.mean(spearmans_min)},spearmans_max:{np.mean(spearmans_max)},\
                                    pearson:{np.mean(pearsons)}, pearsons_var:{np.mean(pearsons_var)},pearson_min:{np.mean(pearsons_min)},pearsons_max:{np.mean(pearsons_max)},kendall:{np.mean(kendalls)}, \
                                    kendall_var:{np.mean(kendalls_var)},kendall_min:{np.mean(kendalls_min)},kendalls_max:{np.mean(kendalls_max)}")
                fep1_spearmans = np.mean(spearmans)

                #  =================== FEP =================
                # rmse
                rmse_gs = []

                file_name = []
                for num_loader in range(len(test_rmse_loader_FEP1)):
                    loader = test_rmse_loader_FEP1[num_loader]
                    file_nm = test_file_names_FEP1[num_loader]
                    file_name.append(file_nm)
                    # if num_loader == 0:
                    mae, rmse, mae_g, rmse_g, valid_prediction, valid_prediction_G, valid_labels,att__1,att__2 = predict(args, model, loader,
                                                                                                           device)


                    # else:
                    #     mae, rmse, mae_g, rmse_g, valid_prediction, valid_prediction_G, valid_labels,_ ,_ = predict(args, model, loader,
                    #                                                                                        device)
                    rmse_gs.append(rmse_g)
                    if num_loader == 0:
                        prediction_of_FEP1 = pd.DataFrame({f'prediction_ic50_{file_nm}': valid_prediction,
                                                           f'prediction_G_{file_nm}': valid_prediction_G,
                                                           f"label_ic50_{file_nm}": valid_labels})
                    else:
                        prediction_of_FEP1_ = pd.DataFrame({f'prediction_ic50_{file_nm}': valid_prediction,
                                                            f'prediction_G_{file_nm}': valid_prediction_G,
                                                            f"label_ic50_{file_nm}": valid_labels})
                        prediction_of_FEP1 = pd.merge(prediction_of_FEP1, prediction_of_FEP1_, how="outer",right_index=True,left_index=True)

                csv_save_dir = os.path.join(save_dir, f"FEP_rmse_{epoch}_{batch_all}_{args.seed}.csv")
                prediction_of_FEP = pd.merge(prediction_of_FEP1, prediction_of_FEP2, how="outer",right_index=True,left_index=True)
                prediction_of_FEP.to_csv(csv_save_dir, index=0)


                # corr
                spearmans = []
                pearsons = []
                kendalls = []

                spearmans_var = []
                pearsons_var = []
                kendalls_var = []

                spearmans_min = []
                pearsons_min = []
                kendalls_min = []

                spearmans_max = []
                pearsons_max = []
                kendalls_max = []
                for num_loader in range(len(test_corr_loader_FEP1)):
                    loader = test_corr_loader_FEP1[num_loader]
                    file_nm = test_file_names_FEP1[num_loader]
                    mae, rmse, mae_g, rmse_g, valid_prediction, valid_prediction_G, _ ,_,_= predict(args, model, loader, device)

                    df = pd.read_csv(
                        f"{save_dir}/data/test_set_fep_graph_rmH_I/input_files/1_reference/train_files/{file_nm}.csv")
                    df["predict_dmpnnmve_pic50"] = valid_prediction

                    abs_label = df.Lable1.values
                    abs_predict = np.array(df.predict_dmpnnmve_pic50.values).astype(float) + np.array(df.Lable2.values).astype(
                        float)

                    df["abs_label_p"] = abs_label
                    df["abs_predict_p"] = abs_predict

                    # =================以PIC50为单位====================
                    reference_num = df.reference_num.values
                    ligand1_num = df.Ligand1_num.values
                    _df = pd.DataFrame({"reference_num": reference_num, f"abs_label_{file_nm}": abs_label, f"abs_predict_{file_nm}": abs_predict, f"ligand1_num_{file_nm}": ligand1_num})

                    # ================ 用来画散点图的 ==============
                    if num_loader == 0:
                        corr_of_FEP1 = _df.groupby(f'ligand1_num_{file_nm}')[[f'abs_label_{file_nm}', f'abs_predict_{file_nm}']].mean().reset_index()
                    else:
                        corr_of_FEP1_ = _df.groupby(f'ligand1_num_{file_nm}')[[f'abs_label_{file_nm}', f'abs_predict_{file_nm}']].mean().reset_index()
                        corr_of_FEP1 = pd.merge(corr_of_FEP1, corr_of_FEP1_, how="outer",right_index=True,left_index=True)

                    # ================ 相关性指标 ==============
                    _df_group = _df.groupby('reference_num')  # [['abs_label', 'abs_predict']].mean().reset_index()

                    spearman_ = []
                    pearson_ = []
                    kendall_ = []
                    for _, _df_onegroup in _df_group:
                        spearman = \
                        _df_onegroup[[f"abs_label_{file_nm}", f"abs_predict_{file_nm}"]].corr(method='spearman').iloc[0, 1]
                        pearson = _df_onegroup[[f"abs_label_{file_nm}", f"abs_predict_{file_nm}"]].corr(method='pearson').iloc[
                            0, 1]
                        kendall = _df_onegroup[[f"abs_label_{file_nm}", f"abs_predict_{file_nm}"]].corr(method='kendall').iloc[
                            0, 1]
                        spearman_.append(spearman)
                        pearson_.append(pearson)
                        kendall_.append(kendall)
                    spearmans.append(np.mean(spearman_))
                    pearsons.append(np.mean(pearson_))
                    kendalls.append(np.mean(kendall_))

                    spearmans_var.append(np.var(spearman_))
                    pearsons_var.append(np.var(pearson_))
                    kendalls_var.append(np.var(kendall_))

                    spearmans_min.append(np.min(spearman_))
                    pearsons_min.append(np.min(pearson_))
                    kendalls_min.append(np.min(kendall_))

                    spearmans_max.append(np.max(spearman_))
                    pearsons_max.append(np.max(pearson_))
                    kendalls_max.append(np.max(kendall_))

                for m_ in range(len(test_file_names_FEP1)):
                    file_nm = test_file_names_FEP1[m_]
                    rmse__ = rmse_gs[m_]
                    s_ = spearmans[m_]
                    p_ = pearsons[m_]
                    k_ = kendalls[m_]

                    s_var_ = spearmans_var[m_]
                    p_var_ = pearsons_var[m_]
                    k_var_ = kendalls_var[m_]

                    s_max_ = spearmans_max[m_]
                    p_max_ = pearsons_max[m_]
                    k_max_ = kendalls_max[m_]

                    s_min_ = spearmans_min[m_]
                    p_min_ = pearsons_min[m_]
                    k_min_ = kendalls_min[m_]

                    logger_writer(f"{file_nm},RMSE:{rmse__},spearman:{s_},spearman_var:{s_var_},spearmans_min:{s_min_},spearmans_max:{s_max_},\
                                    pearson:{p_}, pearsons_var:{p_var_},pearson_min:{p_min_},pearsons_max:{p_max_},kendall:{k_},kendall_var:{k_var_},\
                                    kendall_min:{k_min_},kendalls_max:{k_max_}")

                logger_writer(f"FEP,RMSE:{np.mean(rmse_gs)},spearman:{np.mean(spearmans)},spearman_var:{np.mean(spearmans_var)},spearmans_min:{np.mean(spearmans_min)},spearmans_max:{np.mean(spearmans_max)},\
                                    pearson:{np.mean(pearsons)}, pearsons_var:{np.mean(pearsons_var)},pearson_min:{np.mean(pearsons_min)},pearsons_max:{np.mean(pearsons_max)},kendall:{np.mean(kendalls)}, \
                                    kendall_var:{np.mean(kendalls_var)},kendall_min:{np.mean(kendalls_min)},kendalls_max:{np.mean(kendalls_max)}")

                csv_save_dir = os.path.join(save_dir, f"FEP_corr_{epoch}_{batch_all}_{args.seed}.csv")
                corr_of_FEP = pd.merge(corr_of_FEP1, corr_of_FEP2, how="outer",right_index=True,left_index=True)
                corr_of_FEP.to_csv(csv_save_dir, index=0)

                test_time = time.time()

                logger_writer(f"test time {test_time - train_time}")


                Path = os.path.join(save_dir, f"model_{epoch}_{batch_all}_{args.seed}.pth")
                torch.save(model, Path)
                break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 常用
    parser.add_argument('--log_frequency', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--loss_function', type=str, default="mse",
                        help='The loss function used to train the model: mse, smoothl1, mve, evidential')
    parser.add_argument("--device", type=int, default=0,
                        help="The number of device")
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=2)
    parser.add_argument('--cpu_num', type=int, default=4)
    parser.add_argument('--cs', type=int, default=0,help=("0: close cross docking, 1: open cross docking"))
    parser.add_argument('--two_task', type=int, default=0,help=("0: just regeresion, 1: regeresion + classfication"))


    # 超参
    parser.add_argument('--label_scalar', type=int, default=0,help=("0: close scalar, else: open scalar"))
    parser.add_argument('--continue_learning', type=int, default=1 ,help=("0: 关闭, 1: 打开"))

    parser.add_argument('--GCN_', type=int, default=0 ,help=("0: 关闭, 1: 打开"))
    parser.add_argument('--degree_information', type=int, default=1 ,help=("0: 关闭, 1: 打开"))

    parser.add_argument('--early_stopping_indicator', type=str, default="pearson")
    parser.add_argument('--hidden_dim', type=int, default=200)
    parser.add_argument('--radius', type=int, default=3)
    parser.add_argument('--T', type=int, default=3)
    parser.add_argument('--p_dropout', type=float, default=0.2)
    parser.add_argument('--ffn_num_laryers', type=int, default=3)
    parser.add_argument('--init_lr', type=float, default=0.0001)
    parser.add_argument('--max_lr', type=float, default=0.001)
    parser.add_argument('--final_lr', type=float, default=0.0001)

    parser.add_argument('--result_file', type=str, default=6)
    # message passing model
    parser.add_argument('--encoder_type', type=str, default="DMPNN_res",
                        help="DMPNN_res, SIGN")
    parser.add_argument('--readout_type', type=str, default="atomcrossatt_pair")

    # retrain
    parser.add_argument('--retrain', type=int, default=0,
                        help='Whether to continue training with an incomplete training model')
    parser.add_argument('--retrain_model_path', type=str,
                        help='the dir path of the incomplete training model needed to retrain')
    parser.add_argument('--train_path', type=str,default=f"{code_path}/data/mean_600000train_all_pair_withoutFEP.csv")
    parser.add_argument('--val_path', type=str,default=f"")
    
    parser.add_argument('--fold', type=str,default="0.1")


    args = parser.parse_args()

    setup_cpu(args.cpu_num)
    setup_seed(args.seed)

    cuda = "cuda:" + str(args.device)

    # define model
    if args.retrain == 1:
        model = torch.load(args.retrain_model_path)
        model.to(cuda)
    else:
        if args.loss_function == 'mve':
            model = DMPNN(hidden_dim=args.hidden_dim,
                          radius=args.radius,
                          T=args.T,
                          p_dropout=args.p_dropout,
                          ffn_num_layers=args.ffn_num_laryers,
                          encoder_type=args.encoder_type,
                          readout_type=args.readout_type,
                          output_dim=2,
                          degree_information=args.degree_information,
                          GCN_=args.GCN_,
                          cs=args.cs,
                          two_task=args.two_task).to(cuda)
        elif args.loss_function == 'evidential':
            model = DMPNN(hidden_dim=args.hidden_dim,
                          radius=args.radius,
                          T=args.T,
                          p_dropout=args.p_dropout,
                          ffn_num_layers=args.ffn_num_laryers,
                          encoder_type=args.encoder_type,
                          readout_type=args.readout_type,
                          output_dim=4,
                          degree_information=args.degree_information,
                          GCN_=args.GCN_,
                          cs=args.cs,
                          two_task=args.two_task).to(cuda)
        else:
            model = DMPNN(hidden_dim=args.hidden_dim,
                          radius=args.radius,
                          T=args.T,
                          p_dropout=args.p_dropout,
                          ffn_num_layers=args.ffn_num_laryers,
                          encoder_type=args.encoder_type,
                          readout_type=args.readout_type,
                          output_dim=1,
                          degree_information=args.degree_information,
                          GCN_=args.GCN_,
                          cs=args.cs,
                          two_task=args.two_task).to(cuda)
        initialize_weights(model)
        # model.mu_and_v()

    scalar = None

    train_dataset = LeadOptDataset(args.train_path, scalar)
    file_names = train_dataset.file_names_()

    label_scalar = None
    num_training_samples = len(train_dataset)

    #  1打开0关闭
    if args.continue_learning == 0:
        continue_learning = False
    elif args.continue_learning == 1:
        continue_learning = True

    # valid_dataset = LeadOptDataset(args.val_path)
    train_dataloader = GraphDataLoader(train_dataset, collate_fn=collate_fn, batch_size=args.batch_size,
                                       drop_last=False, shuffle=(not continue_learning),
                                       num_workers=args.num_workers, pin_memory=False)
    valid_dataloader = None
    test_file_name_FEP2 = [i for i in os.listdir(f"{code_path}/data/test_set_fep+_graph_rmH_I/") if i != "input_files"]
    test_file_name_FEP1 = [i for i in os.listdir(f"{code_path}/data/test_set_fep_graph_rmH_I/") if i != "input_files"]

    test_rmse_loader_FEP2 = []
    test_corr_loader_FEP2 = []
    for y in test_file_name_FEP2:
        test_dataset = LeadOptDataset(f"{code_path}/data/test_set_fep+_graph_rmH_I/input_files/0_reference/train_files/{y}.csv")
        test_dataloader = GraphDataLoader(test_dataset, collate_fn=collate_fn, batch_size=args.batch_size,
                                           drop_last=False, shuffle=False,
                                           num_workers=args.num_workers, pin_memory=True)
        test_rmse_loader_FEP2.append(test_dataloader)

        test_dataset = LeadOptDataset(f"{code_path}/data/test_set_fep+_graph_rmH_I/input_files/1_reference/train_files/{y}.csv")
        test_dataloader = GraphDataLoader(test_dataset, collate_fn=collate_fn, batch_size=args.batch_size,
                                           drop_last=False, shuffle=False,
                                           num_workers=args.num_workers, pin_memory=True)
        test_corr_loader_FEP2.append(test_dataloader)


    test_rmse_loader_FEP1 = []
    test_corr_loader_FEP1 = []
    for y in test_file_name_FEP1:
        test_dataset = LeadOptDataset(f"{code_path}/data/test_set_fep_graph_rmH_I/input_files/0_reference/train_files/{y}.csv")
        test_dataloader = GraphDataLoader(test_dataset, collate_fn=collate_fn, batch_size=args.batch_size,
                                           drop_last=False, shuffle=False,
                                           num_workers=args.num_workers, pin_memory=True)
        test_rmse_loader_FEP1.append(test_dataloader)

        test_dataset = LeadOptDataset(f"{code_path}/data/test_set_fep_graph_rmH_I/input_files/1_reference/train_files/{y}.csv")
        test_dataloader = GraphDataLoader(test_dataset, collate_fn=collate_fn, batch_size=args.batch_size,
                                           drop_last=False, shuffle=False,
                                           num_workers=args.num_workers, pin_memory=True)
        test_corr_loader_FEP1.append(test_dataloader)

    train(args, model, train_dataloader, valid_dataloader,test_rmse_loader_FEP2, test_corr_loader_FEP2, test_file_name_FEP2,
          test_rmse_loader_FEP1, test_corr_loader_FEP1, test_file_name_FEP1, label_scalar, cuda, num_training_samples,file_names=file_names)
