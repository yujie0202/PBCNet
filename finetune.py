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

from Dataloader.dataloader import collate_fn, LeadOptDataset, LeadOptDataset_test
from ReadoutModel.readout_bind import DMPNN
from utilis.function import get_loss_func
from utilis.initial import initialize_weights
from utilis.scalar import StandardScaler
from utilis.scheduler import NoamLR_shan
from utilis.trick import Writer,makedirs
from utilis.utilis import gm_process
from tqdm import tqdm


def freezen(model):
    need_updata = ['FNN.0.weight', 'FNN.0.bias', 'FNN.2.weight', 'FNN.2.bias', 'FNN.4.weight', 'FNN.4.bias', 'FNN.6.weight', 'FNN.6.bias',
                   'FNN2.0.weight', 'FNN2.0.bias', 'FNN2.2.weight', 'FNN2.2.bias', 'FNN2.4.weight', 'FNN2.4.bias', 'FNN2.6.weight', 'FNN2.6.bias']

    for name, parameter in model.named_parameters():
        if name in need_updata:
            parameter.requires_grad = False
        else:
            parameter.requires_grad = True



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


        valid_labels += label.tolist()
        valid_1_labels += label1.tolist()
        ref_2_labels += label2.tolist()
        rank += rank1.tolist()
        file += file_name

    # print(valid_labels)
    # print(valid_prediction)


    mae = mean_absolute_error(valid_labels, valid_prediction)
    rmse = mean_squared_error(valid_labels, valid_prediction) ** 0.5

    valid_labels_G = np.log(np.power(10, -np.array(valid_labels).astype(float))) * 297 * 1.9872 * 1e-3
    valid_prediction_G = np.log(np.power(10, -np.array(valid_prediction).astype(float))) * 297 * 1.9872 * 1e-3

    mae_g = mean_absolute_error(valid_labels_G, valid_prediction_G)
    rmse_g = mean_squared_error(valid_labels_G, valid_prediction_G) ** 0.5

    valid_prediction = np.array(valid_prediction).flatten()
    valid_prediction_G = np.array(valid_prediction_G).flatten()

    return mae, rmse, mae_g, rmse_g, valid_prediction, valid_prediction_G, np.array(valid_labels)


@torch.no_grad()
def evaluate(args, model, loader, device):
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




# args: device, loss_function, continue_learning, retrain,batch_size,init_lr, max_lr, final_lr
def train(args, model, train_loader, val_loader, test_loader2, test_loader, device, num_training_samples, file_names,ref,num_again):

    global prediction_of_file
    train_start = time.time()
    # learning rate decay and optimizer
    opt = torch.optim.Adam(model.parameters(), lr=0.00001)
    loss_func = get_loss_func(args.loss_function)
    if args.two_task == 1:
        loss_func_class = get_loss_func("entropy")
    mae_func = torch.nn.L1Loss(reduction='sum')
    mse_func = torch.nn.MSELoss(reduction='sum')

    # "/home/yujie/leadopt/finetune_results/pfkfb3/ref2_results"
    # for log
    save_dir = os.path.join(code_path, "finetune_result", f"{args.finetune_filename}",f"ref{ref}_results")
    # save_dir = save_dir + f"_res_{args.readout_type}_{args.loss_function}_GCN{args.GCN_}_degree{args.degree_information}_hidden{args.hidden_dim}_cs{args.cs}_fold{args.fold}_seed{args.seed}"

    logger_writer = Writer(os.path.join(save_dir, f"{num_again}_record.txt"))

    if args.finetune_validation == 1:
        stop_metric = 0
        not_change_epoch = 0

    # for log in a batch defined step
    batch_all = 0
    loss_for_log = 0


    # without ligands
    mae, rmse, mae_g, rmse_g, valid_prediction, valid_prediction_G, valid_labels = predict(args, model, test_loader, device)

    df = pd.read_csv( f"{code_path}/data/finetune_input_files/{ref}_reference/{args.which_fep}_temp_predict.csv")
    abs_label = np.array(df.Lable.values).astype(float) + np.array(df.Lable2.values).astype(float)
    abs_predict = np.array(valid_prediction).astype(float) + np.array(df.Lable2.values).astype(float)

    # ================= 相关系数的计算 ====================
    Ligand1 = df.Ligand1_num.values

    _df = pd.DataFrame({"Ligand1":Ligand1, "abs_label":abs_label, "abs_predict":abs_predict})
    _df_group = _df.groupby('Ligand1')[['abs_label', 'abs_predict']].mean().reset_index()

    spearman = _df_group[["abs_label", "abs_predict"]].corr(method='spearman').iloc[0, 1]
    pearson = _df_group[["abs_label", "abs_predict"]].corr(method='pearson').iloc[0, 1]
    kendall = _df_group[["abs_label", "abs_predict"]].corr(method='kendall').iloc[0, 1]

    logger_writer(f"without ligands RMSE_G {rmse_g} Spearman {spearman} Pearson {pearson} Kendall {kendall}")


    # with ligands
    mae2, rmse2, mae_g2, rmse_g2, valid_prediction2, valid_prediction_G2, valid_labels2 = predict(args, model, test_loader2, device)

    df2 = pd.read_csv( f"{code_path}/data/finetune_input_files/{ref}_reference/{args.which_fep}_temp_predict_withtuneligs.csv")
    abs_label2 = np.array(df2.Lable.values).astype(float) + np.array(df2.Lable2.values).astype(float)
    abs_predict2 = np.array(valid_prediction2).astype(float) + np.array(df2.Lable2.values).astype(float)

    Ligand1_2 = df2.Ligand1_num.values

    _df2 = pd.DataFrame({"Ligand1":Ligand1_2, "abs_label":abs_label2, "abs_predict":abs_predict2})
    _df_group2 = _df2.groupby('Ligand1')[['abs_label', 'abs_predict']].mean().reset_index()

    spearman2 = _df_group2[["abs_label", "abs_predict"]].corr(method='spearman').iloc[0, 1]
    pearson2 = _df_group2[["abs_label", "abs_predict"]].corr(method='pearson').iloc[0, 1]
    kendall2 = _df_group2[["abs_label", "abs_predict"]].corr(method='kendall').iloc[0, 1]


    logger_writer(f"with ligands RMSE_G {rmse_g2} Spearman {spearman2} Pearson {pearson2} Kendall {kendall2}")
    logger_writer(" ")




    for epoch in range(args.finetune_epoch):
        model.train()
        start = time.time()
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
                # print(label.float())
                # print(logits.squeeze().float())
                # print(logits.float())

                loss = loss_func(logits.squeeze(dim=-1).float(), label.float())
                train_mae_ = mae_func(logits.squeeze(dim=-1).float(), label.float())
                train_mse_ = mse_func(logits.squeeze(dim=-1).float(), label.float())


            # loss_mse.append(loss.item())

            if args.two_task == 1:

                loss_class = loss_func_class(logits_class.float(), label.float())

                # loss_entropy.append(loss_class.item())

                # loss = loss + loss_class

            opt.zero_grad()
            loss.backward()
            opt.step()
            batch_all += 1
            loss_for_log += loss.item()
            training_mae += train_mae_.item()
            training_mse += train_mse_.item()

        _loss = loss_for_log / num_training_samples  # mean loss for each batch with size of log_frequency.

        logger_writer("    ")
        print(f"Epoch {epoch}  Batch {batch_all}  Loss {_loss}")
        logger_writer(f"Epoch {epoch}  Loss {_loss}")
        loss_for_log = 0


        # ============= 每一个Epoch都要之后都要进行一次test上的估计 ==============
        train_time = time.time()
        training_rmse = (training_mse / num_training_samples) ** 0.5
        training_mae = training_mae / num_training_samples

        # without ligands
        mae, rmse, mae_g, rmse_g, valid_prediction, valid_prediction_G, valid_labels = predict(args, model, test_loader, device)

        if epoch == 0:
            prediction_of_file = pd.DataFrame({f'prediction_ic50_{epoch}': valid_prediction,
                                               f'prediction_G_{epoch}': valid_prediction_G,
                                               f"label_ic50_{epoch}": valid_labels})
        else:
            prediction_of_file_ = pd.DataFrame({f'prediction_ic50_{epoch}': valid_prediction,
                                                f'prediction_G_{epoch}': valid_prediction_G,
                                                f"label_ic50_{epoch}": valid_labels})
            prediction_of_file = pd.merge(prediction_of_file, prediction_of_file_, how="outer",
                                          right_index=True, left_index=True)

        df = pd.read_csv( f"{code_path}/data/finetune_input_files/{ref}_reference/{args.which_fep}_temp_predict.csv")
        abs_label = np.array(df.Lable.values).astype(float) + np.array(df.Lable2.values).astype(float)
        abs_predict = np.array(valid_prediction).astype(float) + np.array(df.Lable2.values).astype(float)

        # ================= 相关系数的计算 ====================
        Ligand1 = df.Ligand1_num.values

        _df = pd.DataFrame({"Ligand1":Ligand1, "abs_label":abs_label, "abs_predict":abs_predict})
        _df_group = _df.groupby('Ligand1')[['abs_label', 'abs_predict']].mean().reset_index()

        spearman = _df_group[["abs_label", "abs_predict"]].corr(method='spearman').iloc[0, 1]
        pearson = _df_group[["abs_label", "abs_predict"]].corr(method='pearson').iloc[0, 1]
        kendall = _df_group[["abs_label", "abs_predict"]].corr(method='kendall').iloc[0, 1]

        logger_writer(f"Training Set mae {training_mae}")
        logger_writer(f"Training Set rmse {training_rmse}")
        logger_writer(f"Epoch {epoch}")
        logger_writer(f"without ligands RMSE_G {rmse_g} Spearman {spearman} Pearson {pearson} Kendall {kendall}")


        # with ligands
        mae2, rmse2, mae_g2, rmse_g2, valid_prediction2, valid_prediction_G2, valid_labels2 = predict(args, model, test_loader2, device)

        if epoch == 0:
            prediction_of_file2 = pd.DataFrame({f'prediction_ic50_{epoch}': valid_prediction2,
                                               f'prediction_G_{epoch}': valid_prediction_G2,
                                               f"label_ic50_{epoch}": valid_labels2})
        else:
            prediction_of_file_2 = pd.DataFrame({f'prediction_ic50_{epoch}': valid_prediction2,
                                                f'prediction_G_{epoch}': valid_prediction_G2,
                                                f"label_ic50_{epoch}": valid_labels2})
            prediction_of_file2 = pd.merge(prediction_of_file2, prediction_of_file_2, how="outer",
                                          right_index=True, left_index=True)

        df2 = pd.read_csv( f"{code_path}/data/finetune_input_files/{ref}_reference/{args.which_fep}_temp_predict_withtuneligs.csv")
        abs_label2 = np.array(df2.Lable.values).astype(float) + np.array(df2.Lable2.values).astype(float)
        abs_predict2 = np.array(valid_prediction2).astype(float) + np.array(df2.Lable2.values).astype(float)

        # ================= 相关系数的计算 ====================
        Ligand1_2 = df2.Ligand1_num.values

        _df2 = pd.DataFrame({"Ligand1":Ligand1_2, "abs_label":abs_label2, "abs_predict":abs_predict2})
        _df_group2 = _df2.groupby('Ligand1')[['abs_label', 'abs_predict']].mean().reset_index()

        spearman2 = _df_group2[["abs_label", "abs_predict"]].corr(method='spearman').iloc[0, 1]
        pearson2 = _df_group2[["abs_label", "abs_predict"]].corr(method='pearson').iloc[0, 1]
        kendall2 = _df_group2[["abs_label", "abs_predict"]].corr(method='kendall').iloc[0, 1]

        logger_writer(f"with ligands RMSE_G {rmse_g2} Spearman {spearman2} Pearson {pearson2} Kendall {kendall2}")
        logger_writer(" ")


        if args.finetune_validation == 1:
            val_mae, val_rmse, val_abs_mae, val_abs_rmse, val_spearman, val_pearson, corr_df = evaluate(args,
                                                                                                        model,
                                                                                                        val_loader,
                                                                                                        device)
            if args.early_stopping_indicator == "pearson":
                es = val_pearson
            elif args.early_stopping_indicator == "rmse":
                es = val_rmse
            if es > stop_metric:
                stop_metric = es
                not_change_epoch = 0
            else:
                not_change_epoch += 1
                logger_writer(f"Stop metric not change for {not_change_epoch}")
                logger_writer(f"Best Validation pearson {stop_metric}")

            if not_change_epoch >= args.patience:
                logger_writer("Stop Training")
                prediction_of_file.to_csv(f"/home/yujie/leadopt/results{args.result_file}/{args.finetune_filename}/ref{ref}_results/{num_again}_results.csv", index=0)
                return None

    logger_writer("Stop Training")

    # prediction_of_file.to_csv(
    #     f"/home/yujie/leadopt/results{args.result_file}/{args.finetune_filename}/ref{ref}_results/{num_again}_results.csv",
    #     index=0)
    # prediction_of_file2.to_csv(
    #     f"/home/yujie/leadopt/results{args.result_file}/{args.finetune_filename}/ref{ref}_results/{num_again}_results_withligands.csv",
    #     index=0)


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
    parser.add_argument('--label_scalar', type=int, default=0, help=("0: close scalar, else: open scalar"))
    parser.add_argument('--continue_learning', type=int, default=1, help=("0: 关闭, 1: 打开"))

    parser.add_argument('--GCN_', type=int, default=0, help=("0: 关闭, 1: 打开"))
    parser.add_argument('--degree_information', type=int, default=1, help=("0: 关闭, 1: 打开"))

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
    parser.add_argument('--train_path', type=str,
                        default="/home/yujie/leadopt/data/ic50_graph_rmH_new_2/train_1_pair.csv")
    parser.add_argument('--val_path', type=str,
                        default="/home/yujie/leadopt/data/ic50_graph_rmH_new_2/validation_1_pair.csv")

    parser.add_argument('--fold', type=str, default="0.1")
    parser.add_argument('--finetune_validation', type=int, default=0)
    parser.add_argument('--finetune_epoch', type=int, default=10)
    parser.add_argument('--finetune_filename', type=str, default="pfkfb3")
    parser.add_argument('--which_fep', type=str, default="fep2")
    # parser.add_argument('--finetune_filename', type=str, default="pfkfb3")



    args = parser.parse_args()

    setup_cpu(args.cpu_num)
    setup_seed(args.seed)

    cuda = "cuda:" + str(args.device)


    fep1 = ['PTP1B', 'Thrombin', 'Tyk2', 'CDK2', 'Jnk1', 'Bace', 'MCL1', 'p38']
    fep2 = ['syk', 'shp2','pfkfb3',  'eg5', 'cdk8', 'cmet', 'tnks2', 'hif2a']
    # fep2 = ['hif2a']


    if args.which_fep == "fep1":
        fep = fep1
    else:
        fep = fep2


    for finetune_filename in fep:

        args.finetune_filename = finetune_filename


        for ref in [6,10]:

            df_finetune_all = pd.read_csv(f"{code_path}/data/finetune_input_files/{ref}_reference/{args.which_fep}_finetune_.csv")
            df_prediction_all = pd.read_csv(f"{code_path}/data/finetune_input_files/{ref}_reference/{args.which_fep}_predict_.csv")
            df_prediction2_all = pd.read_csv(f"{code_path}/data/finetune_input_files/{ref}_reference/{args.which_fep}_predict_with_tuneligs_.csv")


            df_finetune_all = df_finetune_all[df_finetune_all.file_name == args.finetune_filename]
            df_prediction_all = df_prediction_all[df_prediction_all.file_name == args.finetune_filename]
            df_prediction2_all = df_prediction2_all[df_prediction2_all.file_name == args.finetune_filename]


            df_finetune_all_group = df_finetune_all.groupby("again_number")
            df_prediction_all_group = df_prediction_all.groupby("again_number")
            df_prediction2_all_group = df_prediction2_all.groupby("again_number")


            for again_num, df_prediction in tqdm(df_prediction_all_group):

                model = torch.load(code_path + args.retrain_model_path,map_location="cpu")
                model.to(cuda)
                # freezen(model)

                df_prediction.to_csv(
                    f"{code_path}/data/finetune_input_files/{ref}_reference/{args.which_fep}_temp_predict.csv",
                    index=0)

                # finetune
                for _again_num, df_finetune in df_finetune_all_group:
                    if _again_num == again_num:

                        # train_finetune_set = pd.read_csv("/home/yujie/leadopt/data/train_files_uni/training_data_300.csv")

                        # df_finetune = pd.concat([train_finetune_set, df_finetune], ignore_index = True)
                        df_finetune.to_csv(
                            f"{code_path}/data/finetune_input_files/{ref}_reference/{args.which_fep}_temp_finetune.csv",
                            index=0)

                # prediction with tune ligands
                for _again_num, df_finetune in df_prediction2_all_group:
                    if _again_num == again_num:

                        # train_finetune_set = pd.read_csv("/home/yujie/leadopt/data/train_files_uni/training_data_300.csv")

                        # df_finetune = pd.concat([train_finetune_set, df_finetune], ignore_index = True)
                        df_finetune.to_csv(
                            f"{code_path}/data/finetune_input_files/{ref}_reference/{args.which_fep}_temp_predict_withtuneligs.csv",
                            index=0)

                finetune_dataset = LeadOptDataset(
                    f"{code_path}/data/finetune_input_files/{ref}_reference/{args.which_fep}_temp_finetune.csv")

                num_training_samples = len(finetune_dataset)
                finetune_dataloader = GraphDataLoader(finetune_dataset, collate_fn=collate_fn, batch_size=args.batch_size,
                                                      drop_last=False, shuffle=True,
                                                      num_workers=args.num_workers, pin_memory=True)

                prediction_dataset = LeadOptDataset(
                    f"{code_path}/data/finetune_input_files/{ref}_reference/{args.which_fep}_temp_predict.csv")
                prediction_dataloader = GraphDataLoader(prediction_dataset, collate_fn=collate_fn,
                                                        batch_size=args.batch_size,
                                                        drop_last=False, shuffle=False,
                                                        num_workers=args.num_workers, pin_memory=True)

                prediction_dataset2 = LeadOptDataset(
                    f"{code_path}/data/finetune_input_files/{ref}_reference/{args.which_fep}_temp_predict_withtuneligs.csv")
                prediction_dataloader2 = GraphDataLoader(prediction_dataset2, collate_fn=collate_fn,
                                                        batch_size=args.batch_size,
                                                        drop_last=False, shuffle=False,
                                                        num_workers=args.num_workers, pin_memory=True)

                train(args=args, model=model, train_loader=finetune_dataloader, test_loader2=prediction_dataloader2,
                      test_loader=prediction_dataloader, device=cuda, num_training_samples=num_training_samples,
                      ref=ref, num_again=again_num,val_loader=None,file_names=None)
