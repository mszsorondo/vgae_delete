import argparse
import os
import time
from tqdm import tqdm
import dgl
import pdb
import model
import numpy as np
import scipy.sparse as sp
import torch
import sys
import torch.nn.functional as F
from dgl.data import CiteseerGraphDataset, CoraGraphDataset, PubmedGraphDataset
from input_data import load_data
from preprocess import (
    mask_test_edges,
    mask_test_edges_dgl,
    preprocess_graph,
    sparse_to_tuple,
)
from sklearn.metrics import average_precision_score, roc_auc_score

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

parser = argparse.ArgumentParser(description="Variant Graph Auto Encoder")
parser.add_argument(
    "--learning_rate", type=float, default=0.01, help="Initial learning rate."
)
parser.add_argument(
    "--epochs", "-e", type=int, default=200, help="Number of epochs to train."
)
parser.add_argument(
    "--hidden1",
    "-h1",
    type=int,
    default=32,
    help="Number of units in hidden layer 1.",
)
parser.add_argument(
    "--hidden2",
    "-h2",
    type=int,
    default=16,
    help="Number of units in hidden layer 2.",
)
parser.add_argument(
    "--datasrc",
    "-s",
    type=str,
    default="dgl",
    help="Dataset download from dgl Dataset or website.",
)
parser.add_argument(
    "--dataset", "-d", type=str, default="cora", help="Dataset string."
)
parser.add_argument("--gpu_id", type=int, default=0, help="GPU id to use.")
args = parser.parse_args()


# check device
device = torch.device(
    "cuda:{}".format(args.gpu_id) if torch.cuda.is_available() else "cpu"
)
print(device)
# device = "cpu"

# roc_means = []
# ap_means = []


def compute_loss_para(adj):
    pos_weight = (adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
    norm = (
        adj.shape[0]
        * adj.shape[0]
        / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)
    )
    weight_mask = adj.view(-1) == 1
    weight_tensor = torch.ones(weight_mask.size(0)).to(device)
    weight_tensor[weight_mask] = pos_weight
    return weight_tensor, norm


def get_acc(adj_rec, adj_label):
    labels_all = adj_label.view(-1).long()
    preds_all = (adj_rec > 0.5).view(-1).long()
    accuracy = (preds_all == labels_all).sum().float() / labels_all.size(0)
    return accuracy


def get_scores(edges_pos, edges_neg, adj_rec):
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    adj_rec = adj_rec.cpu()
    # Predict on test set of edges
    preds = []
    for e in edges_pos:
        preds.append(sigmoid(adj_rec[e[0], e[1]].item()))

    preds_neg = []
    for e in edges_neg:
        preds_neg.append(sigmoid(adj_rec[e[0], e[1]].data))

    preds_all = np.hstack([preds, preds_neg])
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])
    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)

    return roc_score, ap_score


def dgl_main(test_val_neg_to_pos_edge_ratio = 1, runs_dict = None, print_loopwise=True, dataset_name = "cora"):


    # Load from DGL dataset
    #supress prints for GraphDataset constructors...
    
    if dataset_name == "cora":
        dataset = CoraGraphDataset(reverse_edge=False,verbose=False)
    elif dataset_name == "citeseer":
        dataset = CiteseerGraphDataset(reverse_edge=False,verbose=False)
    elif dataset_name == "pubmed":
        dataset = PubmedGraphDataset(reverse_edge=False,verbose=False)
    else:
        raise NotImplementedError
    graph = dataset[0]


    # Extract node features
    feats = graph.ndata.pop("feat").to(device)
    in_dim = feats.shape[-1]

    # generate input
    adj_orig = graph.adj_external().to_dense()

    dataset_neg_to_pos_edge_ratio = ((graph.num_nodes()**2)-graph.num_edges())/graph.num_edges()
    print(f"There's {int(dataset_neg_to_pos_edge_ratio)} neg edges for each positive edge")
    runs_dict["ratio"] = dataset_neg_to_pos_edge_ratio

    # build test set with 10% positive links
    (
        train_edge_idx,
        val_edges,
        val_edges_false,
        test_edges,
        test_edges_false,
    ) = mask_test_edges_dgl(graph, adj_orig, test_val_neg_to_pos_edge_ratio)



    graph = graph.to(device)

    # create train graph
    train_edge_idx = torch.tensor(train_edge_idx).to(device)
    train_graph = dgl.edge_subgraph(graph, train_edge_idx, relabel_nodes=False)
    train_graph = train_graph.to(device)
    adj = train_graph.adj_external().to_dense().to(device)

    # compute loss parameters
    weight_tensor, norm = compute_loss_para(adj)

    # create model
    vgae_model = model.VGAEModel(in_dim, args.hidden1, args.hidden2)
    vgae_model = vgae_model.to(device)

    # create training component
    optimizer = torch.optim.Adam(vgae_model.parameters(), lr=args.learning_rate)
    """print(
        "Total Parameters:",
        sum([p.nelement() for p in vgae_model.parameters()]),
    )"""

    # create training epoch
    
    for epoch in tqdm(range(args.epochs), desc="Epochs"):
        t = time.time()

        # Training and validation using a full graph
        vgae_model.train()

        logits = vgae_model.forward(graph, feats)

        # compute loss
        loss = norm * F.binary_cross_entropy(
            logits.view(-1), adj.view(-1), weight=weight_tensor
        )

        kl_divergence = (
            0.5
            / logits.size(0)
            * (
                1
                + 2 * vgae_model.log_std
                - vgae_model.mean**2
                - torch.exp(vgae_model.log_std) ** 2
            )
            .sum(1)
            .mean()
        )
        loss -= kl_divergence

        # backward

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_acc = get_acc(logits, adj)

        val_roc, val_ap = get_scores(val_edges, val_edges_false, logits)

        # Print out performance
        if print_loopwise and ((epoch + 1)%5)==0:
            tqdm.write(
            "Epoch: " +
            str("%04d" % (epoch + 1)) +
            "\nval_roc=" +
            str("{:.5f}".format(val_roc)) +
            "val_ap="+
            str("{:.5f}".format(val_ap))
            
            )

    test_roc, test_ap = get_scores(test_edges, test_edges_false, logits)
    # roc_means.append(test_roc)
    # ap_means.append(test_ap)
    print(
        "End of training!",
        "test_roc=",
        "{:.5f}".format(test_roc),
        "test_ap=",
        "{:.5f}".format(test_ap),
    )
    runs_dict["roc"].append(test_roc)
    runs_dict["ap"].append(test_ap)


def web_main():
    adj, features = load_data(args.dataset)

    features = sparse_to_tuple(features.tocoo())

    # Store original adjacency matrix (without diagonal entries) for later
    adj_orig = adj
    adj_orig = adj_orig - sp.dia_matrix(
        (adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape
    )
    adj_orig.eliminate_zeros()

    (
        adj_train,
        train_edges,
        val_edges,
        val_edges_false,
        test_edges,
        test_edges_false,
    ) = mask_test_edges(adj)
    adj = adj_train

    # # Create model
    # graph = dgl.from_scipy(adj)
    # graph.add_self_loop()

    # Some preprocessing
    adj_normalization, adj_norm = preprocess_graph(adj)

    # Create model
    graph = dgl.from_scipy(adj_normalization)
    graph.add_self_loop()

    # Create Model
    pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
    norm = (
        adj.shape[0]
        * adj.shape[0]
        / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)
    )

    adj_label = adj_train + sp.eye(adj_train.shape[0])
    adj_label = sparse_to_tuple(adj_label)

    adj_norm = torch.sparse.FloatTensor(
        torch.LongTensor(adj_norm[0].T),
        torch.FloatTensor(adj_norm[1]),
        torch.Size(adj_norm[2]),
    )
    adj_label = torch.sparse.FloatTensor(
        torch.LongTensor(adj_label[0].T),
        torch.FloatTensor(adj_label[1]),
        torch.Size(adj_label[2]),
    )
    features = torch.sparse.FloatTensor(
        torch.LongTensor(features[0].T),
        torch.FloatTensor(features[1]),
        torch.Size(features[2]),
    )

    weight_mask = adj_label.to_dense().view(-1) == 1
    weight_tensor = torch.ones(weight_mask.size(0))
    weight_tensor[weight_mask] = pos_weight

    features = features.to_dense()
    in_dim = features.shape[-1]

    vgae_model = model.VGAEModel(in_dim, args.hidden1, args.hidden2)
    # create training component
    optimizer = torch.optim.Adam(vgae_model.parameters(), lr=args.learning_rate)
    print(
        "Total Parameters:",
        sum([p.nelement() for p in vgae_model.parameters()]),
    )

    def get_scores(edges_pos, edges_neg, adj_rec):
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        # Predict on test set of edges
        preds = []
        pos = []
        for e in edges_pos:
            # print(e)
            # print(adj_rec[e[0], e[1]])
            preds.append(sigmoid(adj_rec[e[0], e[1]].item()))
            pos.append(adj_orig[e[0], e[1]])

        preds_neg = []
        neg = []
        for e in edges_neg:
            preds_neg.append(sigmoid(adj_rec[e[0], e[1]].data))
            neg.append(adj_orig[e[0], e[1]])

        preds_all = np.hstack([preds, preds_neg])
        labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])
        roc_score = roc_auc_score(labels_all, preds_all)
        ap_score = average_precision_score(labels_all, preds_all)

        return roc_score, ap_score

    def get_acc(adj_rec, adj_label):
        labels_all = adj_label.to_dense().view(-1).long()
        preds_all = (adj_rec > 0.5).view(-1).long()
        accuracy = (preds_all == labels_all).sum().float() / labels_all.size(0)
        return accuracy

    # create training epoch
    for epoch in range(args.epochs):
        t = time.time()

        # Training and validation using a full graph
        vgae_model.train()

        logits = vgae_model.forward(graph, features)

        # compute loss
        loss = norm * F.binary_cross_entropy(
            logits.view(-1), adj_label.to_dense().view(-1), weight=weight_tensor
        )
        kl_divergence = (
            0.5
            / logits.size(0)
            * (
                1
                + 2 * vgae_model.log_std
                - vgae_model.mean**2
                - torch.exp(vgae_model.log_std) ** 2
            )
            .sum(1)
            .mean()
        )
        loss -= kl_divergence

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_acc = get_acc(logits, adj_label)

        val_roc, val_ap = get_scores(val_edges, val_edges_false, logits)

        # Print out performance
        print(
            "Epoch:",
            "%04d" % (epoch + 1),
            "train_loss=",
            "{:.5f}".format(loss.item()),
            "train_acc=",
            "{:.5f}".format(train_acc),
            "val_roc=",
            "{:.5f}".format(val_roc),
            "val_ap=",
            "{:.5f}".format(val_ap),
            "time=",
            "{:.5f}".format(time.time() - t),
        )

    test_roc, test_ap = get_scores(test_edges, test_edges_false, logits)
    print(
        "End of training!",
        "test_roc=",
        "{:.5f}".format(test_roc),
        "test_ap=",
        "{:.5f}".format(test_ap),
    )
    # roc_means.append(test_roc)
    # ap_means.append(test_ap)


# if __name__ == '__main__':
#     for i in range(10):
#         web_main()
#
#     roc_mean = np.mean(roc_means)
#     roc_std = np.std(roc_means, ddof=1)
#     ap_mean = np.mean(ap_means)
#     ap_std = np.std(ap_means, ddof=1)
#     print("roc_mean=", "{:.5f}".format(roc_mean), "roc_std=", "{:.5f}".format(roc_std), "ap_mean=",
#           "{:.5f}".format(ap_mean), "ap_std=", "{:.5f}".format(ap_std))

datasets = ["citeseer", "pubmed"]
def run():
    dataset_neg_to_pos_edge_ratios = [1, 5, 10, 20, 0.5, 0.2]
    results_all = {"cora": [], "citeseer": [], "pubmed": [], }
    for dataset in datasets:
        args.dataset = dataset
        runs_dict = {"roc": [], "ap": [], "dataset": dataset, "ratios": []}
        for ratio in dataset_neg_to_pos_edge_ratios:
            
            if args.datasrc == "dgl":
                print(f"Dataset name: {dataset}, ratio: {ratio}")
                dgl_main(test_val_neg_to_pos_edge_ratio=ratio, runs_dict=runs_dict, print_loopwise=True, dataset_name=dataset)
                print(runs_dict)
                print(f"Mean AUC for 10 runs: {np.array(runs_dict['roc']).mean()}")
                print(f"Mean AP for 10 runs: {np.array(runs_dict['ap']).mean()}")
                runs_dict["mean_auc"] = np.array(runs_dict['roc']).mean()
                runs_dict["mean_ap"] = np.array(runs_dict['ap']).mean()
                runs_dict["ratios"].append(ratio)
                results_all[dataset].append(runs_dict)
                time.sleep(1)

            elif args.datasrc == "website":
                web_main()
        #save results to file as object
        #import pickle
        #with open(f"results_{dataset}.pkl", "wb") as f:
            #pickle.dump(results_all["dataset"], f)
        #load results from file
        #with open(f"results_{args.datasrc}.pkl", "rb") as f:
            #results = pickle.load(f)

def run_custom_exp(dataset_name, neg_to_pos_edge_ratio):
    results_all = {"cora": [], "citeseer": [], "pubmed": [], }
    runs_dict = {"roc": [], "ap": [], "dataset": dataset_name, "ratios": []}
    print(f"Dataset name: {dataset_name}, ratio: {ratio}")
    dgl_main(test_val_neg_to_pos_edge_ratio=neg_to_pos_edge_ratio, 
             runs_dict=runs_dict, print_loopwise=True, dataset_name=dataset_name)
    print(runs_dict)
    print(f"Mean AUC for 10 runs: {np.array(runs_dict['roc']).mean()}")
    print(f"Mean AP for 10 runs: {np.array(runs_dict['ap']).mean()}")
    runs_dict["mean_auc"] = np.array(runs_dict['roc']).mean()
    runs_dict["mean_ap"] = np.array(runs_dict['ap']).mean()
    runs_dict["ratios"].append(ratio)
    results_all[dataset_name].append(runs_dict)
    time.sleep(1)

def original_run():
    for dataset in datasets:
        args.dataset = dataset
        dgl_main(test_val_neg_to_pos_edge_ratio=1, runs_dict = {"roc": [], "ap": []}, print_loopwise=True)



if __name__ == "__main__":
    #original_run()
    if sys.argv[1]!="original":
        dataset_name = sys.argv[2]
        ratio = sys.argv[3]
        run_custom_exp(dataset_name,ratio)
    else:
        original_run()     
        
    