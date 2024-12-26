import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys
import torch
import torch.nn as nn
from torch.backends import cudnn

from sklearn.neighbors import KDTree

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

from loading_pointclouds import *
import models.PointNetVlad as PNV
from tqdm import tqdm
import numpy as np
import config as cfg

cudnn.enabled = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate():
    model = PNV.PointNetVlad(global_feat=True, feature_transform=True, max_pool=False,
                             input_dim=cfg.INPUT_DIM, output_dim=cfg.FEATURE_OUTPUT_DIM, num_points=cfg.NUM_POINTS)
    model = model.to(device)

    resume_filename = cfg.MODEL_FILENAME
    print("Resuming From ", resume_filename)
    checkpoint = torch.load(resume_filename)
    saved_state_dict = checkpoint['state_dict']
    model.load_state_dict(saved_state_dict)

    model = nn.DataParallel(model)

    print(evaluate_model(model))


def save_recall_results(recall_dict, one_percent_recall_dict, result_dir):
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    # 构建 all_recalls 数据
    all_recalls = {}
    for q_key, pair_recall in recall_dict.items():
        all_recalls[q_key] = {
            "recall_at_n": {
                1: pair_recall[0],  # Recall @1
                5: pair_recall[4],  # Recall @5
                10: pair_recall[9],  # Recall @10
                20: pair_recall[19]  # Recall @20
            },
            "recall_at_1_percent": one_percent_recall_dict[q_key]  # Top 1% Recall
        }

    # 计算平均 Recall
    ave_recall_1 = np.mean([metrics['recall_at_n'][1] for metrics in all_recalls.values()])
    ave_recall_5 = np.mean([metrics['recall_at_n'][5] for metrics in all_recalls.values()])
    ave_recall_10 = np.mean([metrics['recall_at_n'][10] for metrics in all_recalls.values()])
    ave_recall_20 = np.mean([metrics['recall_at_n'][20] for metrics in all_recalls.values()])
    ave_one_percent_recall = np.mean([metrics['recall_at_1_percent'] for metrics in all_recalls.values()])

    with open(cfg.OUTPUT_FILE, "w") as output:

        for q_key, metrics in all_recalls.items():
            output.write(f"Query Set: {q_key}\n")
            output.write(f"Recall@1: {metrics['recall_at_n'][1]:.4f}\n")
            output.write(f"Recall@5: {metrics['recall_at_n'][5]:.4f}\n")
            output.write(f"Recall@10: {metrics['recall_at_n'][10]:.4f}\n")
            output.write(f"Recall@20: {metrics['recall_at_n'][20]:.4f}\n")
            output.write(f"Top 1% Recall: {metrics['recall_at_1_percent']:.4f}\n\n")

        output.write(f"Average Recall @1: {ave_recall_1:.4f}\n")
        output.write(f"Average Recall @5: {ave_recall_5:.4f}\n")
        output.write(f"Average Recall @10: {ave_recall_10:.4f}\n")
        output.write(f"Average Recall @20: {ave_recall_20:.4f}\n")
        output.write(f"Average Top 1% Recall: {ave_one_percent_recall:.4f}\n")

    print(f"==> Results saved to {result_dir}")


def evaluate_model(model):
    EVAL_DATABASE_FILE = os.path.join(cfg.DATASET_FOLDER, cfg.EVAL_DATABASE_FILE)
    EVAL_QUERY_FILE = os.path.join(cfg.DATASET_FOLDER, cfg.EVAL_QUERY_FILE)
    DATABASE_SETS = get_sets_dict(EVAL_DATABASE_FILE)
    QUERY_SETS = get_sets_dict(EVAL_QUERY_FILE)

    if not os.path.exists(cfg.RESULTS_FOLDER):
        os.mkdir(cfg.RESULTS_FOLDER)

    recall = np.zeros(25)
    count = 0

    one_percent_recall_dict = {}
    recall_dict = {}

    DATABASE_VECTORS = []
    QUERY_VECTORS = []


    for set in tqdm(DATABASE_SETS, desc='Computing database embeddings'):
        DATABASE_VEC=get_latent_vectors(model, DATABASE_SETS[set])
        DATABASE_VECTORS.append(DATABASE_VEC)

    for seq_set in tqdm(QUERY_SETS, desc='Computing query embeddings'):
        QUERY_VEC = get_latent_vectors(model, QUERY_SETS[seq_set])
        QUERY_VECTORS.append(QUERY_VEC)

    for i, g_key in enumerate(DATABASE_SETS):
        for j, q_key in enumerate(QUERY_SETS):
            if g_key != q_key:
                continue
            pair_recall, pair_opr = get_recall(i, j, g_key, q_key, DATABASE_VECTORS, QUERY_VECTORS, QUERY_SETS)
            recall += np.array(pair_recall)
            recall_dict[q_key] = pair_recall
            count += 1
            one_percent_recall_dict[q_key] = pair_opr


    ave_recall = recall / count
    ave_one_percent_recall = np.mean(list(one_percent_recall_dict.values()))

    save_recall_results(
        recall_dict=recall_dict,
        one_percent_recall_dict=one_percent_recall_dict,
        result_dir=cfg.RESULTS_FOLDER
    )

    # with open(cfg.OUTPUT_FILE, "w") as output:
    #     output.write("Average Recall @N:\n")
    #     output.write(str(ave_recall))
    #     output.write("\n\n")
    #     output.write("\n\n")
    #     output.write("Average Top 1% Recall:\n")
    #     output.write(str(ave_one_percent_recall))

    return ave_recall, ave_one_percent_recall


def get_latent_vectors(model, set):

    model.eval()
    train_file_idxs = np.arange(0, len(set))

    batch_num = cfg.EVAL_BATCH_SIZE
    q_output = []

    for q_index in range(len(train_file_idxs) // batch_num):
        file_indices = train_file_idxs[q_index * batch_num:(q_index+1)*(batch_num)]
        file_names = []
        for index in file_indices:
            file_names.append(set[index]["query"])
        queries = load_pc_files(file_names)

        with torch.no_grad():
            feed_tensor = torch.from_numpy(queries).float()
            if cfg.INPUT_DIM == 3:
                feed_tensor = feed_tensor[:, :, :3]
            elif cfg.INPUT_DIM == 4:
                feed_tensor = feed_tensor[:, :, :4]
            feed_tensor = feed_tensor.unsqueeze(1)
            feed_tensor = feed_tensor.to(device)
            out = model(feed_tensor)
            # out, _ = model(feed_tensor)

        out = out.detach().cpu().numpy()
        out = np.squeeze(out)

        #out = np.vstack((o1, o2, o3, o4))
        q_output.append(out)

    q_output = np.array(q_output)
    if(len(q_output) != 0):
        q_output = q_output.reshape(-1, q_output.shape[-1])

    # handle edge case
    index_edge = len(train_file_idxs) // batch_num * batch_num
    if index_edge < len(set):
        file_indices = train_file_idxs[index_edge:len(set)]
        file_names = []
        for index in file_indices:
            file_names.append(set[index]["query"])
        queries = load_pc_files(file_names)

        with torch.no_grad():
            feed_tensor = torch.from_numpy(queries).float()
            if cfg.INPUT_DIM == 3:
                feed_tensor = feed_tensor[:, :, :3]
            elif cfg.INPUT_DIM == 4:
                feed_tensor = feed_tensor[:, :, :4]
            feed_tensor = feed_tensor.unsqueeze(1)
            feed_tensor = feed_tensor.to(device)
            o1 = model(feed_tensor)
            # o1, _ = model(feed_tensor)

        output = o1.detach().cpu().numpy()
        output = np.squeeze(output)
        if (q_output.shape[0] != 0):
            q_output = np.vstack((q_output, output))
        else:
            q_output = output
    return q_output


def get_recall(m, n, g_key, q_key, database_vectors, query_vectors, query_sets):
    database_output = database_vectors[m]
    queries_output = query_vectors[n]

    # When embeddings are normalized, using Euclidean distance gives the same
    # nearest neighbour search results as using cosine distance
    database_nbrs = KDTree(database_output)

    num_neighbors = 25
    recall = [0] * num_neighbors

    one_percent_retrieved = 0
    threshold = max(int(round(len(database_output) / 100.0)), 1)

    num_evaluated = 0
    for i in range(len(queries_output)):
        # i is query element ndx
        query_details = query_sets[q_key][i]
        true_neighbors = query_details[g_key]
        if len(true_neighbors) == 0:
            continue
        num_evaluated += 1

        # Find nearest neightbours
        distances, indices = database_nbrs.query(np.array([queries_output[i]]), k=num_neighbors)

        for j in range(len(indices[0])):
            if indices[0][j] in true_neighbors:
                recall[j] += 1
                break

        if len(list(set(indices[0][0:threshold]).intersection(set(true_neighbors)))) > 0:
            one_percent_retrieved += 1

    one_percent_recall = (one_percent_retrieved / float(num_evaluated)) * 100
    recall = (np.cumsum(recall) / float(num_evaluated)) * 100

    return recall, one_percent_recall

if __name__ == "__main__":
    # params
    parser = argparse.ArgumentParser()
    parser.add_argument('--positives_per_query', type=int, default=4,
                        help='Number of potential positives in each training tuple [default: 2]')
    parser.add_argument('--negatives_per_query', type=int, default=12,
                        help='Number of definite negatives in each training tuple [default: 20]')
    parser.add_argument('--eval_batch_size', type=int, default=12,
                        help='Batch Size during training [default: 1]')
    parser.add_argument('--dimension', type=int, default=256)
    parser.add_argument('--decay_step', type=int, default=200000,
                        help='Decay step for lr decay [default: 200000]')
    parser.add_argument('--decay_rate', type=float, default=0.7,
                        help='Decay rate for lr decay [default: 0.8]')
    parser.add_argument('--results_dir', default='results/',
                        help='results dir [default: results]')
    parser.add_argument('--dataset_folder', default='../../dataset/',
                        help='PointNetVlad Dataset Folder')
    FLAGS = parser.parse_args()

    #BATCH_SIZE = FLAGS.batch_size
    #cfg.EVAL_BATCH_SIZE = FLAGS.eval_batch_size
    cfg.NUM_POINTS = 1024
    cfg.FEATURE_OUTPUT_DIM = 256
    cfg.DATA_DIM = 5
    cfg.INPUT_DIM = 4
    cfg.EVAL_POSITIVES_PER_QUERY = FLAGS.positives_per_query
    cfg.EVAL_NEGATIVES_PER_QUERY = FLAGS.negatives_per_query
    cfg.DECAY_STEP = FLAGS.decay_step
    cfg.DECAY_RATE = FLAGS.decay_rate

    cfg.RESULTS_FOLDER = FLAGS.results_dir

    cfg.EVAL_DATABASE_FILE = 'generating_queries/oxford_evaluation_database.pickle'
    cfg.EVAL_QUERY_FILE = 'generating_queries/oxford_evaluation_query.pickle'

    cfg.LOG_DIR = 'log/'
    data_type = cfg.DATASET_FOLDER.split('/')[-1]
    cfg.OUTPUT_FILE = os.path.join(cfg.RESULTS_FOLDER, data_type + '_results.txt')

    cfg.DATASET_FOLDER = FLAGS.dataset_folder

    evaluate()
