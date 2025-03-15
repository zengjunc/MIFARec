# -*- coding: UTF-8 -*-
# @Author  : Anonymous
# @Email   : Anonymous


import torch
import torch.nn as nn
import numpy as np
import scipy.sparse as sp
import utils

from models.BaseModel import SequentialModel
from models.BaseImpressionModel import ImpressionSeqModel
from utils import layers

class MIFARecBase(object):
    @staticmethod
    def parse_model_args(parser):
        parser.add_argument('--emb_size', type=int, default=64,
                            help='Size of embedding vectors.')
        parser.add_argument('--num_layers', type=int, default=1,
                            help='Number of self-attention layers.')
        parser.add_argument('--num_heads', type=int, default=4,
                            help='Number of attention heads.')
        parser.add_argument('--time_size', type=int, default=256,
                            help='Size of embedding vectors.')
        parser.add_argument('--max_time', type=int, default=256,
                            help='Max time intervals.')
        return parser

    def _base_init(self, args, corpus):
        self.emb_size = args.emb_size
        self.time_size = args.time_size
        self.max_his = args.history_max
        self.num_layers = args.num_layers
        self.num_heads = args.num_heads
        self.max_time = args.max_time
        self.len_range = torch.from_numpy(np.arange(self.max_his)).to(self.device)
        self._base_define_params()
        self.apply(self.init_weights)
        self.R = 0
        self.gram_matrix = 0

        # Obtain all interaction times before training, and calculate the maximum, minimum, and minimum time interval to normalize time information
        time_seqs = []
        # The training set provides the minimum timestamp
        for u, user_df in corpus.all_df.groupby('user_id'):
            time_seqs.extend(user_df['time'].values.tolist())
        # The test set provides the maximum timestamp
        for u, user_df in corpus.data_df['test'].groupby('user_id'):
            time_seqs.extend(user_df['time'].values.tolist())
        time_seqs = sorted(set([int(_) for _ in time_seqs]))
        self.min_timestamp = time_seqs[0]
        self.max_timestamp = time_seqs[-1]
        self.min_interval = 0xFFFFFFFF
        for idx in range(len(time_seqs) - 1):
            self.min_interval = min(self.min_interval, time_seqs[idx+1] - time_seqs[idx])
        if self.min_interval == 0:
            self.min_interval = 1
        self.min_timestamp_converted = 0
        self.max_timestamp_converted = (self.max_timestamp - self.min_timestamp) / self.min_interval


    def _base_define_params(self):
        # Initialize item Embedding, position Embedding, and time Embedding
        self.i_embeddings = nn.Embedding(self.item_num, self.emb_size)
        self.p_embeddings = nn.Embedding(self.max_his + 1, self.emb_size)
        self.t_embeddings = nn.Embedding(self.max_time + 2, self.emb_size)

        # Use only a single layer of self-attention encoding block
        self.transformer_block = nn.ModuleList([
            layers.TransformerLayer_Rt(d_model=self.emb_size, d_ff=self.emb_size, d_t=self.time_size, n_heads=self.num_heads,
                                    dropout=self.dropout, kq_same=False)
            for _ in range(self.num_layers)
        ])

    def forward(self, feed_dict):
        self.check_list = []

        # Algorithm input section
        u_ids = feed_dict['user_id']  # User IDs; MIFARec does not directly model user features, so user IDs are not used
        i_ids = feed_dict['item_id']  # Target item IDs; during training: 1 positive item + 1 negative item, during validation/testing: 1 positive item + 999 negative items
        history = feed_dict['history_items']  # Sequence of items in the session
        t_history = feed_dict['history_times']  # Timestamps of historical interactions
        t_target = feed_dict['target_time']  # Actual timestamps corresponding to the target positive item IDs, consistent with item_id[:, 0], simulating real-time recommendation effects
        lengths = feed_dict['lengths']  # Effective length of user session sequences
        batch_size, seq_len = history.shape
        valid_his = (history > 0).long()

        # Interest encoding
        interests_sim = self.gram_matrix[history]
        interests_input = interests_sim @ self.i_embeddings.weight
        his_vectors = interests_input

        # Position encoding
        position = (lengths[:, None] - self.len_range[None, :seq_len]) * valid_his
        pos_vectors = self.p_embeddings(position)

        # Time encoding
        # First, obtain the time interval (current_interval) between each interaction and the latest interaction
        t_history = t_history
        t_history = (t_history - self.min_timestamp).relu()
        t_history = t_history / self.min_interval
        realtime = (t_target - self.min_timestamp) / self.min_interval
        current_interval = realtime.unsqueeze(-1).expand_as(t_history) - t_history
        convert_log_a = torch.pow(torch.tensor(self.max_timestamp_converted), torch.tensor(1. / self.max_time))
        idx = (torch.log(current_interval + 1) / torch.log(convert_log_a)).int()
        time_vectors = self.t_embeddings(idx) # Monotonic part completed, but the convolutional part is not yet included

        # Combine interest encoding, position encoding, and time encoding to obtain the hybrid interest Embedding
        his_vectors = his_vectors + pos_vectors + time_vectors

        # Perform self-attention encoding
        causality_mask = np.tril(np.ones((1, 1, seq_len, seq_len), dtype=np.int32))
        attn_mask = torch.from_numpy(causality_mask).to(torch.device('cuda'))
        attn_mask_full = torch.ones_like(attn_mask) # Unlike the SASRec architecture, MIFARec does not use a sequential mask
        for block in self.transformer_block:
            his_vectors, weight_t = block(his_vectors, attn_mask_full) # Obtain the encoded hybrid interest and frequency domain information respectively
        his_vectors = his_vectors * valid_his[:, :, None].float()

        # Normalize the time interval granularity, adjust its unit to seconds, and specify the frequency domain boundaries from a periodic perspective; there is redundant encoding here
        if self.min_interval == 86400:
            unit_oneday = 1
        elif self.min_interval == 1:
            unit_oneday = 86400
        else:
            unit_oneday = 86400 / self.min_interval
        first_element = unit_oneday * 1
        last_element = unit_oneday * 10 * 365

        # Calculate omega
        length = self.time_size
        ratio = (last_element - first_element) / length
        period = (first_element + (ratio * torch.arange(length))).float().to(self.device)
        omega = (2 * torch.pi / period).to(self.device)
        # Obtain the relative weights (alpha) for each frequency band using the Fourier series formula
        time_attenuation = period[None, None, :] / (period[None, None, :] + 0.01 * current_interval[:, :, None]) # Add periodic loss
        alpha = weight_t * time_attenuation * ((torch.cos(current_interval[:, :, None] * omega[None, None, :]) + 1) / 2)
        numda = alpha.sum(-1)

        # Obtain the embeddings of positive and negative items
        i_vectors = self.i_embeddings(i_ids)
        # Obtain the weighted hybrid interest representation
        his_vectors = his_vectors * numda[:, :, None]
        his_vectors = his_vectors * valid_his[:, :, None].float()

        # Output side: Compute the inner product of all vectors together and calculate the weighted matching value
        prediction = (his_vectors[:, None, :, :] * i_vectors[:, :, None, :])
        prediction = prediction.sum(-1).sum(-1)
        prediction = prediction[:, :] / lengths[:, None]

        return {'prediction': prediction.view(batch_size, -1)}


class MIFARec(SequentialModel, MIFARecBase):
    reader = 'SeqReader'
    runner = 'BaseRunner'
    extra_log_args = ['emb_size', 'num_layers', 'num_heads']

    @staticmethod
    def parse_model_args(parser):
        parser = MIFARecBase.parse_model_args(parser)
        return SequentialModel.parse_model_args(parser)

    def __init__(self, args, corpus):
        SequentialModel.__init__(self, args, corpus)
        self._base_init(args, corpus)

    # Obtain the item similarity matrix based on the interaction matrix
    def get_gram_matrix(self, dataset):
        R = torch.zeros(self.user_num, self.item_num)
        for (user_index, item_index) in zip(dataset.data['user_id'], dataset.data['item_id']):
            R[user_index, item_index] = 1
        self.R = R.to(self.device)

        # Obtain the first-order similarity matrix
        row_sum = np.array(R.sum(axis=1))
        d_inv = np.power(row_sum, -0.5).flatten()
        d_inv[np.isposinf(d_inv)] = 0.
        d_mat = sp.diags(d_inv)
        norm_mat = d_mat.dot(R)
        col_sum = np.array(R.sum(axis=0))
        d_inv = np.power(col_sum, -0.5).flatten()
        d_inv[np.isposinf(d_inv)] = 0.
        d_mat = sp.diags(d_inv)
        norm_mat = norm_mat.dot(d_mat.toarray()).astype(np.float32)
        gram_matrix = norm_mat.T.dot(norm_mat)
        gram_matrix =  torch.Tensor(gram_matrix).to(self.device)

        # Obtain higher-order similarity
        item_embedding_r2 = gram_matrix @ self.R.T
        gram_matrix_r2 = item_embedding_r2 @ item_embedding_r2.T
        gram_matrix_r2 =  torch.nn.functional.normalize(gram_matrix_r2)
        gram_matrix_r2 = gram_matrix_r2 / gram_matrix_r2.mean() * gram_matrix.mean()
        gram_matrix = gram_matrix * 0.8 + gram_matrix_r2 * 0.2

        # Take the top 10% of similarities
        top = int(self.item_num * 0.1)
        indices = torch.topk(gram_matrix, top, dim=1).indices
        gram_matrix_topk = torch.zeros_like(gram_matrix)
        gram_matrix_topk.scatter_(1, indices, gram_matrix.gather(1, indices))
        gram_matrix_topk = torch.nn.functional.normalize(gram_matrix_topk, p=2)
        self.gram_matrix = gram_matrix_topk

        return


    def forward(self, feed_dict):
        out_dict = MIFARecBase.forward(self, feed_dict)
        return {'prediction': out_dict['prediction']}
