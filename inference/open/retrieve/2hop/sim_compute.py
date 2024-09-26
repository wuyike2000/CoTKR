import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from sentence_transformers import SentenceTransformer, util
import torch
import torch.nn.functional as F

class Similarity:
    def __init__(self):
        # Load model
        self.model = SentenceTransformer('../../../../../pretrain/all-MiniLM-L6-v2',device='cuda:0')

    def compute(self, query, relations):
        embedding1 = self.model.encode(query, show_progress_bar=False,device='cuda:0',convert_to_tensor=True)
        embedding2 = self.model.encode(relations,batch_size=1024,show_progress_bar=False, device='cuda:0',convert_to_tensor=True)
        cosine_scores = util.pytorch_cos_sim(embedding1, embedding2)[0]
        sim_relations = list(zip(cosine_scores.tolist(), relations))
        sim_relations = sorted(sim_relations, key=lambda x: x[0], reverse=True)
        sorted_relations = [relation for _, relation in sim_relations]
        return sorted_relations