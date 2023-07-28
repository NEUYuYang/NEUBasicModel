import torch.nn as nn
import torch
from model.layer import FeatureEmbedding
from model.layer import MultiLayerPerceptron
class MLP(nn.Module):
    def __init__(self, feature_num, model_param):
        super(MLP, self).__init__()
        self.embedding = FeatureEmbedding(feature_num, model_param['embedding_dim'])
        self.embed_output_dim = len(feature_num) * model_param['embedding_dim']
        self.mlp = MultiLayerPerceptron(self.embed_output_dim, model_param['hidden_dim'], model_param['dropout'])

    def forward(self, x):
        x = x.long()
        #print(x.shape)
        x_embedding = self.embedding(x)
        #print(x_embedding.shape)
        output = self.mlp(x_embedding.view(-1,self.embed_output_dim))
        #print(output.shape)
        output = torch.sigmoid(output.squeeze(1))
        #print(output.shape)
        return output
