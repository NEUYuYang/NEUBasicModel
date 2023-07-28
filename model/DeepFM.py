import torch
from model.layer import FactorizationMachine,FeatureEmbedding,FeaturesLinear,MultiLayerPerceptron
 
class DeepFM(torch.nn.Module):
    def __init__(self, feature_dim, model_param):
        super().__init__()
        self.linear = FeaturesLinear(feature_dim)
        self.fm = FactorizationMachine(model_param['reduce_sum'])
        self.embedding = FeatureEmbedding(feature_dim, model_param['embedding_dim'])
        self.embed_output_dim = len(feature_dim) * model_param['embedding_dim']
        self.mlp = MultiLayerPerceptron(self.embed_output_dim, model_param['hidden_dim'], model_param['dropout'])

    def forward(self, x):
        x = x.long()
        embed_x = self.embedding(x)
        x = self.linear(x) + self.fm(embed_x) + self.mlp(embed_x.view(-1, self.embed_output_dim))
        return torch.sigmoid(x.squeeze(1))