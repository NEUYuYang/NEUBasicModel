import torch
from model.layer import FactorizationMachine,FeatureEmbedding,FeaturesLinear

class FM(torch.nn.Module):
    def __init__(self, feature_num,model_param):
        super().__init__()
        self.embedding = FeatureEmbedding(feature_num, model_param['embedding_dim'])
        self.linear = FeaturesLinear(feature_num)
        self.fm = FactorizationMachine(model_param['reduce_sum'])
    
    def forward(self, x):
        x = x.long()
        x_embedding = self.embedding(x)
        #print(x_embedding.shape)
        output_linear = self.linear(x)
        output_fm = self.fm(x_embedding)
        #print(output_linear.shape)
        #print(output_fm.shape)
        logit = output_linear +  output_fm
        logit = torch.sigmoid(logit)
        return logit.view(-1)