from torch_geometric.datasets import Planetoid
import torch_geometric.nn as hnn
import torch.nn.functional as F

class HyperGraph_Class(torch.nn.Module):
    def __init__(self,data):
        super(Hyper_Attention_Class, self).__init__()
        self.hconv1 = hnn.HypergraphConv(
            data.num_features,
            8,
            use_attention=True,
            heads=8,
            concat=True,
            bias=True,
            dropout=0.6
        )
        self.hconv2 = hnn.HypergraphConv(
            8 * 8,
            8,
            use_attention=False,
            heads=1,
            concat=True,
            bias=True,
            dropout=0.6
        )

    def forward(self, data):
        x = F.dropout(data.x, p=0.6, training=self.training)
        x = F.elu(self.hconv1(x, data.edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.hconv2(x, data.edge_index)
        return F.log_softmax(x, dim=1)

if __name__ == '__main__':
    
    #Dataset CORA
    data = (Planetoid(root="Cora", name="Cora"))[0]

    def train_model():
        model.train()
        optimizer.zero_grad()  
        out = model(data)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

    def test_model():
        model.eval()
        logits, accs = model(data), []

        for _, mask in data('train_mask', 'val_mask', 'test_mask'):
            pred = logits[mask].max(1)[1]
            acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
            accs.append(acc)

        return accs

    model = HyperGraph(data=data)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3, weight_decay=5e-4)

    for epoch in range(100):
        train_model()
        train_, val_, test_ = test_model()

        log = 'Epoch: {:03d} , accuracy: Train: {:.54f} , Val: {:.5f} , Test: {:.5f}'
        print(log.format(epoch, train_, val_, test_))
