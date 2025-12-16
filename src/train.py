import torch 
import torch.nn as nn
import torch.optim as optim
import mlflow 
import os
from tqdm import tqdm

class RNN(nn.Module) : 
    def __init__(self,vocab_size , embed_dim,hidden_dim) :
        super().__init__()
        self.embedding = nn.Embedding(vocab_size , embed_dim , padding_idx = 0)
        self.rnn = nn.RNN(embed_dim , hidden_dim , batch_first = True)
        self.fc = nn.Linear(hidden_dim , 1)
    def forward(self,x) :
        x = self.embedding(x)
        out , _ = self.rnn(x)
        out = self.fc(out[:,-1,:])
        return out

def  train ():
    data = torch.load("data/processed/train_data.pt")
    X_train , y_train = data['X'] , data['y'].unsqueeze(1).float()
    
    vocab_size = 50
    embed_dim = 16
    hidden_dim = 32
    lr = 0.001
    epochs = 100

    mlflow.set_experiment("Sentiment_Pipeline_project")

    with mlflow.start_run():
        mlflow.log_params({"lr" : lr , "hidden_dim" : hidden_dim , "epoch" : epochs})

        model = RNN(vocab_size,embed_dim ,hidden_dim)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(),lr=lr)

        print("Trainig has been started !!")
        for epoch in tqdm(range(epochs)):
            optimizer.zero_grad()
            outputs = model(X_train)
            loss = criterion(outputs,y_train)
            loss.backward()
            optimizer.step()

            if epoch % 10 == 0 :
                print(f"Epoch {epoch} , Loss : {loss.item()}")
                mlflow.log_metric("loss",loss.item(),step = epoch)
        
        os.makedirs("models" , exist_ok = True)
        torch.save(model.state_dict(),"models/sentiment_model.pth")
        mlflow.log_artifact("models/sentiment_model.pth")
        print("Training done. Model saved.")
if __name__ == "__main__":
    train()