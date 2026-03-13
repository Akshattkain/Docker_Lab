import torch
import torch.nn as nn
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from ucimlrepo import fetch_ucirepo
import pickle

class BanknoteModel(nn.Module):
    def __init__(self):
        super(BanknoteModel, self).__init__()
        self.layer1 = nn.Linear(4, 16)
        self.layer2 = nn.Linear(16, 8)
        self.layer3 = nn.Linear(8, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.sigmoid(self.layer3(x))
        return x

if __name__ == '__main__':
    # Fetch banknote authentication dataset from UCI
    banknote = fetch_ucirepo(id=267)
    X = banknote.data.features.values
    y = banknote.data.targets.values.ravel()
    
    # Split and standardize
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Save scaler
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    # Convert to tensors
    X_train = torch.FloatTensor(X_train)
    y_train = torch.FloatTensor(y_train).unsqueeze(1)
    X_test = torch.FloatTensor(X_test)
    y_test = torch.FloatTensor(y_test).unsqueeze(1)
    
    # Train model
    model = BanknoteModel()
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    for epoch in range(100):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 20 == 0:
            with torch.no_grad():
                test_outputs = model(X_test)
                predicted = (test_outputs > 0.5).float()
                accuracy = (predicted == y_test).sum().item() / len(y_test)
                print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}, Test Acc: {accuracy:.4f}")
    
    # Save model
    torch.save(model.state_dict(), 'banknote_model.pth')
    print("Model trained and saved!")