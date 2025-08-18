# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

Explain the problem statement

## Neural Network Model

Include the neural network model diagram.

## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM
### Name: DINAGARAN JOHNY S
### Register Number: 212223220020
```python
class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1, 12)
        self.fc2 = nn.Linear(12, 10)
        self.fc3 = nn.Linear(10, 14)
        self.fc4 = nn.Linear(14, 1)
        self.relu = nn.ReLU()
        self.history={'loss':[]}

  def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x



# Initialize the Model, Loss Function, and Optimizer
johny=NeuralNet()
criterion = nn.MSELoss()
optimizer = optim.RMSprop(aaron.parameters(), lr=0.001)


def train_model(johny, X_train, y_train, criterion, optimizer, epochs=2000):
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = criterion(johny(X_train), y_train)
        loss.backward()
        optimizer.step()

        johny.history['loss'].append(loss.item())
        if epoch % 200 == 0:
            print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.6f}')


```
## Dataset Information
![WhatsApp Image 2025-08-18 at 23 10 53_d0097c86](https://github.com/user-attachments/assets/d17a32c2-8db7-473d-b104-ebdb6b7a3c63)


## OUTPUT

### Training Loss Vs Iteration Plot
![WhatsApp Image 2025-08-18 at 23 06 40_243a5818](https://github.com/user-attachments/assets/7e98f0f1-28df-4626-b3dd-6e90c33700c4)


### New Sample Data Prediction

![WhatsApp Image 2025-08-18 at 23 08 55_c685b2fb](https://github.com/user-attachments/assets/d9c89d49-9332-494a-9e69-c9badf690d75)

## RESULT
Thus the Neural Network Regression Model is developed, trained and tested Successfully.
