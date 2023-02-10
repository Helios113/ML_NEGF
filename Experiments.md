## Exp 1
loss_function = torch.nn.MSELoss()  
optimizer = torch.optim.Adam(model.parameters(),
                             lr=1e-4)

data: 0-1 for whole device

## Exp 2
loss_function = torch.nn.MSELoss()  
optimizer = torch.optim.Adam(model.parameters(),
                             lr=1e-3)

data: 0-1 for whole device


## Exp 3
loss_function = torch.nn.MSELoss()  
optimizer = torch.optim.Adam(model.parameters(),
                             lr=1e-3)

data: mean 0, 1 std
Tanh



## Exp 4
loss_function = torch.nn.MSELoss()  
optimizer = torch.optim.Adam(model.parameters(),
                             lr=1e-4)

data: mean 0, 1 std
Tanh

## Exp 4
loss_function = torch.nn.MSELoss()  
optimizer = torch.optim.Adam(model.parameters(),
                             lr=1e-4)

data: mean 0, 1 std
Relu



