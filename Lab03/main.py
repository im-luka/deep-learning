from cProfile import label
from turtle import color
import torch
import math
import random
import matplotlib.pyplot as plt

dtype = torch.float
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

minimum = random.uniform(1,2)
maximum = random.uniform(1,2)

print(minimum, maximum)

x = torch.linspace(-minimum, maximum, 2000, device=device, dtype=dtype)
y = torch.sin(x) * torch.cos(x)

plt.plot(x.cpu(), y.cpu(), label="y=sin(x) * cos(x)", color="#0000ff")
plt.title("MAIN GRAPH")
plt.xlabel('x', color="#000", fontsize=15, fontweight="bold")
plt.ylabel('y', color="#000", fontsize=15, fontweight="bold")
plt.legend(loc="upper right")
plt.grid()
plt.show()

p = torch.tensor([1, 2, 3], device=device)
xx = x.unsqueeze(-1).pow(p)

model = torch.nn.Sequential(
  torch.nn.Linear(3, 1),
  torch.nn.Flatten(0, 1)
)

loss_fn = torch.nn.MSELoss(reduction="sum")

if device == "cuda":
  model = model.cuda()
  loss_fn = loss_fn.cuda()

learning_rate = 1e-5

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# LETS START TRAINING
for t in range(5001):
  y_pred = model(xx)

  loss = loss_fn(y_pred, y)
  if t % 500 == 0:
    print(f"Current loss on {t} iteration: {loss}")

  # Reset gradients
  optimizer.zero_grad()

  # Calculate gradients
  loss.backward()

  # Adjust values based on gradients
  optimizer.step()

  # Lets see how our trained graph looks like
  linear_layer = model[0]

  a = linear_layer.bias.item()
  b = linear_layer.weight[:, 0].item()
  c = linear_layer.weight[:, 1].item()
  d = linear_layer.weight[:, 2].item()
  # e = linear_layer.weight[:, 3].item()
  # f = linear_layer.weight[:, 4].item()

  if t % 500 == 0:
    y_graph = a + b*x + c*x**2 + d*x**3
    plt.plot(x.cpu().detach(), y.cpu().detach(), label="main graph", color="#000")
    plt.plot(x.cpu().detach(), y_graph.cpu().detach(), label="training graph", color="#ff0000")
    plt.title("Calculation")
    plt.xlabel('x', color="#000", fontsize=15, fontweight="bold")
    plt.ylabel('y', color="#000", fontsize=15, fontweight="bold")
    plt.legend(loc="upper right")
    plt.grid()
    plt.show()

print(f"Result: y = {a} + {b}x + {c}x^2 + {d}x^3")