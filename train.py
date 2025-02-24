import argparse
import timeit

from tinygrad import Context, Device, GlobalCounters, Tensor, TinyJit, nn
from tinygrad.nn.datasets import mnist

from utils import DatasetFolder, Dataloader

# watch -n0.1 nvidia-smi

# gcc13-13.3.0-1  gcc13-libs-13.3.0-1  opencl-nvidia-565.57.01-1  cuda-12.6.2-2

print(Device.DEFAULT)

# class Model:
#     def __init__(self, num_classes):
#         self.l1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=1)
#         self.l2 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=1)
#         self.l3 = nn.Conv2d(128, 256, kernel_size=(1, 1), stride=1)
#         self.l4 = nn.Linear(20736, num_classes)
#
#     def __call__(self, x: Tensor) -> Tensor:
#         x = self.l1(x).relu().max_pool2d((3, 3))
#         x = self.l2(x).relu().max_pool2d((3, 3))
#         x = self.l3(x).relu().max_pool2d((3, 3))
#         x = x.flatten(1).dropout(0.5)
#         # print(x.shape)
#         return self.l4(x)

class Model:
  def __init__(self, num_of_classes):
    self.l1 = nn.Conv2d(3, 32, kernel_size=(3,3))
    self.l2 = nn.Conv2d(32, 64, kernel_size=(3,3))
    self.l3 = nn.Linear(64 * 62 * 62, num_of_classes)

  def __call__(self, x:Tensor) -> Tensor:
    x = self.l1(x).relu().max_pool2d((2,2))
    x = self.l2(x).relu().max_pool2d((2,2))
    return self.l3(x.flatten(1).dropout(0.5))


parser = argparse.ArgumentParser(description="analyse a dataset from a given directory")
parser.add_argument("directory", help="the directory to parse")
args = parser.parse_args()

folder: DatasetFolder = DatasetFolder(args.directory)
# print(folder.mapped_dictionnary)
# print(folder.count_dictionnary)
# print(folder.indices_dictionnary)
loader: Dataloader = Dataloader(folder, 1000)
X_train, Y_train, X_test, Y_test = loader.get_tensor()
# print(X_train[0].numpy())
# print(X_train, Y_train, X_test, Y_test)
# print(X_train.dtype, Y_train.dtype)
# print(folder[0])
# (60000, 1, 28, 28) dtypes.uchar (60000,) dtypes.uchar

model = Model(len(folder.classes))
acc = (model(X_test).argmax(axis=1) == Y_test).mean()
# NOTE: tinygrad is lazy, and hasn't actually run anything by this point
print(acc.item())  # ~10% accuracy, as expected from a random model

optim = nn.optim.Adam(nn.state.get_parameters(model))
batch_size = 16
# batch_size = 128


def t_step():
    Tensor.training = True  # makes dropout work
    samples = Tensor.randint(batch_size, high=X_train.shape[0])
    # print(samples)
    X, Y = X_train[samples], Y_train[samples]
    # print(Y_train[samples].numpy())
    optim.zero_grad()
    loss = model(X).sparse_categorical_crossentropy(Y).backward()
    optim.step()
    return loss

# import timeit
# timeit.repeat(step, repeat=5, number=1)

for step in range(500):
    loss = t_step()
    if step % 100 == 0:
        Tensor.training = False
        model_prediction = model(X_test).argmax(axis=1)
        print(model_prediction.numpy())
        print(Y_test.numpy())
        acc = (model_prediction == Y_test).mean().item()
        if acc >= 0.9:
            break
        print(f"step {step:4d}, loss {loss.item():.2f}, acc {acc*100.:.2f}%")
print(f"step {step:4d}, loss {loss.item():.2f}, acc {acc*100.:.2f}%")
# print(loss.dtype)
