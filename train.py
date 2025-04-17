import argparse
import timeit

from tinygrad import Context, Device, GlobalCounters, Tensor, TinyJit, nn
from tinygrad.nn.datasets import mnist
from resnet import ResNet

from resnet import ResNet

from utils import DatasetFolder, Dataloader
import numpy as np

# watch -n0.1 nvidia-smi

# gcc13-13.3.0-1  gcc13-libs-13.3.0-1  opencl-nvidia-565.57.01-1  cuda-12.6.2-2

print(Device.DEFAULT)

class BigModel:
    def __init__(self, num_classes):
        self.l1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=1)
        self.l2 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=1)
        self.l3 = nn.Conv2d(128, 256, kernel_size=(1, 1), stride=1)
        self.l4 = nn.Linear(20736, num_classes)

    def __call__(self, x: Tensor) -> Tensor:
        x = self.l1(x).relu().max_pool2d((3, 3))
        x = self.l2(x).relu().max_pool2d((3, 3))
        x = self.l3(x).relu().max_pool2d((3, 3))
        x = x.flatten(1).dropout(0.5)
        # print(x.shape)
        return self.l4(x)

class SmallModel:
  def __init__(self, num_of_classes):
    self.l1 = nn.Conv2d(3, 32, kernel_size=(3,3))
    self.l2 = nn.Conv2d(32, 64, kernel_size=(3,3))
    self.l3 = nn.Linear(64 * 62 * 62, num_of_classes)

  def __call__(self, x:Tensor) -> Tensor:
    x = self.l1(x).relu().max_pool2d((2,2))
    x = self.l2(x).relu().max_pool2d((2,2))
    return self.l3(x.flatten(1).dropout(0.5))


parser = argparse.ArgumentParser(description="Train our model on a dataset from a train and test directory")
parser.add_argument("train_directory", help="the train set directory to parse")
parser.add_argument("test_directory", help="the validation set directory to parse")
args = parser.parse_args()

train_folder: DatasetFolder = DatasetFolder(args.train_directory)
test_folder: DatasetFolder = DatasetFolder(args.test_directory)
# print(folder.mapped_dictionnary)
# print(folder.count_dictionnary)
# print(folder.indices_dictionnary)
loader: Dataloader = Dataloader(train_folder, 1000)
test_loader: Dataloader = Dataloader(test_folder, 100, shuffle=True)
X_train, Y_train = loader.get_tensor()
X_test, Y_test = test_loader.get_tensor()
# with np.printoptions(threshold=np.inf):
#     print(Y_test.numpy())
samples = Tensor.randint(10, high=X_test.shape[0])
X_test, Y_test = X_test[samples], Y_test[samples]
# X_test, Y_test = X_test[:100], Y_test[:100]
print(Y_test.numpy())
print(X_test.shape, X_test.dtype)
# print(X_train[0].numpy())
# print(X_train, Y_train, X_test, Y_test)
# print(X_train.dtype, Y_train.dtype)
# print(X_test.dtype, Y_test.dtype)
# print(folder[0])
# (60000, 1, 28, 28) dtypes.uchar (60000,) dtypes.uchar

model = ResNet(18, len(train_folder.classes))
model.load_from_pretrained()
acc = (model(X_test).argmax(axis=1) == Y_test).mean()
# NOTE: tinygrad is lazy, and hasn't actually run anything by this point
print(acc.item())  # ~10% accuracy, as expected from a random model
# exit(0)

optim = nn.optim.Adam(nn.state.get_parameters(model))
batch_size = 1


def t_step():
    Tensor.training = True  # makes dropout work
    samples = Tensor.randint(batch_size, high=X_train.shape[0])
    # print(samples)
    X, Y = X_train[samples], Y_train[samples]
    # print(Y.numpy())
    # print(X.shape)
    optim.zero_grad()
    loss = model(X).sparse_categorical_crossentropy(Y).backward()
    optim.step()
    return loss

timeit.repeat(t_step, repeat=5, number=1)

GlobalCounters.reset()
with Context(DEBUG=2): t_step()

jit_step = TinyJit(t_step)
timeit.repeat(jit_step, repeat=5, number=1)

# for step in range(500):
#     loss = t_step()
#     if step % 100 == 0:
#         Tensor.training = False
#         model_prediction = model(X_test).argmax(axis=1)
#         acc = (model_prediction == Y_test).mean().item()
#         if acc >= 0.9:
#             break
#         print(f"step {step:4d}, loss {loss.item():.2f}, acc {acc*100.:.2f}%")
# print(f"step {step:4d}, loss {loss.item():.2f}, acc {acc*100.:.2f}%")
# print(loss.dtype)
