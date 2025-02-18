import argparse
import timeit

from tinygrad import Context, Device, GlobalCounters, Tensor, TinyJit, nn
from tinygrad.nn.datasets import mnist

from utils import DatasetFolder, Dataloader

# watch -n0.1 nvidia-smi

# gcc13-13.3.0-1  gcc13-libs-13.3.0-1  opencl-nvidia-565.57.01-1  cuda-12.6.2-2

# print(Device.DEFAULT)


class Model:
    def __init__(self, num_classes):
        self.l1 = nn.Conv2d(3, 64, kernel_size=(3, 3), padding=1)
        self.l2 = nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1)
        self.l3 = nn.Conv2d(128, 256, kernel_size=(3, 3), padding=1)
        self.l4 = nn.Linear(256 * 32 * 32, num_classes)

    def __call__(self, x: Tensor) -> Tensor:
        x = self.l1(x).relu().max_pool2d((2, 2))
        x = self.l2(x).relu().max_pool2d((2, 2))
        x = self.l3(x).relu().max_pool2d((2, 2))
        return self.l4(x.flatten(1).dropout(0.5))


parser = argparse.ArgumentParser(description="analyse a dataset from a given directory")
parser.add_argument("directory", help="the directory to parse")
args = parser.parse_args()

folder: DatasetFolder = DatasetFolder(args.directory)
# print(folder.mapped_dictionnary)
# print(folder.count_dictionnary)
# print(folder.indices_dictionnary)
loader: Dataloader = Dataloader(folder, 1000)
X_train, Y_train = loader.get_tensor()
# exit(0)
X_test, Y_test = X_train, Y_train
# print(folder[0])
# (60000, 1, 28, 28) dtypes.uchar (60000,) dtypes.uchar

model = Model(len(folder.classes))
acc = (model(X_test).argmax(axis=1) == Y_test).mean()
# NOTE: tinygrad is lazy, and hasn't actually run anything by this point
print(acc.item())  # ~10% accuracy, as expected from a random model

optim = nn.optim.Adam(nn.state.get_parameters(model))
batch_size = 128


def step():
    Tensor.training = True  # makes dropout work
    samples = Tensor.randint(batch_size, high=X_train.shape[0])
    X, Y = X_train[samples], Y_train[samples]
    optim.zero_grad()
    loss = model(X).sparse_categorical_crossentropy(Y).backward()
    optim.step()
    return loss


# timeit.repeat(step, repeat=5, number=1)

GlobalCounters.reset()
with Context(DEBUG=2):
    step()

jit_step = TinyJit(step)

for step in range(300):
    loss = jit_step()
    if step % 100 == 0:
        Tensor.training = False
        acc = (model(X_test).argmax(axis=1) == Y_test).mean().item()
        print(f"step {step:4d}, loss {loss.item():.2f}, acc {acc*100.:.2f}%")

timeit.repeat(jit_step, repeat=5, number=1)
