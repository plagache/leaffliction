import argparse
import timeit

from tinygrad import Context, Device, GlobalCounters, Tensor, TinyJit, nn
from tinygrad.nn.datasets import mnist

from utils import DatasetFolder

# watch -n0.1 nvidia-smi

# gcc13-13.3.0-1  gcc13-libs-13.3.0-1  opencl-nvidia-565.57.01-1  cuda-12.6.2-2

print(Device.DEFAULT)


class Model:
    def __init__(self):
        self.l1 = nn.Conv2d(1, 32, kernel_size=(3, 3))
        self.l2 = nn.Conv2d(32, 64, kernel_size=(3, 3))
        self.l3 = nn.Linear(1600, 10)

    def __call__(self, x: Tensor) -> Tensor:
        x = self.l1(x).relu().max_pool2d((2, 2))
        x = self.l2(x).relu().max_pool2d((2, 2))
        return self.l3(x.flatten(1).dropout(0.5))


X_train, Y_train, X_test, Y_test = mnist()
print(X_train.shape, X_train.dtype, Y_train.shape, Y_train.dtype)
print(X_train[0], Y_train[0])


parser = argparse.ArgumentParser(description="analyse a dataset from a given directory")
parser.add_argument("directory", help="the directory to parse")
args = parser.parse_args()

folder: DatasetFolder = DatasetFolder(args.directory)
folder.to_numpy()
print(folder[0])
exit()
# (60000, 1, 28, 28) dtypes.uchar (60000,) dtypes.uchar

model = Model()
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
