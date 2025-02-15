# leaffliction
Image classification  by diseas recognition on leaves

## Installation and dependencies
You need python, [uv](https://github.com/astral-sh/uv?tab=readme-ov-file#uv), make
```bash
git clone https://github.com/plagache/leaffliction
cd leaffliction
make setup
```

## Usage

### Mandatory part

```sh
# Source and activate the environement
source activate
```

```sh
# Transform images
python Transformation.py -src images -dst debug
# python Transformation.py "images/Apple/Apple_healthy/image (9).JPG"
```

```bash
# display information about the nvidia graphic card in use
make nvidia

# dll and extract datasets
# make extract
make get_dateset

# display information about the distribution of the datasets
make distribution
```
