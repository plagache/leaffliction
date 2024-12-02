#------------------------------------------------#
#   VARIABLES                                    #
#------------------------------------------------#
PYTHON_VERSION=3.12
VENV=.venv
BIN=${VENV}/bin
PYTHON=${BIN}/python
ACTIVATE=${BIN}/activate

PROGRAM=

# ARGUMENTS=


#------------------------------------------------#
#   SETUP                                        #
#------------------------------------------------#
setup: venv pip_upgrade install

venv:
	uv venv --python ${PYTHON_VERSION} ${VENV} --seed
	ln -sf ${ACTIVATE} activate

uv_upgrade:
	uv self update

pip_upgrade:
	uv pip install --upgrade pip

install: \
	requirements \
	module \
#
requirements: requirements.txt
	uv pip install -r requirements.txt --upgrade

module: setup.py
	uv pip install -e . --upgrade

extract:
	wget https://cdn.intra.42.fr/document/document/17547/leaves.zip
	unzip leaves.zip
	mkdir -p images/Apple images/Grape
	mkdir -p images/Apple/toto
	mkdir -p images/Apple/tato/tota/
	mv images/Apple_Black_rot images/Apple/toto/
	mv images/Apple_rust images/Apple/tato/tota/
	mv images/Apple_* images/Apple/
	mv images/Grape_* images/Grape/

augmented_directory:
	mkdir -p augmented_directory

#------------------------------------------------#
#   INFO                                         #
#------------------------------------------------#
list:
	uv pip list

version:
	uv python list

size:
	du -hd 0
	du -hd 0 ${VENV}

nvidia:
	watch -n0.1 nvidia-smi


#------------------------------------------------#
#   RECIPES                                      #
#------------------------------------------------#
run:
	${PYTHON} ${PROGRAM} \
	# ${ARGUMENTS}

dataloaders:
	${PYTHON} helpers.py

distribution:
	${PYTHON} Distribution.py images
	${PYTHON} Distribution.py images/Apple
	# ${PYTHON} Distribution.py images/Grape

augmentation: augmented_directory
	${PYTHON} Augmentation.py

clean:
	rm -rf images/
	rm -rf "leaves.zip"

fclean: clean
	rm -rf ${VENV}
	rm -rf activate

re: fclean setup run


#------------------------------------------------#
#   SPEC                                         #
#------------------------------------------------#
.SILENT:
.PHONY: setup venv uv_upgrade pip_upgrade install module requirements list version size run clean fclean re
