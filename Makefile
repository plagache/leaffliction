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

download:
	wget -N https://cdn.intra.42.fr/document/document/17547/leaves.zip

extract: download
	unzip leaves.zip
	mkdir -p images/Apple images/Grape
	# mkdir -p images/Apple/toto
	# mkdir -p images/Apple/tato/tota/
	# mv images/Apple_Black_rot images/Apple/toto/
	# mv images/Apple_rust images/Apple/tato/tota/
	mv images/Apple_* images/Apple/
	mv images/Grape_* images/Grape/

debug_directory:
	mkdir -p debug


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

get_dataset:
	${PYTHON} helpers.py

distribution:
	${PYTHON} Distribution.py images
	# ${PYTHON} Distribution.py augmented_directory
	# ${PYTHON} Distribution.py images/Apple
	# ${PYTHON} Distribution.py images/Grape

augmentation:
	# ${PYTHON} Augmentation.py "images/Apple_healthy/image (42).JPG"
	# ${PYTHON} Augmentation.py "images/Apple_healthy/image (9).JPG"
	# ${PYTHON} Augmentation.py images/Apple_healthy
	# ${PYTHON} Augmentation.py images/Apple
	${PYTHON} Augmentation.py images

transformation: debug_directory
	# ${PYTHON} Transformation.py -src images -dst debug
	# ${PYTHON} Transformation.py -src images/Apple_Black_rot -dst debug
	${PYTHON} Transformation.py "images/Grape_Black_rot/image (1).JPG"
	# ${PYTHON} Transformation.py "images/Grape_Black_rot/image (1)_Rotate.JPG"
	# ${PYTHON} Transformation.py "images/Grape_Black_rot/image (9).JPG"
	# ${PYTHON} Transformation.py "images/Apple_Black_rot/image (33).JPG"
	# ${PYTHON} Transformation.py "images/Grape_Black_rot/image (33).JPG"
	# ${PYTHON} Transformation.py "images/Apple_healthy/image (9).JPG"
	# These usages are incorrect and should throw error
	# ${PYTHON} Transformation.py -src images -dst
	# ${PYTHON} Transformation.py -src -dst debug
	# ${PYTHON} Transformation.py -src images -dst debug "images/Apple/Apple_healthy/image (9).JPG"
	# ${PYTHON} Transformation.py debug "images/Apple_healthy/image (9).JPG"

sample: augmentation
	${PYTHON} sample.py augmented_directory

train:
	# ${PYTHON} train.py
	${PYTHON} train.py images/Apple
	# ${PYTHON} train.py images

predict:
	# ${PYTHON} predict.py
	${PYTHON} predict.py "images/Grape_Black_rot/image (1).JPG"

gradio:
	${BIN}/gradio web_interface.py

viz:
	VIZ=1 ${PYTHON} train_.py train validation

resnet:
	${PYTHON} resnet.py

alexnet:
	${PYTHON} alex_torch.py

fast_inference:
	${PYTHON} fast_inference.py

fine:
	${PYTHON} fine_tune.py train validation

reaugmentation: clean extract augmentation

test:
	${PYTHON} test_utils.py -v

clean:
	rm -rf debug*
	rm -rf augmented*
	rm -rf images*
	rm -rf train/
	rm -rf validation/

fclean: clean
	rm -rf ${VENV}
	rm -rf activate
	rm -rf "leaves.zip"

re: fclean setup run


#------------------------------------------------#
#   SPEC                                         #
#------------------------------------------------#
.SILENT:
.PHONY: setup venv uv_upgrade pip_upgrade install module requirements list version size run clean fclean re download train
