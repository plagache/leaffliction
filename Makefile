#------------------------------------------------#
#   VARIABLES                                    #
#------------------------------------------------#
PYTHON_VERSION=3.12
VENV=.venv
BIN=${VENV}/bin
PYTHON=${BIN}/python
ACTIVATE=${BIN}/activate


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
	# mkdir -p images/Apple images/Grape
	# mv images/Apple_* images/Apple/
	# mv images/Grape_* images/Grape/

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
distribution:
	# ${PYTHON} Distribution.py images
	${PYTHON} Distribution.py augmented_directory
	# ${PYTHON} Distribution.py images/Apple
	# ${PYTHON} Distribution.py images/Grape

augmentation:
	# ${PYTHON} Augmentation.py "images/Apple_healthy/image (42).JPG"
	${PYTHON} Augmentation.py "images/Apple_healthy/image (9).JPG"
	# ${PYTHON} Augmentation.py images
	# ${PYTHON} Augmentation.py images/Apple_healthy
	# ${PYTHON} Augmentation.py images/Apple

transformation: debug_directory
	${PYTHON} Transformation.py "images/Grape_Black_rot/image (1).JPG"
	# ${PYTHON} Transformation.py "images/Apple_Black_rot/image (33).JPG"
	# ${PYTHON} Transformation.py -src images/Apple_Black_rot -dst debug

sample: augmentation
	${PYTHON} sample.py augmented_directory

train:
	${PYTHON} train.py images
	# ${PYTHON} train.py images/Apple
	# ${PYTHON} rain.py images/Grape

predict:
	${PYTHON} predict.py "images/Grape_Black_rot/image (1).JPG"
	# ${PYTHON} predict.py

gradio:
	${BIN}/gradio web_interface.py

clean:
	rm -rf debug*
	rm -rf images
	rm -rf *_dataset
	rm -rf augmented_directory

fclean: clean
	rm -rf ${VENV}
	rm -rf activate
	rm -rf "leaves.zip"

re: fclean setup run


#------------------------------------------------#
#   SPEC                                         #
#------------------------------------------------#
.SILENT:
.PHONY: setup venv uv_upgrade pip_upgrade install requirements module download extract debug_directory list version size nvidia distribution augmentation transformation sample train predict gradio clean fclean re
