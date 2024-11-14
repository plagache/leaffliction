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
	unzip -uo leaves.zip

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


#------------------------------------------------#
#   RECIPES                                      #
#------------------------------------------------#
run:
	${PYTHON} ${PROGRAM} \
	# ${ARGUMENTS}

distribution: extract
	${PYTHON} Distribution.py images

clean:
	rm -rf images/

fclean: clean
	rm -rf ${VENV}
	rm -rf activate

re: fclean setup run


#------------------------------------------------#
#   SPEC                                         #
#------------------------------------------------#
.SILENT:
.PHONY: setup venv uv_upgrade pip_upgrade install module requirements list version size run clean fclean re
