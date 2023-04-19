PACKAGE_DIR=speedypanther
TEST_DIR = tests

test:
	poetry run pytest -s ${TEST_DIR}/ -vvv

format:
	poetry run black ${PACKAGE_DIR} ${TEST_DIR}
	poetry run isort ${PACKAGE_DIR} ${TEST_DIR}

checkformat:
	poetry run black --check --diff ${PACKAGE_DIR} ${TEST_DIR}
	poetry run isort --check-only --diff ${PACKAGE_DIR} ${TEST_DIR}

lint:
	poetry run pylint ${PACKAGE_DIR} ${TEST_DIR}

jupyter:
	poetry run jupyter lab