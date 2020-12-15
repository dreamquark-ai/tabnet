#!/bin/bash
set -e

SCRIPT_DIR=$(dirname ${BASH_SOURCE[0]})
DOC_FOLDER_NAME="source/generated_docs"
DOC_FOLDER="${SCRIPT_DIR}/${DOC_FOLDER_NAME}"
TARGET_FOLDER="../pytorch_tabnet"

rm -rf ${DOC_FOLDER}
mkdir -p ${DOC_FOLDER}

MODULES_LIST=$(find ${SCRIPT_DIR}/${TARGET_FOLDER} -type f 2>/dev/null | grep -v \.cache | grep -v "/tests" |  grep -e '.py$' | sed -e 's/^[\.+/]*\///g' | sed -e 's/\.py$//g' | sed -e 's/\//./g') 
PACKAGE_LIST=$(echo ${MODULES_LIST} | tr ' ' '\n' | sed -e 's/^\(.*\)\.\(.*\)/\1/g' | sort | uniq)

#echo $(seq -s= $(($(echo pytorch_tabnet.sparsemax|wc -c) + 9))|tr -d '[:digit:]')

echo ${PACKAGE_LIST} | tr ' ' '\n' | xargs -n1 -I {} echo "cp ${SCRIPT_DIR}/package_template.tpl ${DOC_FOLDER}/{}.rst" | sh
echo ${PACKAGE_LIST} | tr ' ' '\n' | xargs -n1 -I {} echo "sed -si \"s/#PACKAGE_NAME#/{} package\n\$(seq -s= \$((\$(echo {}|wc -c) + 8))|tr -d '[:digit:]')/g\" ${DOC_FOLDER}/{}.rst" | sh

# module
echo ${MODULES_LIST} |\
 tr ' ' '\n' |\
 xargs -n1 -I {} echo "cat ${SCRIPT_DIR}/module_template.tpl | sed -s \"s/#MODULE_NAME_TITLE#/{} module\n\$(seq -s. \$((\$(echo {}|wc -c) + 9))|tr -d '[:digit:]')/g\" | sed -s \"s/#MODULE_NAME#/{}/g\" >> \$(echo ${DOC_FOLDER}/{} | sed -e 's/^\(.*\)\.\(.*\)/\1.rst/g')" | sh
cp "${SCRIPT_DIR}/../README.md" "${DOC_FOLDER}/README.md"