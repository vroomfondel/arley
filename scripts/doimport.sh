#!/bin/bash

#exec 3>&1 4>&2 >/tmp/bootlog.txt 2>&1
#set -x

#logfile=arley_import_$(date +%Y%m%d_%H%M%S).log

SCRIPTDIR=$(dirname "$0")
source "${SCRIPTDIR}/include.sh"

function importdir() {
	medir="${1}"
	date;
  echo -e "\n\n\n###########################\n";
  echo "medir: ${medir}" ;
  PYTHONPATH=$PYTHONPATH:$(pwd) python3 arley/vectorstore/importhelper.py convert "${medir}"
  return $?
}

for i in "${DOCUMENT_DIRS[@]}" ; do
	if ! [ -d "${i}" ] ; then
		echo SKIP: $i
		continue
	fi
	if [ -e "${i}/DONE" ] ; then
	  echo ALREADY DONE: "${i}"
	  continue
	else
	  echo "NOT DONE: ${i}"
	fi

	logfile="${i}/arley_import_$(date +%Y%m%d_%H%M%S).log"

	importdir "${i}" 2>&1 | tee -a "${logfile}"
	ret=${PIPESTATUS[0]}
	if [ ${ret} -eq 0 ] ; then
	  touch "${i}/DONE"
	else
	  echo FAIL="${ret}" IN "${i}"
	fi
done 

#exec 1>&3 2>&4

