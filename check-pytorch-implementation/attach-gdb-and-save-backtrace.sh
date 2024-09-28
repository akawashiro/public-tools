#! /bin/bash

set -eux -o pipefail
PID=$(ps aux | grep victim.py | grep -v grep | awk '{print $2}')
TMPFILE=$(mktemp)
LOGFILE=$(mktemp)

cat <<EOF > ${TMPFILE}
set pagination off
set logging file ${LOGFILE}
set logging on
backtrace
rbreak scatter_value
commands
backtrace
continue
end
EOF

gdb -p ${PID} -x ${TMPFILE}
echo Check ${LOGFILE}
