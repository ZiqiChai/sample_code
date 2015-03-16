#!/bin/bash

#Trap non-normal exit signals: 1/HUP, 2/INT, 3/QUIT, 15/TERM, ERR
#trap onexit 1 2 3 15 ERR

#function onexit() {
#  local exit_status=$? #${1:-$?}
#  echo "Exiting $0 with error code: "$exit_status"."
#  echo "See log file for further details"
#  exit $exit_status
#}
source ~/.bashrc
cd /home/thomas/Documents/pairwise_incremental_reistration
