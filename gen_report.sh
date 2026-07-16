#!/bin/bash


EXP_NAME="$1"
M=$2
#Line renders report: qmd -> jupyter -> latex -> pdf. -P flag passes config and generates the report based on the config parameters and results
quarto render report_template.qmd -P exp_name:"$EXP_NAME" -P M:$M -o "$EXP_NAME.pdf" 
echo "done"

