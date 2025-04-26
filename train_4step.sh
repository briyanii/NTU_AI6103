#!/bin/bash

# STEPS
# train model, save model state
# load latest state from step1a, generate proposals
# train RPN head using proposals from step1b
# load state from step2, train RPN head only
# load state from step3, train detection head only

pre_post_check() {
	if [[ -n $3 && -e $3 ]]; then
		# if expected output file exists, skip step
		echo "IMPT: $3 exists, skipping $2"
	elif [[ -z $1 || -e $1 ]]; then
		# if expected input file exists, run step
		echo "IMPT: prereq $1 found, running $2"
		python $2
		if [[ -n $3 && -e $3 ]]; then
			echo "IMPT: $3 exists, completed $2"
		else
			echo "IMPT: $3 not found, incomplete $2"
			exit 2
		fi
	else
		echo "IMPT: $1 not found, aborting $2"
		exit 1
	fi
}

f1=outputs/checkpoint_step1_80000.pt
f2=outputs/roi_proposals.pkl
f3=outputs/checkpoint_step2_80000.pt
f4=outputs/checkpoint_step3_80000.pt
f5=outputs/checkpoint_step4_80000.pt

mkdir -p outputs
pre_post_check "" step1a.py $f1
pre_post_check $f1 step1b.py $f2
pre_post_check $f2 step2.py $f3
pre_post_check $f3 step3.py $f4
pre_post_check $f4 step4.py $f5
