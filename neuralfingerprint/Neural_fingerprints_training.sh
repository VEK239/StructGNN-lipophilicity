#!/bin/bash

i=4
for l in -3 -2
do
	for p in -4 -3 -2 -1
	do
		for c in 10 20 30
		do
			for d in 3 4 6
			do
				for f in 20 30 40 50 60 70 80
				do
					for n in logp_mean logP_wo_parameters
					do
						python training_logs.py -n $i -f $f -d $d -c $c -s $p -l $l -t $n
						i=$(($i+1))
						echo $i
					done
				done
			done
		done
	done
done
