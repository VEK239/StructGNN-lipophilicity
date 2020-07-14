#!/bin/bash
# starting number of experiment
i=60
# learning rate
for l in -3 -2
do
    # l2 penalty
	for p in -4 -3 -2 -1
	do
        # number of convolutions
		for c in 10 20 30
		do
            # fingerprint depth (radius)
			for d in 3 4 6
			do
                # fingerprint length
				for f in 32 64 128
				do
                    # dataset name
					for n in logp_mean logP_wo_parameters
					do
						python training_with_logs.py -n $i -f $f -d $d -c $c -s $p -l $l -t $n -b 500 -e 50
						i=$(($i+1))
						echo $i
					done
				done
			done
		done
	done
done
