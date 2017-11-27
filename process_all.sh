#!/bin/bash
#for f in `ls /afs/cs.stanford.edu/u/jsteinhardt/data-poisoning-scratch/$1/$1_mean_attack*.mat`
for f in `ls /afs/cs.stanford.edu/u/jsteinhardt/data-poisoning-scratch/$1/$1_attack_clean-centroid*.npz | grep -v dists`
do
  # num_dots=`echo $(basename $f) | grep -o "\." | wc -l`
  # if [ "$num_dots" -gt "1" ]; then
    # continue;
  # fi
  echo $(basename $f)
  if [ -f ${f}_defense_report.html ]; then
    if [ $f -nt ${f}_defense_report.html ]; then      
      echo "Defense report for $f exists but appears to be stale, re-running..."
      python test_defenses.py $1 $(basename $f) --min_leverage 0.3
    else
      echo "Defense report already exists for $f and is up to date. Skipping this file."
    fi
  else
    echo "No defense report found for $f, so I'll generate one now."
    python test_defenses.py $1 $(basename $f) --min_leverage 0.3
  fi
done
