#!/bin/bash

for func in  "count" "sum" "avg"   
do	
	for num in {1..10}
	do
	       touch  $func${num}".txt"
	done
done
touch stats.txt

