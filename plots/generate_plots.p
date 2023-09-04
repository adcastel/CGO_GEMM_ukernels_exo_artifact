
set terminal postscript color font "Helvetica, 16" enhanced 
#set key autotitle columnhead
set style data histograms
set style histogram cluster gap 3
set style fill solid
set boxwidth 3 

set grid ytics

set key top left box Left reverse width 2
set yrange [0:35]


set boxwidth 1

set title "Microkernel performance - Single core" font "Helvetica-Bold,22"


set output "figure13.eps"
set ylabel font ",22" "GFLOPS" 
set xlabel font ",22" "Microkernel dimensions (mr x nr)"
plot "../driver_microkernel/output/ukernel_results.dat" using 2:xtic(1) t "NEON" lc 6, \
      '' u 3 t "BLIS" lc 4, \
      '' u 4 t "EXO" lc 1, \


