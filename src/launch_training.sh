echo script
echo $HOSTNAME
echo $PATH
th main.lua -expID synth-30000-batch-6 -trainBatch 6 -trainIters 5000 -validBatch 6 -validIters 250 -nEpochs 100 -saveHeatmaps -saveInput -finalPredictions -dataDir /sequoia/data1/yhasson/hourglass-hands/data/annots


