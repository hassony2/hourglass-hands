echo script
echo $HOSTNAME
echo $PATH
th main.lua -expID test-run -trainBatch 2 -trainIters 4000 -validBatch 2 -validIters 1000 -nEpochs 100 -saveHeatmaps -saveInput -finalPredictions -dataDir /sequoia/data1/yhasson/hourglass-hands/data/annots


