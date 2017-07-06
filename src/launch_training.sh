echo script
echo $HOSTNAME
echo $PATH
th main.lua -expID test-depth-run -trainBatch 6 -trainIters 4500 -validBatch 6 -validIters 250 -nEpochs 50 -saveHeatmaps -saveInput -finalPredictions -dataDir /sequoia/data1/yhasson/hourglass-hands/data/annots -useDepth


