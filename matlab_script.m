% Script to transform the unreadible numbers from the EEG data into a table
% with the extracted features per epoch
% MAKE SURE TO CHANGE FILEPATH AND NAME

eeglab;
basePath = 'C:\\Users\\setoain\\OneDrive - Universitat de Barcelona\\Escritorio\\EEGData\\sub-xx\\ses-xx\\';
EEG = pop_loadset('filename', 'sub-01_ses-01_task-rest_eeg.set', ...
                  'filepath', basePath);
EEG = eeg_checkset(EEG);
nChannels = EEG.nbchan;
fs = EEG.srate;
totalPoints = EEG.pnts;
totalTime = totalPoints / fs;

% change epochs if necessary
epochLength = 5;        
stepLength = 2.1;       
nEpochs = floor((totalTime - epochLength) / stepLength) + 1;

EEG.event = [];
for i = 1:nEpochs
    EEG.event(i).type = 'SEG';
    EEG.event(i).latency = (i-1) * stepLength * fs + 1;
    EEG.event(i).duration = 0;
end
EEG = eeg_checkset(EEG, 'eventconsistency');
EEG = pop_epoch(EEG, {'SEG'}, [0 epochLength], 'epochinfo', 'yes');

bandDefs = [0.5 4; 4 8; 8 12; 12 30; 30 100];
bandNames = {"Delta", "Theta", "Alpha", "Beta", "Gamma"};
nBands = size(bandDefs,1);
filtCoeffs = struct();
for b = 1:nBands
    Wp = [bandDefs(b,1) bandDefs(b,2)] / (fs/2);
    [filtCoeffs(b).B, filtCoeffs(b).A] = butter(4, Wp);
end

contEEG = pop_loadset('filename', 'sub-01_ses-01_task-rest_eeg.set', ...
                      'filepath', basePath);
contEEG = eeg_checkset(contEEG);
contData = contEEG.data;
chanMean = mean(contData, 2);
chanStd  = std(double(contData'), 0, 1)';
burstThresh = chanMean + 2*chanStd;

nFeatures = 19 * nChannels + 13;
featuresMatrix = zeros(nFeatures, EEG.trials);
featureNames = cell(nFeatures, 1);
connPairs = { 'F3','F4'; 'C3','C4'; 'O1','O2'; 'T7','T8'; 'Fz','Pz'; 'P3','P4' };

for ep = 1:EEG.trials
    epochData = double(EEG.data(:,:,ep));
    nPoints = size(epochData, 2);
    epochData_TxC = epochData';
    featIndex = 1;
    alphaData = [];
    for b = 1:nBands
        B = filtCoeffs(b).B; A = filtCoeffs(b).A;
        band_signal = filtfilt(B, A, epochData_TxC);
        maxAmp = max(band_signal);
        minAmp = min(band_signal);
        meanAbsAmp = mean(abs(band_signal));
        for ch = 1:nChannels
            featuresMatrix(featIndex, ep) = maxAmp(ch);
            featIndex = featIndex + 1;
        end
        for ch = 1:nChannels
            featuresMatrix(featIndex, ep) = minAmp(ch);
            featIndex = featIndex + 1;
        end
        for ch = 1:nChannels
            featuresMatrix(featIndex, ep) = meanAbsAmp(ch);
            featIndex = featIndex + 1;
        end
        if strcmpi(bandNames{b}, 'Alpha')
            alphaData = band_signal;
        end
    end
    chanLabels = {EEG.chanlocs.labels};
    if isempty(alphaData)
        alphaData = filtfilt(filtCoeffs(3).B, filtCoeffs(3).A, epochData_TxC);
    end
    for p = 1:size(connPairs,1)
        chName1 = connPairs{p,1};
        chName2 = connPairs{p,2};
        chIdx1 = find(strcmpi(chanLabels, chName1));
        chIdx2 = find(strcmpi(chanLabels, chName2));
        if isempty(chIdx1) || isempty(chIdx2)
            PLV = NaN; AEC_val = NaN;
        else
            sig1 = alphaData(:, chIdx1);
            sig2 = alphaData(:, chIdx2);
            analytic1 = hilbert(sig1);
            analytic2 = hilbert(sig2);
            phaseDiff = angle(analytic1) - angle(analytic2);
            PLV = abs(mean(exp(1i * phaseDiff)));
            env1 = abs(analytic1);
            env2 = abs(analytic2);
            R = corrcoef(env1, env2);
            AEC_val = R(1,2);
        end
        featuresMatrix(featIndex, ep) = PLV;
        featIndex = featIndex + 1;
        featuresMatrix(featIndex, ep) = AEC_val;
        featIndex = featIndex + 1;
    end
    [Pxx, f] = periodogram(epochData_TxC, [], [], fs);
    fFitIdx = find(f >= 2 & f <= 100);
    for ch = 1:nChannels
        Pfreq = Pxx(fFitIdx, ch);
        fvals = f(fFitIdx);
        safeIdx = Pfreq > 0;
        if sum(safeIdx) > 1
            coeffs = polyfit(log10(fvals(safeIdx)), log10(Pfreq(safeIdx)), 1);
            slope = coeffs(1);
        else
            slope = NaN;
        end
        featuresMatrix(featIndex, ep) = slope;
        featIndex = featIndex + 1;
    end
    for ch = 1:nChannels
        sig = epochData(ch, :);
        thresh = burstThresh(ch);
        aboveThresh = abs(sig) > thresh;
        if any(aboveThresh)
            padded = [0, aboveThresh, 0];
            diffMask = diff(padded);
            burstStarts = find(diffMask == 1);
            burstEnds = find(diffMask == -1) - 1;
            nBursts = length(burstStarts);
            durations = (burstEnds - burstStarts + 1) / fs;
            avgDur = mean(durations);
        else
            nBursts = 0;
            avgDur = 0;
        end
        featuresMatrix(featIndex, ep) = nBursts;
        featIndex = featIndex + 1;
        featuresMatrix(featIndex, ep) = avgDur;
        featIndex = featIndex + 1;
    end
    gfp = std(epochData, 0, 1);
    meanGFP = mean(gfp);
    featuresMatrix(featIndex, ep) = meanGFP;
    featIndex = featIndex + 1;
    freqIdx = find(f >= 0.5 & f <= 100);
    for ch = 1:nChannels
        [~, maxI] = max(Pxx(freqIdx, ch));
        peakFreq = f(freqIdx(maxI));
        featuresMatrix(featIndex, ep) = peakFreq;
        featIndex = featIndex + 1;
    end
end

chanLabels = {EEG.chanlocs.labels};
featIndex = 1;
featureNames = cell(19 * nChannels + 13, 1);
for b = 1:length(bandNames)
    band = bandNames{b};
    for ch = 1:nChannels
        featureNames{featIndex} = sprintf('Max %s amp at %s', band, chanLabels{ch});
        featIndex = featIndex + 1;
    end
    for ch = 1:nChannels
        featureNames{featIndex} = sprintf('Min %s amp at %s', band, chanLabels{ch});
        featIndex = featIndex + 1;
    end
    for ch = 1:nChannels
        featureNames{featIndex} = sprintf('Mean %s amp at %s', band, chanLabels{ch});
        featIndex = featIndex + 1;
    end
end
for ch = 1:nChannels
    featureNames{featIndex} = sprintf('1/f slope at %s', chanLabels{ch});
    featIndex = featIndex + 1;
end
for ch = 1:nChannels
    featureNames{featIndex} = sprintf('Burst count at %s', chanLabels{ch});
    featIndex = featIndex + 1;
    featureNames{featIndex} = sprintf('Burst duration at %s', chanLabels{ch});
    featIndex = featIndex + 1;
end
for ch = 1:nChannels
    featureNames{featIndex} = sprintf('Peak freq at %s', chanLabels{ch});
    featIndex = featIndex + 1;
end
for p = 1:size(connPairs,1)
    ch1 = connPairs{p,1};
    ch2 = connPairs{p,2};
    featureNames{featIndex} = sprintf('PLV (alpha) %s-%s', ch1, ch2);
    featIndex = featIndex + 1;
    featureNames{featIndex} = sprintf('AEC (alpha) %s-%s', ch1, ch2);
    featIndex = featIndex + 1;
end
featureNames{featIndex} = 'Mean Global Field Power';

T = table;
T.Feature = featureNames;
for i = 1:size(featuresMatrix, 2)
    T.(['Epoch_' num2str(i)]) = featuresMatrix(:, i);
end
writetable(T, fullfile(basePath, 'eeg_features_table.csv'));
