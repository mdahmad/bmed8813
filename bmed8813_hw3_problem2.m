% HW3 problem1.m 

function problem2
    
    %% data and quality control from problem 1
    
    % download the physionet data if not already done, load it in
    if ~exist('PhysionetData.mat','file')
        addpath(fullfile(matlabroot,'Examples','deeplearning_shared','main'));
        ReadPhysionetData;
        clear;
    end
    load PhysionetData Signals Labels
    
    mask = cellfun(@(x)numel(x)~=9000, Signals);
    Labels(mask) = [];
    Signals(mask) = [];
    
    %% PART A: divide the samples into test/train sets
    
    % randomize into train/validate sets
    
    [trainA,validateA,~] = dividerand(sum(Labels=='A'),0.8,0.2,0);
    [trainN,validateN,~] = dividerand(sum(Labels=='N'),0.8,0.2,0);
    
    % balance classes, where N is larger, by repeating A samples to match N
    
    trainA = trainA(mod((1:length(trainN))-1,length(trainA))+1);
    validateA = validateA(mod((1:length(validateN))-1,length(validateA))+1);
    
    % use those random indices to split up the data
    
    Aindex = find(Labels=='A');
    Nindex = find(Labels=='N');
    
    trainmask = [Aindex(trainA); Nindex(trainN)];
    validmask = [Aindex(validateA); Nindex(validateN)];
    
    disp('2A: Training Data');
    summary(Labels(trainmask));
    
    disp('2A: Validation Data');
    summary(Labels(validmask));
    
    %% PART B: extract frequency-domain features, normalize
    
    % frequency-domain features

    function y = freq_feat(x)
        fs = 300; %Hz
        [p,f,t] = pspectrum(x,fs,'spectrogram','TimeResolution',0.5,'OverlapPercent',60);
        y = [instfreq(p,f,t), pentropy(p,f,t)];
    end
    
    Spectra = cellfun(@freq_feat, Signals, 'UniformOutput', false);
    
    % z-score normalization
    
    means = mean([Spectra{:}],2);
    stdvs = std([Spectra{:}],[],2);
    Spectra = cellfun(@(x)(x-means)./stdvs, Spectra, 'UniformOutput', false);
    
    %% PART B: plot extracted features
    
    Nsamp = randperm(length(Nindex),2);
    Asamp = randperm(length(Aindex),2);
    
    figure;
    
    subplot(2,2,1)
    plot(Spectra{Nindex(Nsamp(1))});
    title Normal
    subplot(2,2,3)
    plot(Spectra{Nindex(Nsamp(2))});
    title Normal
    
    subplot(2,2,2)
    plot(Spectra{Aindex(Asamp(1))});
    title AF
    subplot(2,2,4)
    plot(Spectra{Aindex(Asamp(2))});
    title AF
    
    sgtitle('2B: Time-Frequency Features')
    
    %% PART C: design and train LSTM network
    
    layers = [
        sequenceInputLayer(2)
        bilstmLayer(100,'OutputMode','last')
        fullyConnectedLayer(2)
        softmaxLayer
        classificationLayer
        ];
    
    options = trainingOptions('adam', ...
        'MaxEpochs',30, ...
        'MiniBatchSize', 150, ...
        'InitialLearnRate', 0.01, ...
        'GradientThreshold', 1, ...
        'ExecutionEnvironment','auto', ...
        'Verbose',true);
    
    [net,info] = trainNetwork(Spectra(trainmask),Labels(trainmask),layers,options);
    
    % plot training accuracy
    
    figure;
    plot(info.TrainingAccuracy);
    ylim([0 100]);
    xlabel('Iteration');
    ylabel('Accuracy');
    title('2C: Training Accuracy');
    
    %% PART D: run validation data through the trained network
    
    predv = classify(net,Spectra(validmask));
    accv = sum(predv==Labels(validmask))./length(validmask);
    
    disp('2D: validation accuracy');
    disp(accv);
    
    figure;
    plotconfusion(Labels(validmask), predv)
    title('2D: Confusion Matrix')
    
end