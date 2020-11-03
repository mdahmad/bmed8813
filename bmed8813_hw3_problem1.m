% HW3 problem1.m 

function problem1
    
    %% PART A: data
    
    if ~exist('PhysionetData.mat','file')
        addpath(fullfile(matlabroot,'Examples','deeplearning_shared','main'));
        ReadPhysionetData;
        clear;
    end
    load PhysionetData Signals Labels
    
    disp('1A: Class Sizes');
    summary(Labels);
    
    % % the hard way:
    %
    % ulabels = unique(Labels);
    % counts = zeros(size(ulabels));
    % for i = 1:length(ulabels)
    %     counts(i) = sum(Labels == ulabels(i));
    %     fprintf('%s %i\n',ulabels(i),counts(i));
    % end
    
    %% PART B: filter out any signals with length != 9000 samples
    
    mask = cellfun(@(x)numel(x)~=9000, Signals);
    Labels(mask) = [];
    Signals(mask) = [];
    
    disp('1B: Quality-Controlled Class Sizes');
    summary(Labels);
    
    %% PART C: divide the samples into test/train sets
    
    
    [trainA,validateA,~] = dividerand(sum(Labels=='A'),0.8,0.2,0);
    [trainN,validateN,~] = dividerand(sum(Labels=='N'),0.8,0.2,0);
    
    % balance classes, where N is larger, by downsampling N to match A
    
    trainN = trainN(randperm(length(trainN),length(trainA)));
    validateN = validateN(randperm(length(validateN),length(validateA)));
    
    
    Aindex = find(Labels=='A');
    Nindex = find(Labels=='N');
    
    trainmask = [Aindex(trainA); Nindex(trainN)];
    validmask = [Aindex(validateA); Nindex(validateN)];
    
    disp('1C: Training Data');
    summary(Labels(trainmask));
    
    disp('1C: Validation Data');
    summary(Labels(validmask));
    
    %% PART D: design and train LSTM network
    
    layers = [
        sequenceInputLayer(1)
        bilstmLayer(100,'OutputMode','last')
        fullyConnectedLayer(2)
        softmaxLayer
        classificationLayer
        ];
    
    options = trainingOptions('adam', ...
        'MaxEpochs',10, ...
        'MiniBatchSize', 150, ...
        'InitialLearnRate', 0.01, ...
        'SequenceLength', 1000, ...
        'GradientThreshold', 1, ...
        'ExecutionEnvironment','auto',...
        'Verbose',true);
    
    [net,info] = trainNetwork(Signals(trainmask),Labels(trainmask),layers,options);
    
    % plot training accuracy
    
    figure;
    plot(info.TrainingAccuracy);
    ylim([0 100]);
    xlabel('Iteration');
    ylabel('Accuracy');
    title('1D: Training Accuracy');
    
    %% PART E: run validation data through the trained network
    
    predv = classify(net,Signals(validmask),'MiniBatchSize',32);
    accv = sum(predv==Labels(validmask))./length(validmask);
    
    disp('1E: validation accuracy');
    disp(accv);
    
    figure;
    plotconfusion(Labels(validmask), predv);
    title('1E: Confusion Matrix');
    
end
