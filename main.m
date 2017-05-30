%% test area
% load train data
load('trainData.mat');
% preprocess train data
trainVectors = trainVectors';
trainLabels = full(ind2vec(trainLabels'));
% generate noisy data
std = 0.2;
noiseData = [trainVectors + rand(size(trainVectors)) * (std / 2), ...
        trainVectors + rand(size(trainVectors)) * std, ...
        trainVectors + rand(size(trainVectors)) * (std * 2)];
noiseLables = [trainLabels, trainLabels, trainLabels];
perms = randperm(size(noiseData, 2));
noiseData = noiseData(:, perms);
noiseLables = noiseLables(:, perms);

% create new feedforwarding network
net = newff(trainVectors, trainLabels, [20], {'tansig','softmax'});

net.performFcn = 'crossentropy';
% net.performParam.regularization = 0.5;

net.trainFcn = 'trainscg';

net.divideParam.trainRatio = 0.6;
net.divideParam.valRatio = 0.2;
net.divideParam.testRatio = 0.2;
net.trainFcn = 'traingdm';
net.trainParam.epochs = 5000;
net.trainParam.goal = 1e-2;
net.trainParam.lr = 0.3;
net.trainParam.mc = 0.5;

net.plotFcns = {'plotperform', 'plottrainstate', 'ploterrhist', ...
        'plotconfusion', 'plotroc'};
% train on normal data
[net,tr] = train(net, trainVectors, trainLabels);
% train on noise data
[net,tr] = train(net, noiseData, noiseLables);
% retrain on normal data
[net,tr] = train(net, trainVectors, trainLabels);

% load test data
load('testData.mat');
% simulate network over test data
results = sim(net, testVectors');
% preprocess results
[mx, results] = max(results);
% export
export2csv(results);

%% cv test area
load('trainData.mat');

trainVectors = trainVectors';
trainLabels = int2labelVec(trainLabels');

[Xtr, ytr, Xv, yv] = cv_split(trainVectors, trainLabels, 0.2);

net = newff(Xtr, ytr, [20], {'tansig','softmax'});
net.trainFcn = 'trainscg';
net.trainParam.epochs = 1000;
net.trainParam.goal = 1e-3;
net = train(net, Xtr, ytr);


% validation
res = sim(net, Xv);
[mx, res] = max(res);
[mx, yv] = max(yv);
classperf(res, yv)


%% Iterate trough all functions
load('trainData.mat');

trainVectors = trainVectors';
trainLabels = int2labelVec(trainLabels');
% 'trainlm', 'trainbfg' <- hard tarin
% 'traingdm' <- found mvp 0.0413
train_functions = {'trainrp', 'trainscg', ...	
    'traincgb', 'traincgf', 'traincgp', 'trainoss', 'traingdx'};

for tf = train_functions
    disp(char(tf))
    net = newff(Xtr, ytr, [20], {'tansig','tansig','tansig'});
    net.trainFcn = char(tf);
    net.trainParam.goal = 1e-2;
    [net, tr] = train(net, Xtr, ytr);
    %plotperf(tr);
    disp('Evaluate your performance on function above');
    pause()
end


%% grid search traingdm

%% test area
load('trainData.mat');

trainVectors = trainVectors';
trainLabels = int2labelVec(trainLabels');
%trainLabels = de2bi(trainLabels)';
p1s = [0.05 5 500]*1.0e-5;
p2s = [0.05 5 500]*1.0e-7;
results = [];
for p1 = p1s
    for p2 = p2s
        disp([p1, p2])
        % create new feedforwarding network
        net = newff(trainVectors, trainLabels, [10], {'tansig','softmax'});
        net.performFcn = 'crossentropy';
        net.performParam.regularization = 0.5;
        net.trainFcn = 'trainscg';
        net.trainParam.sigma = p1;
        net.trainParam.lambda = p2;
        net.trainParam.showCommandLine = true;
        net.trainParam.showWindow = false;
        net.trainParam.show = 50;
        net.trainParam.epochs = 2000;
        % train on normal data
        [net,tr] = train(net, trainVectors, trainLabels);

        [net, tr] = train(net, trainVectors, trainLabels);
        results = [results;p1 p2 tr.best_perf tr.best_vperf tr.best_tperf tr.best_epoch];
    end
end

results(results(:,5) == min(results(:,5)), :)

%% MVP
load('trainData.mat');

trainVectors = trainVectors';
trainLabels = int2labelVec(trainLabels');

net = newff(trainVectors, trainLabels, [40], {'tansig','tansig','tansig'});
net.divideParam.trainRatio = 0.6;
net.divideParam.valRatio = 0.2;
net.divideParam.testRatio = 0.2;
net.trainFcn = 'traingdm';
net.trainParam.epochs = 5000;
net.trainParam.goal = 1e-2;
net.trainParam.lr = 0.3;
net.trainParam.mc = 0.5;
net.plotFcns = {'plotperform', 'plottrainstate', 'ploterrhist', ...
        'plotconfusion', 'plotroc'};
[net,tr] = train(net, trainVectors, trainLabels);

load('testData.mat');
results = sim(net, testVectors');
[mx, results] = max(results);

export2csv(results);

%% test area2
% load train data
load('trainData.mat');
% preprocess train data
trainVectors = trainVectors';
trainLabels = full(ind2vec(trainLabels'));

% generate noisy data
std = 0.2;
noiseData = [trainVectors + rand(size(trainVectors)) * (std / 2), ...
        trainVectors + rand(size(trainVectors)) * std, ...
        trainVectors + rand(size(trainVectors)) * (std * 2)];
noiseLables = [trainLabels, trainLabels, trainLabels];
perms = randperm(size(noiseData, 2));
noiseData = noiseData(:, perms);
noiseLables = noiseLables(:, perms);

% create new feedforwarding network
net = cascadeforwardnet(20);

% net.performFcn = 'crossentropy';
% % net.performParam.regularization = 0.5;
% 
% net.trainFcn = 'trainscg';
% 
% net.plotFcns = {'plotperform', 'plottrainstate', 'ploterrhist', ...
%         'plotconfusion', 'plotroc'};
% % train on normal data
[net,tr] = train(net, trainVectors, trainLabels);


%% testarea 3 - > regulatization, > std, > epochs (last matlab)
load('trainData.mat');
trainVectors = trainVectors';
trainLabels = trainLabels';

nets = cell(5,1);
perfs = cell(5,1);

for i=1:5
    oneVsAllLabels = double(trainLabels == i);
    positiveData = trainVectors(:, oneVsAllLabels == 1);
    oneVsAllVectors = [trainVectors, positiveData, positiveData, positiveData];
    oneVsAllLabels = [oneVsAllLabels, ones(1,size(positiveData, 2)*3)];

    
    % generate noisy data
    std = 0.8;
    noiseData = [oneVsAllVectors + rand(size(oneVsAllVectors)) * (std / 2), ...
            oneVsAllVectors + rand(size(oneVsAllVectors)) * std, ...
            oneVsAllVectors + rand(size(oneVsAllVectors)) * (std * 2)];
    noiseLables = [oneVsAllLabels, oneVsAllLabels, oneVsAllLabels];
    perms = randperm(size(noiseData, 2));
    noiseData = noiseData(:, perms);
    noiseLables = noiseLables(:, perms);
    
    oneVsAllVectors = [oneVsAllVectors, noiseData];
    oneVsAllLabels = [oneVsAllLabels, noiseLables];
    
    perms = randperm(size(oneVsAllVectors, 2));
    oneVsAllVectors = oneVsAllVectors(:, perms);
    oneVsAllLabels = oneVsAllLabels(:, perms);
    
    nets{i} = newff(oneVsAllVectors, oneVsAllLabels, [20], {'tansig','tansig'});
    
    nets{i}.performFcn = 'crossentropy';
    net.performParam.regularization = 0.8;

%     nets{i}.divideParam.trainRatio = 0.6;
%     nets{i}.divideParam.valRatio = 0.2;
%     nets{i}.divideParam.testRatio = 0.2;
    nets{i}.trainFcn = 'traingdm';
    nets{i}.trainParam.epochs = 7000;
    nets{i}.trainParam.goal = 1e-3;
    nets{i}.trainParam.max_fail = 10;
    nets{i}.trainParam.lr = 0.3;
    nets{i}.trainParam.mc = 0.5;
    nets{i}.trainParam.show = 500;
    nets{i}.trainParam.showWindow=0;
    nets{i}.trainParam.showCommandLine=1;
    nets{i}.plotFcns = {'plotperform', 'plottrainstate', 'ploterrhist', ...
            'plotconfusion', 'plotroc'};
    [nets{i},perfs{i}] = train(nets{i}, oneVsAllVectors, oneVsAllLabels);
end

[aux, results] = max([sim(nets{1}, trainVectors);
                        sim(nets{2}, trainVectors);
                        sim(nets{3}, trainVectors);
                        sim(nets{4}, trainVectors);
                        sim(nets{5}, trainVectors);]);
 
plotconfusion(full(ind2vec(trainLabels)), full(ind2vec(results)));                 

% load test data
load('testData.mat');
% simulate network over test data
[aux, results] = max([sim(nets{1}, testVectors');
                        sim(nets{2}, testVectors');
                        sim(nets{3}, testVectors');
                        sim(nets{4}, testVectors');
                        sim(nets{5}, testVectors');]);
% export
export2csv(results);


%% bootstrap
load('trainData.mat');

trainVectors = trainVectors';
trainLabels = int2labelVec(trainLabels');

%train validation test split
[trainInd,valInd,testInd] = dividerand(1:size(trainVectors,2), 0.6, 0.2, 0.2);
valVectors = trainVectors(:, valInd);
valLabels = trainLabels(:, valInd);
testVectors = trainVectors(:, testInd);
testLabels = trainLabels(:, testInd);
trainVectors = trainVectors(:, trainInd);
trainLabels = trainLabels(:, trainInd);

% cross validation values
nets_num_vals = [5, 10, 20, 50, 100, 200, 300, 500, 1000, 5000, 10000];
data_repetition_vals = [0.25, 0.5, 1, 2.5, 5, 10, 25, 50, 150];

% cross validation results
performance_obj.best_perf = 0;
performance_hist = cell(size(nets_num_vals, 2) * size(data_repetition_vals, 2) + 1, 1);
index = 0;

for nets_num = nets_num_vals
    fprintf('%d\n',nets_num);
    for data_repetition_ratio = data_repetition_vals
        fprintf('    %0.5f\n',data_repetition_ratio / nets_num);
        index = index + 1;
        if data_repetition_ratio / nets_num > 1 || data_repetition_ratio / nets_num < 0.0005
            fprintf('        passed\n');
            continue
        end
        subsample_size = round(data_repetition_ratio / nets_num * size(trainVectors, 2));
        nets = cell(nets_num,1);
        for i = 1:nets_num 
            if mod(i,round(nets_num/5)) == 0
                fprintf('        %d\n', i);
            end
            subsampleIndexes = round( ...
                (size(trainVectors, 2) - 1) * ...
                    rand(subsample_size, 1) ...
                ) + 1;
            subsampleVectors = trainVectors(:, subsampleIndexes);
            subsampleLabels = trainLabels(:, subsampleIndexes);
            net = newff(trainVectors, trainLabels, [10], {'tansig','tansig'});
            net.divideParam.trainRatio = 1;
            net.divideParam.valRatio = 0;
            net.divideParam.testRatio = 0;
            net.trainFcn = 'traingdx';
            net.trainParam.epochs = 1000;
            net.trainParam.goal = 1e-8;
            net.trainParam.lr = 0.3;
            net.trainParam.showWindow=0;
            net.trainParam.mc = 0.5;
            net.performParam.regularization = 0.8;
            net.plotFcns = {'plotperform', 'plottrainstate', 'ploterrhist', ...
                    'plotconfusion', 'plotroc'};
            [net,tr] = train(net, subsampleVectors, subsampleLabels);
            nets{i} = net;
        end

        nets_resuts = zeros(nets_num, size(valLabels, 1), ...
                size(valLabels, 2));

        for i = 1:nets_num
            nets_resuts(i,:,:) = sim(nets{i}, valVectors);
        end
        nets_resuts = mean(nets_resuts);
        nets_resuts = squeeze(nets_resuts);
        [mx, results] = max(nets_resuts);
        cp = classperf(vec2ind(valLabels), results);
        if performance_obj.best_perf < cp.CorrectRate
            performance_obj.best_perf = cp.CorrectRate;
            performance_obj.nets = nets;
            performance_obj.nets_num = nets_num;
            performance_obj.subsample_size = subsample_size;
        end
        current_performance.nets_num = nets_num;
        current_performance.subsample_size = subsample_size;
        current_performance.perf = cp.CorrectRate;
        performance_hist{index} = current_performance;
    end
end




load('testData.mat');
results = sim(net, testVectors');
[mx, results] = max(results);

export2csv(results);



%% let's try this

load('trainData.mat');

trainVectors = trainVectors';
trainLabels = trainLabels';

testVectors = trainVectors(:, 1:1900);
testLabels = trainLabels(:, 1:1900);
trainVectors = trainVectors(:, 1901:end);
trainLabels = trainLabels(:, 1901:end);

std = 0.6;
noiseData = [trainVectors + rand(size(trainVectors)) * (std / 2), ...
        trainVectors + rand(size(trainVectors)) * std, ...
        trainVectors + rand(size(trainVectors)) * (std * 2)];
noiseLables = [trainLabels, trainLabels, trainLabels];
perms = randperm(size(noiseData, 2));
noiseData = noiseData(:, perms);
noiseLables = noiseLables(:, perms);
trainVectors = [trainVectors, noiseData];
trainLabels = [trainLabels, noiseLables];

testVectors1 = trainVectors(:, 1:8000);
testLabels1 = trainLabels(:, 1:8000);
trainVectors = trainVectors(:, 8001:end);
trainLabels = trainLabels(:, 8001:end);

%clasifier 1_2_5 vs 3_4
local_trainVectors = trainVectors;
local_trainLabels = or(trainLabels == 3, trainLabels == 4);

best_perf = 1;
for i=1:50
    net = newff(local_trainVectors, local_trainLabels, [20 ,10], ...
        {'tansig', 'tansig', 'tansig'});
    net.divideParam.trainRatio = 0.8;
    net.divideParam.valRatio = 0.2;
    net.divideParam.testRatio = 0;
    net.trainFcn = 'traingdx';
    net.trainParam.epochs = 500;
    net.trainParam.goal = 1e-8;
    net.trainParam.lr = 0.03;
    net.trainParam.mc = 0.5;
    net.performParam.regularization = 0.2;
    net.performFcn = 'crossentropy';
    net.plotFcns = {'plotperform', 'plottrainstate', 'ploterrhist', ...
            'plotconfusion', 'plotroc'};
    [net,tr] = train(net, local_trainVectors, local_trainLabels);
    if tr.best_vperf < best_perf 
        net_all = net;
        best_perf = tr.best_vperf;
    end
end


%clasifier 1_2_5
local_trainVectors = trainVectors(:, or(or(trainLabels == 1, ...
    trainLabels == 2), trainLabels == 5));
local_trainLabels = trainLabels(:, or(or(trainLabels == 1, ...
    trainLabels == 2), trainLabels == 5));
local_trainLabels(local_trainLabels == 5) = 3;
local_trainLabels = full(ind2vec(local_trainLabels));

best_perf = 1;
for i=1:50
    net = newff(local_trainVectors, local_trainLabels, [20 10], ...
        {'tansig', 'tansig', 'softmax'});
    net.divideParam.trainRatio = 0.8;
    net.divideParam.valRatio = 0.2;
    net.divideParam.testRatio = 0;
    net.trainFcn = 'traingdx';
    net.trainParam.epochs = 500;
    net.trainParam.goal = 1e-8;
    net.trainParam.lr = 0.03;
    net.trainParam.mc = 0.5;
    net.performParam.regularization = 0.5;
    net.performFcn = 'crossentropy';
    net.plotFcns = {'plotperform', 'plottrainstate', 'ploterrhist', ...
            'plotconfusion', 'plotroc'};
    [net,tr] = train(net, local_trainVectors, local_trainLabels);
    if tr.best_vperf < best_perf 
        net_1_2_5 = net;
        best_perf = tr.best_vperf;
    end
end

%clasifier 3_4
local_trainVectors = trainVectors(:, or(trainLabels == 3, ...
    trainLabels == 4));
local_trainLabels = trainLabels(:, or(trainLabels == 3, ...
    trainLabels == 4));
local_trainLabels = local_trainLabels == 4;

best_perf = 1;
for i=1:50
    net = newff(local_trainVectors, local_trainLabels, [5], ...
        {'tansig','tansig'});
    net.divideParam.trainRatio = 0.6;
    net.divideParam.valRatio = 0.2;
    net.divideParam.testRatio = 0.2;
    net.trainFcn = 'traingdx';
    net.trainParam.epochs = 500;
    net.trainParam.goal = 1e-8;
    net.trainParam.lr = 0.03;
    net.trainParam.mc = 0.5;
    net.performParam.regularization = 0.9;
    net.performFcn = 'crossentropy';
    net.plotFcns = {'plotperform', 'plottrainstate', 'ploterrhist', ...
            'plotconfusion', 'plotroc'};
    [net,tr] = train(net, local_trainVectors, local_trainLabels);
    if tr.best_vperf < best_perf 
        net_3_4 = net;
        best_perf = tr.best_vperf;
    end
end

%USING TEST DATA
testVectorsCurr = testVectors;
testLabelsCurr = testLabels;

results = zeros(1, size(testVectorsCurr,2));
results_1 = hardlim(sim(net_all, testVectorsCurr) - 0.5);
c_1_2_5 = testVectorsCurr(:, results_1 == 0);
c_3_4 = testVectorsCurr(:, results_1 == 1);
[aux, results_2] = max(sim(net_1_2_5, c_1_2_5));
results_2(results_2 == 3) = 5;
results_3 = hardlim(sim(net_3_4, c_3_4) - 0.5);
results_3 = double(results_3) + 3;
results(results_1 == 0) = results_2;
results(results_1 == 1) = results_3;

plotconfusion(full(ind2vec(testLabelsCurr)),full(ind2vec(results)));

% export
load('testData.mat');
testVectors = testVectors';

results = zeros(1, size(testVectors,2));
results_1 = hardlim(sim(net_all, testVectors) - 0.5);

c_1_2_5 = testVectors(:, results_1 == 0);
c_3_4 = testVectors(:, results_1 == 1);

[aux, results_2] = max(sim(net_1_2_5, c_1_2_5));
results_2(results_2 == 3) = 5;

results_3 = hardlim(sim(net_3_4, c_3_4) - 0.5);
results_3 = double(results_3) + 3;

results(results_1 == 0) = results_2;
results(results_1 == 1) = results_3;

export2csv(results);

%% one more

%% MVP
load('trainData.mat');

trainVectors = trainVectors';
trainLabels = int2labelVec(trainLabels');

net = newff(trainVectors, trainLabels, [500, 40], {'tansig','tansig','tansig'});
net.divideParam.trainRatio = 0.6;
net.divideParam.valRatio = 0.2;
net.divideParam.testRatio = 0.2;
net.trainFcn = 'traingdx';
net.trainParam.epochs = 5000;
net.trainParam.goal = 1e-2;
net.trainParam.lr = 0.3;
net.trainParam.mc = 0.5;
net.plotFcns = {'plotperform', 'plottrainstate', 'ploterrhist', ...
        'plotconfusion', 'plotroc'};
net.performFcn = 'crossentropy';
net.performParam.regularization = 0.5;
[net,tr] = train(net, trainVectors, trainLabels);

load('testData.mat');
results = sim(net, testVectors');
[mx, results] = max(results);

export2csv(results);


%% testarea 3 - > regulatization, > std, > epochs (last matlab #2)
load('trainData.mat');
trainVectors = trainVectors';
trainLabels = trainLabels';

nets = cell(5,1);
perfs = cell(5,1);

for i=1:5
    local_best_perf = 1;
    train_perf = 1;
    for j=1:5
        oneVsAllLabels = double(trainLabels == i);
        positiveData = trainVectors(:, oneVsAllLabels == 1);
        oneVsAllVectors = [trainVectors, positiveData, positiveData, positiveData];
        oneVsAllLabels = [oneVsAllLabels, ones(1,size(positiveData, 2)*3)];


        % generate noisy data
        std = 0.8;
        noiseData = [oneVsAllVectors + rand(size(oneVsAllVectors)) * (std / 2), ...
                oneVsAllVectors + rand(size(oneVsAllVectors)) * std, ...
                oneVsAllVectors + rand(size(oneVsAllVectors)) * (std * 2)];
        noiseLables = [oneVsAllLabels, oneVsAllLabels, oneVsAllLabels];
        perms = randperm(size(noiseData, 2));
        noiseData = noiseData(:, perms);
        noiseLables = noiseLables(:, perms);

        oneVsAllVectors = [oneVsAllVectors, noiseData];
        oneVsAllLabels = [oneVsAllLabels, noiseLables];

        perms = randperm(size(oneVsAllVectors, 2));
        oneVsAllVectors = oneVsAllVectors(:, perms);
        oneVsAllLabels = oneVsAllLabels(:, perms);

        net = newff(oneVsAllVectors, oneVsAllLabels, [20], {'tansig','tansig'});

        net.performFcn = 'crossentropy';
        net.performParam.regularization = 0.3;

        net.divideParam.trainRatio = 0.8;
        net.divideParam.valRatio = 0.2;
        net.divideParam.testRatio = 0;
        net.trainFcn = 'traingdm';
        net.trainParam.epochs = 7000;
        net.trainParam.goal = 1e-3;
        net.trainParam.max_fail = 10;
        net.trainParam.lr = 0.3;
        net.trainParam.mc = 0.5;
        net.trainParam.show = 500;
        net.trainParam.showWindow=0;
        net.trainParam.showCommandLine=1;
        net.plotFcns = {'plotperform', 'plottrainstate', 'ploterrhist', ...
                'plotconfusion', 'plotroc'};
        [net,perf] = train(net, oneVsAllVectors, oneVsAllLabels);
        if perf.best_vperf < local_best_perf
            local_best_perf = perf.best_vperf;
            train_perf = perf.best_perf;
            nets{i} = net;
        end
    end
    fprintf('%s/%s validtion perf/train perf on %s\n\n', local_best_perf, train_perf, i);
end

[aux, results] = max([sim(nets{1}, trainVectors);
                        sim(nets{2}, trainVectors);
                        sim(nets{3}, trainVectors);
                        sim(nets{4}, trainVectors);
                        sim(nets{5}, trainVectors);]);
 
plotconfusion(full(ind2vec(trainLabels)), full(ind2vec(results)));                 

% load test data
load('testData.mat');
% simulate network over test data
[aux, results] = max([sim(nets{1}, testVectors');
                        sim(nets{2}, testVectors');
                        sim(nets{3}, testVectors');
                        sim(nets{4}, testVectors');
                        sim(nets{5}, testVectors');]);
% export
export2csv(results);
