% Copyright (c) 2016 Fatih Cakir Kun He
% Permission is hereby granted, free of charge, to any person obtaining a 
% copy of this software and associated documentation files (the "Software"), 
% to deal in the Software without restriction, including without limitation 
% the rights to use, copy, modify, merge, publish, distribute, sublicense, 
% and/or sell copies of the Software, and to permit persons to whom the 
% Software is furnished to do so, subject to the following conditions:
% 
% The above copyright notice and this permission notice shall be included 
% in all copies or substantial portions of the Software.
% 
% THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS 
% OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
% FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
% AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
% WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF 
% OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
%
% If using this software please consider citing the below article:
%
% @inproceedings{Cakir:2014:SHE,
%  author = {Cakir, Fatih and Sclaroff, Stan},
%  title = {Supervised Hashing with Error Correcting Codes},
%  booktitle = {Proceedings of the 22Nd ACM International Conference on Multimedia},
%  year = {2014}
% } 
% Email: fcakirs@gmail.com

function [Xtrain, Ytrain, Xtest, Ytest, Names] = load_cnn(dataset, opts, normalizeX)
	if nargin < 3, normalizeX = 1; end
	if ~normalizeX, myLogInfo('will NOT pre-normalize data'); end

	% NOTE: labels are originally [0, L-1], first add 1 to make [1, L]
	%       then multiply by 10 to make [10, L*10]
	%
	%       Next, for each item, if HIDE label in training, +1 to its Y
	%       So eg. for first class, labeled ones have 10, unlabeled have 11
	%
	%       At test time labels can be recovered by dividing 10

	tic;
	if strcmp(dataset, 'cifar')
		if opts.windows
			basedir = '\\ivcfs1\codebooks\hashing_project\data';
		else
			basedir = '/research/codebooks/hashing_project/data';
		end
		load([basedir '/cifar-10/descriptors/trainCNN.mat']); % trainCNN
		load([basedir '/cifar-10/descriptors/traininglabelsCNN.mat']); % traininglabels
		load([basedir '/cifar-10/descriptors/testCNN.mat']); % testCNN
		load([basedir '/cifar-10/descriptors/testlabelsCNN.mat']); % testlabels
		X = [trainCNN; testCNN];
		Y = [traininglabels+1; testlabels+1];
		Y = Y .* 10;
		T = 100;

		if normalizeX 
			% normalize features
			X = bsxfun(@minus, X, mean(X,1));  % first center at 0
			X = normalize(double(X));  % then scale to unit length
		end

		% fully supervised
		% TODO names
		[ind_train, ind_test, Ytrain, Ytest] = split_train_test(X, Y, T, 0);
		Xtrain = X(ind_train, :);
		Xtest  = X(ind_test, :);
		clear Names
		Names.train = num2cell(ind_train);
		Names.test = num2cell(ind_test);


	elseif strcmp(dataset, 'sun')

		load('/research/codebooks/hashing_project/data/sun397/alltrain_sun_fc7_final.mat');
    
		X = alldata;
		Y = allclasses;
		Y = (Y + 1)*10;  % NOTE labels are 0 to 396
		T = 10;
		clear alldata allclasses

		if normalizeX 
			% normalize features
			X = bsxfun(@minus, X, mean(X,1));  % first center at 0
			X = normalize(double(X));  % then scale to unit length
		end
		[ind_train, ind_test, Ytrain, Ytest] = ...
			split_train_test(X, Y, T ,0);

		Xtrain = X(ind_train, :);
		Xtest  = X(ind_test, :);

	elseif strcmp(dataset, 'places')
		if opts.windows
			basedir = '\\kraken\object_detection\data';
		else
			basedir = '/research/object_detection/data';
		end
		% loads variables: pca_feats, labels, images
		clear pca_feats labels images
		load([basedir '/places/places_alexnet_fc7pca128.mat']);
		X = pca_feats;
		Y = (labels + 1)*10;
		T = 20;
		L = opts.labelspercls;  % default 2500, range {0}U[500, 5000]

		if normalizeX 
			% normalize features
			X = bsxfun(@minus, X, mean(X,1));  % first center at 0
			X = normalize(double(X));  % then scale to unit length
		end

		% semi-supervised
		[ind_train, ind_test, Ytrain, Ytest] = split_train_test(X, Y, T, L);
		Xtrain = X(ind_train, :);
		Xtest  = X(ind_test, :);
		clear Names
		Names.train = images(ind_train);
		Names.test  = images(ind_test);


	elseif strcmp(dataset, 'nus')
		if opts.windows
			basedir = '\\ivcfs1\codebooks\hashing_project\data';
		else
			basedir = '/research/codebooks/hashing_project/data';
		end
	    load([basedir '/nuswide/AllNuswide_fc7.mat']);  % FVs
	    Y = load([basedir '/nuswide/AllLabels81.txt']);
		X = double(FVs);  clear FVs
		T = 30;

		if normalizeX 
			% normalize features
			X = bsxfun(@minus, X, mean(X,1));  % first center at 0
			X = normalize(double(X));  % then scale to unit length
		end

		% TODO Names
		[Xtrain, Ytrain, Xtest, Ytest] = split_train_test_nus(X, Y, T);
		Names = [];

	else, error(['unknown dataset: ' dataset]); end

	whos Xtrain Ytrain Xtest Ytest
	myLogInfo('Dataset "%s" loaded in %.2f secs', dataset, toc);
end

% --------------------------------------------------------
function [ind_train, ind_test, Ytrain, Ytest] = split_train_test(X, Y, T, L)
	% X: original features
	% Y: original labels
	% T: # test points per class
	% L: [optional] # labels to retain per class
	if nargin < 4, L = 0; end

	% randomize
	%I = randperm(size(X, 1));
	%X = X(I, :);
	%Y = Y(I);

	D = size(X, 2)

	labels = unique(Y);
	ntest  = length(labels) * T;
	ntrain = size(X, 1) - ntest;
	%Xtrain = zeros(ntrain, D);  Xtest = zeros(ntest, D);
	Ytrain = zeros(ntrain, 1);  Ytest = zeros(ntest, 1);
	ind_train = [];
	ind_test  = [];
	
	% construct test and training set
	cnt = 0;
	for i = 1:length(labels)
		% find examples in this class, randomize ordering
		ind = find(Y == labels(i));
		n_i = length(ind);
		ind = ind(randperm(n_i));

		% assign test
		%Xtest((i-1)*T+1:i*T, :) = X(ind(1:T), :);
		Ytest((i-1)*T+1:i*T)    = labels(i);
		ind_test = [ind_test; ind(1:T)];

		% assign train
		st = cnt + 1; 
		ed = cnt + n_i - T;
		%Xtrain(st:ed, :) = X(ind(T+1:end), :);
		ind_train = [ind_train; ind(T+1:end)];
		Ytrain(st:ed)    = labels(i);
		if L > 0  
			% if requested, hide some labels
			if st + L > ed
				warning(sprintf('%s Class%d: ntrain=%d<%d=labelspercls, keeping all', ...
					labels(i), n_i-T, L));
			else
				% add 1 to mark unlabeled items
				Ytrain(st+L: ed) = Ytrain(st+L: ed) + 1;
			end
		end
		cnt = ed;
	end

	% randomize again
	ind    = randperm(ntrain);
	%Xtrain = Xtrain(ind, :);
	Ytrain = Ytrain(ind);
	ind_train = ind_train(ind);

	ind    = randperm(ntest);
	%Xtest  = Xtest(ind, :);
	Ytest  = Ytest(ind);
	ind_test = ind_test(ind);
end


% --------------------------------------------------------
function [Xtrain, Ytrain, Xtest, Ytest] = ...
		split_train_test_nus(gist, tags, tstperclass)

	% normalize features
	gist = bsxfun(@minus, gist, mean(gist,1));  % first center at 0
	gist = normalize(gist);  % then scale to unit length

	% construct test and training set
	num_classes = 81;
	testsize    = num_classes * tstperclass;
	ind         = randperm(size(gist, 1));
	Xtest       = gist(ind(1:testsize), :);
	Ytest       = tags(ind(1:testsize), :);
	Xtrain      = gist(ind(testsize+1:end), :);
	Ytrain      = tags(ind(testsize+1:end), :);

	% randomize again
	ind    = randperm(size(Xtrain, 1));
	Xtrain = Xtrain(ind, :);
	Ytrain = Ytrain(ind, :);
	ind    = randperm(size(Xtest, 1));
	Xtest  = Xtest(ind, :);
	Ytest  = Ytest(ind, :);
end
