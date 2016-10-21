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

	%
	%       Next, for each item, if HIDE label in training, +1 to its Y
	%       So eg. for first class, labeled ones have 10, unlabeled have 11
	%
	%       At test time labels can be recovered by dividing 10

	f strcmp(dataset, 'cifar')
		load([opts.datadir '/cifar/train_vgg_fc7.mat']); % trainCNN
		load([opts.datadir '/cifar/traininglabels.mat']); % traininglabels
		load([opts.datadir '/cifar/test_vgg_fc7.mat']); % testCNN
		load([opts.datadir '/cifar/testlabels.mat']); % testlabels
		X = [trainCNN; testCNN];
		Y = [traininglabels+1; testlabels+1];
		T = 100;

		if normalizeX 
			% normalize features
			X = bsxfun(@minus, X, mean(X,1));  % first center at 0
			X = normalize(double(X));  % then scale to unit length
		end

		% fully supervised
		% TODO names
		[ind_train, ind_test, Ytrain, Ytest] = split_train_test(X, Y, T);
		Xtrain = X(ind_train, :);
		Xtest  = X(ind_test, :);
		clear Names
		Names.train = num2cell(ind_train);
		Names.test = num2cell(ind_test);


	elseif strcmp(dataset, 'sun')

		load([opts.datadir '/sun/sun_vgg_fc7.mat');
    
		X = alldata;
		Y = allclasses;
		Y = (Y + 1);  % NOTE labels are 0 to 396
		T = 10;
		clear alldata allclasses

		if normalizeX 
			% normalize features
			X = bsxfun(@minus, X, mean(X,1));  % first center at 0
			X = normalize(double(X));  % then scale to unit length
		end
		[ind_train, ind_test, Ytrain, Ytest] = ...
			split_train_test(X, Y, T);

		Xtrain = X(ind_train, :);
		Xtest  = X(ind_test, :);

	else, error(['unknown dataset: ' dataset]); end

	whos Xtrain Ytrain Xtest Ytest
	myLogInfo('Dataset "%s" loaded in %.2f secs', dataset, toc);i
end

% --------------------------------------------------------
function [ind_train, ind_test, Ytrain, Ytest] = split_train_test(X, Y, T)
	% X: original features
	% Y: original labels
	% T: # test points per class

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
