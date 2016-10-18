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

function [Xtrain, Ytrain, Xtest, Ytest] = load_gist(dataset, opts, normalizeX)
	if nargin < 3, normalizeX = 1; end
	if ~normalizeX, myLogInfo('will NOT pre-normalize data'); end

	if opts.windows
		basedir = '\\ivcfs1\codebooks\hashing_project\data';
	else
		basedir = '/research/codebooks/hashing_project/data';
	end

	tic;
	if strcmp(dataset, 'cifar')
		load([basedir '/cifar-10/descriptors/gist.mat']);
		gist        = [traingist; testgist];
		gistlabels  = [trainlabels+1; testlabels+1];  % NOTE labels are 0 to 9
		gistlabels = gistlabels .* 10;
		tstperclass = 100;

		if normalizeX 
			% normalize features
			gist = bsxfun(@minus, gist, mean(gist,1));  % first center at 0
			gist = normalize(double(gist));  % then scale to unit length
		end
		[Xtrain, Ytrain, Xtest, Ytest] = ...
			split_train_test(gist, gistlabels, tstperclass);

	elseif strcmp(dataset, 'sun')
		load([basedir '/sun397/SUN_gist.mat']);
		gistlabels  = labels+1;  % NOTE labels are 0 to 396
		gistlabels = gistlabels .* 10;
		tstperclass = 10;

		if normalizeX 
			% normalize features
			gist = bsxfun(@minus, gist, mean(gist,1));  % first center at 0
			gist = normalize(double(gist));  % then scale to unit length
		end
		[Xtrain, Ytrain, Xtest, Ytest] = ...
			split_train_test(gist, gistlabels, tstperclass);

	elseif strcmp(dataset, 'nus')
	    gist = load([basedir '/nuswide/BoW_int.dat']);
	    tags = load([basedir '/nuswide/AllLabels81.txt']);
	    tstperclass = 30;

		if normalizeX 
			% normalize features
			gist = bsxfun(@minus, gist, mean(gist,1));  % first center at 0
			gist = normalize(double(gist));  % then scale to unit length
		end
		[Xtrain, Ytrain, Xtest, Ytest] = ...
			split_train_test_nus(gist, tags, tstperclass);

	else, error('unknown dataset'); end

	whos Xtrain Ytrain Xtest Ytest
	myLogInfo('Dataset "%s" loaded in %.2f secs', dataset, toc);
end

% --------------------------------------------------------
function [Xtrain, Ytrain, Xtest, Ytest] = ...
		split_train_test(gist, gistlabels, tstperclass)

	% normalize features
	gist = bsxfun(@minus, gist, mean(gist,1));  % first center at 0
	gist = normalize(gist);  % then scale to unit length

	% construct test and training set
	clslbl      = unique(gistlabels);
	num_classes = length(clslbl);
	testsize    = num_classes * tstperclass;
	trainsize   = size(gist, 1) - testsize;
	gistdim     = size(gist, 2);
	Xtest       = zeros(testsize, gistdim);
	Ytest       = zeros(testsize, 1);
	Xtrain      = zeros(trainsize, gistdim);
	Ytrain      = zeros(trainsize, 1);
	count = 0;
	for i = 1:num_classes
		ind = find(gistlabels == clslbl(i));
		n_i = length(ind);
		ind = ind(randperm(n_i));
		Xtest((i-1)*tstperclass+1:i*tstperclass,:) = gist(ind(1:tstperclass),:);
		Ytest((i-1)*tstperclass+1:i*tstperclass)   = clslbl(i);
		Xtrain(count+1:count+n_i-tstperclass, :)   = gist(ind(tstperclass+1:end),:);
		Ytrain(count+1:count+n_i-tstperclass, :)   = clslbl(i);
		count = count + n_i - tstperclass;
	end
	% randomize again
	ind    = randperm(size(Xtrain, 1));
	Xtrain = Xtrain(ind, :);
	Ytrain = Ytrain(ind);
	ind    = randperm(size(Xtest, 1));
	Xtest  = Xtest(ind, :);
	Ytest  = Ytest(ind);

	%cateTrainTest = repmat(Ytrain, 1, length(Ytest)) ...
		%== repmat(Ytest, 1, length(Ytrain))';
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

	% TODO after eliminating cateTrainTest, get_results will have to deal with
	% the multi-label case explicitly
	%cateTrainTest = (Ytrain * Ytest' > 0);
end
