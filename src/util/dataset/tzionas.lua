local M = {}
Dataset = torch.class('pose.Dataset',M)

function randomSplit(list, fraction)
	---- splits list into two random portions
	---- the first one containing fraction of the initial values
	---- and the second one 1 - fraction
    local mixedIdx = torch.randperm(list:size(1))
    local splitIdx = math.floor(list:numel()*fraction)
    local firstIdx = mixedIdx[{{1, splitIdx}}]
    local firstSplit = list:index(1, firstIdx:long())
    local secondIdx = mixedIdx[{{splitIdx + 1, -1}}]
    local secondSplit = list:index(1, secondIdx:long())
    return firstSplit, secondSplit
 end



function Dataset:__init()
    local folderNb = 20
    self.folderNames = {}
	
    for i=1, folderNb do
        self.folderNames[i] = string.format("%02d", i) .. '/1/'
    end
	self.trainFolders = {1, 3, 5, 6, 7, 8, 9, 10,
                         13, 14, 15, 16, 18, 19}
	self.testFolders = {2, 12, 17}
	self.valFolders = {4, 11, 20}
	self.valFolders = {4}
    self.annotationDir = 'joints_2D_GT'
    self.nJoints = 28
	self.accIdx = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
				  15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28}
    self.flipRef = {{1, 15},   {2,16},   {3,17},
                    {4,18}, {5,19}, {6,20},
					{7, 21}, {8,22}, {9,23},
					{10,24}, {11,25} ,{12,26},
					{13,27}, {14,28}}
    -- Pairs of joints for drawing skeleton
    self.skeletonRef = {{1,2,1},    {2,3,1},
                        {4,5,2},    {5,6,2},
                        {7,8,0},    {8,9,0},
                        {10,11,3},  {11,12,3},
                        {13,14,4}}
    -- percentage of test samples
	self.testFrac = 0.2
    -- percentage of validation samples among remaining samples
	self.valFrac = 0.2

    -- number of frames in each video
    self.imageNbs = {230, 152, 154, 172, 264, 
                     135, 193, 105, 137, 138,
                     133, 726, 777, 377, 208,
                     196, 399, 295, 250, 185}
	-- in case where annotation doesn't start at 0, value of beginning
	self.beginNb = {[12]=420, [13]=480, [14]=250}
	--- modulo for index mapping from and to folder
    self.folderMult = 1000    
          
    -- Index reference
    if not opt.idxRef then
    	opt.idxRef = {}
		opt.idxRef.test = self:getVideoIndexes(self.testFolders)
		opt.idxRef.train = self:getVideoIndexes(self.trainFolders)
		opt.idxRef.valid = self:getVideoIndexes(self.valFolders) 	
        torch.save(opt.save .. '/options.t7', opt)
    end

    self.nsamples = {train=opt.idxRef.train:numel(),
                     valid=opt.idxRef.valid:numel(),
                     test=opt.idxRef.test:numel()}

    -- For final predictions
    opt.testIters = self.nsamples.test
    opt.testBatch = 1
end

function Dataset:getFolderImFromIdx(idx)
	--- gets folder and image number from idx
	local folderNb = math.floor(idx/self.folderMult)
	local imgNb = idx - folderNb*self.folderMult
	return folderNb, imgNb 
end

function Dataset:getVideoIndexes(folderNbs)
    --- folderNbs as lua table
	--- returns a list of idxs according to the following convention
	--- image 015 in folder 2 : idx = 2015 (= 2*1000 + 15)
    local idxs = torch.Tensor({})
    for i, folderNb in ipairs(folderNbs) do
        annotatedIdxs = torch.range(self.beginNb[folderNb] or 0, self.imageNbs[folderNb], 5):add(self.folderMult*folderNb)

        idxs = torch.cat(idxs, annotatedIdxs , 1)
    end
    return idxs
end

function Dataset:size(set)
    return self.nsamples[set]
end

function Dataset:getPath(idx)
	local imgNb
	local folderNb
	folderNb, imgNb = self:getFolderImFromIdx(idx)
    return paths.concat(opt.dataDir, self.folderNames[folderNb] .. 'rgb/' .. string.format("%03d", imgNb) .. '.png' )
end

function Dataset:loadImage(idx)
    return image.load(self:getPath(idx))
end

function Dataset:getPartInfo(idx, verbose)
	-- retrieve joint localizations
	verbose = verbose or false
	local imgNb
	local folderNb
	folderNb, imgNb = self:getFolderImFromIdx(idx)
    local annotationPath = paths.concat(opt.dataDir, self.folderNames[folderNb] .. self.annotationDir)
	local fileName = paths.concat(annotationPath, string.format("%03d", imgNb) .. '.txt')

    local file = io.open(fileName)
    local pts = torch.zeros(self.nJoints, 2)
    if file then
		if verbose then
        	print('got file!')
		end
        for line in file:lines() do
            local idx, x, y = unpack(line:split("\t")) 
            idx = idx + 1 -- from 0 to 1 indexing
            pts[idx][1] = x
            pts[idx][2] = y
        end
    else
		if verbose then
        	print('file '.. fileName .. ' not found')
		end
    end
    pts:add(1)

    -- center of hand is considered as center of image
    local c = torch.Tensor({320, 240})
    local s = 2
    return pts, c, s
end

function Dataset:normalize(idx)
    -- Used to account for image size variation in PCK measurement
    return 100
end

return M.Dataset
