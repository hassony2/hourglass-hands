require 'lfs'
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

function filesInDir(path)
    local files = {}
    for file in lfs.dir(path) do
        if file ~= "." and file ~= ".." then
            table.insert(files, file)
        end
    end
    return files
end

function Dataset:__init()
    self.rgbFolder = paths.concat(opt.dataDir, 'rgb')
    self.annotFolder = paths.concat(opt.dataDir, '2Dcoord')
    files = filesInDir(self.rgbFolder)
    self.prefixes = {}
    for i, file in ipairs(files) do
        local prefix = string.split(file, "%.")[1]
        table.insert(self.prefixes, prefix)
    end
    self.nJoints = 20
	self.accIdx = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
				  15, 16, 17, 18, 19, 20}
    self.flipRef = {}
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
	--- modulo for index mapping from and to folder
    self.folderMult = 1000    
          
    -- Index reference
    allIdxs = torch.range(1, table.getn(self.prefixes))
    opt.idxRef = {}
    local residualIdxs = nil
    opt.idxRef.test, residualIdxs = randomSplit(allIdxs, self.testFrac)
    opt.idxRef.valid, opt.idxRef.train = randomSplit(residualIdxs, self.valFrac)
    torch.save(opt.save .. '/options.t7', opt)

    self.nsamples = {train=opt.idxRef.train:numel(),
                     valid=opt.idxRef.valid:numel(),
                     test=opt.idxRef.test:numel()}

    -- For final predictions
    opt.testIters = self.nsamples.test
    opt.testBatch = 1
end

function Dataset:size(set)
    return self.nsamples[set]
end

function Dataset:getPath(idx)
    local prefix = self.prefixes[idx]
    return paths.concat(self.rgbFolder, prefix .. '.png' )
end

function Dataset:loadImage(idx)
    return image.load(self:getPath(idx))
end

function Dataset:getPartInfo(idx, verbose)
	-- retrieve joint localizations
	verbose = verbose or false
    local prefix = self.prefixes[idx]
    local fileName = paths.concat(self.annotFolder, prefix .. '.txt')

    local file = io.open(fileName)
    local pts = torch.zeros(self.nJoints, 2)
    if file then
		if verbose then
        	print('got file!')
		end
        local lineIdx = 1
        for line in file:lines() do
            local x, y = unpack(line:split(" ")) 
            pts[lineIdx][1] = x
            pts[lineIdx][2] = 540 - y
            lineIdx = lineIdx + 1
        end
    else
		if verbose then
        	print('file '.. fileName .. ' not found')
		end
    end
    pts:add(1)

    -- center of hand is considered as center of image
    local c = torch.Tensor({320, 240})
    local s = 4
    return pts, c, s
end

function Dataset:normalize(idx)
    -- Used to account for image size variation in PCK measurement
    return 100
end

return M.Dataset
