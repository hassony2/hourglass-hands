require 'lfs'
paths.dofile("datasethelper.lua")
local M = {}
Dataset = torch.class('pose.Dataset',M)

function Dataset:__init()
    self.dataDir =  opt.dataDir
    self.useDepth = opt.useDepth
    print('opt.dataDir : ' .. self.dataDir)
    seq1Dir = paths.concat(self.dataDir, 'Seq1')
    self.annotFolder = seq1Dir
    files1 = datasethelper.filesInDir(seq1Dir)
    self.prefixes = {}
    for i, file in ipairs(files1) do
        local prefix = string.match(file, "(%a*%d+).jpg")
        if prefix  then
            local annotPath = paths.concat(self.annotFolder, prefix .. '-1.txt')
            local annotFile = io.open(annotPath, 'r')
            if annotFile then
                table.insert(self.prefixes, prefix)
            end
        end
    end
    local synthIdx = {22, 23, 24, 25,
                          18, 19, 20, 21,
                          14, 15, 16, 17,
                          10, 11, 12, 13,
                          6, 7, 8, 9}
    local jointCorresp = {}
    for k, v in pairs(synthIdx) do
        jointCorresp[v] = k
    end
    self.jointCorresp = jointCorresp
    self.nJoints = table.getn(synthIdx)

	self.accIdx = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
				  15, 16, 17, 18, 19, 20}
    self.flipRef = {}
    -- Pairs of joints for drawing skeleton
    self.skeletonRef = {{1,2,1},    {2,3,1}, {3, 4, 1},
                        {5,6,2},    {6,7,2}, {7, 8, 2},
                        {9, 10, 3},    {10, 11, 3}, {11, 12, 3},
                        {13,14,4}, {14, 15, 4}, {15, 16, 4},
                        {17, 18, 5}, {18, 19, 5}, {19, 20, 5}}
    -- percentage of test samples
	self.testFrac = 0.9 
    -- percentage of validation samples among remaining samples
	self.valFrac = 0.5

    -- Index reference
    allIdxs = torch.range(1, table.getn(self.prefixes))
    opt.idxRef = {}
    local residualIdxs = nil
    opt.idxRef.test, residualIdxs = datasethelper.randomSplit(allIdxs, self.testFrac)
    opt.idxRef.valid, opt.idxRef.train = datasethelper.randomSplit(residualIdxs, self.valFrac)
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
    local path = nil
    if self.useDepth then
        path = paths.concat(self.annotFolder, prefix .. '_z.png')
    else
        path = paths.concat(self.annotFolder, prefix .. '.jpg' )
    end
    return path
end

function Dataset:loadImage(idx)
    return image.load(self:getPath(idx))
end

function Dataset:getPartInfo(idx, verbose)
	-- retrieve joint localizations
	verbose = verbose or false
    local prefix = self.prefixes[idx]
    local fileName = nil
    if self.useDepth then
        fileName = paths.concat(self.annotFolder, prefix .. '-1_z.txt')
    else
        fileName = paths.concat(self.annotFolder, prefix .. '-1.txt')
    end

    local file = io.open(fileName)
    local pts = torch.zeros(self.nJoints, 2)
    if file then
		if verbose then
        	print('got file!')
		end
        local lineIdx = 1
        for line in file:lines() do
            local jointIdx = self.jointCorresp[lineIdx]
            if jointIdx ~= nil then
                local boneName, coordinates = unpack(line:split(":"))
                local x, y, z = unpack(coordinates:split("\t")) 
                pts[jointIdx][1] = x
                pts[jointIdx][2] = y
            end
            lineIdx = lineIdx + 1
        end
    else
		if verbose then
        	print('file '.. fileName .. ' not found')
		end
    end
    pts:add(1)
    -- center of hand is considered as center of image
    local c = (torch.min(pts, 1) + torch.max(pts, 1))/2  
    c = torch.squeeze(c)
    local s = 4
    return pts, c, s
end

function Dataset:normalize(idx)
    -- Used to account for image size variation in PCK measurement
    return 100
end

return M.Dataset
