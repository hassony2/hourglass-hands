local M = {}
Dataset = torch.class('pose.Dataset',M)

function randomSplit(list, fraction)
	local mixedlist = torch.randperm(list:size(1))
	local splitIdx = math.floor(list:numel()*fraction)
	local firstSplit = mixedlist[{{1, splitIdx}}]
	local secondSplit = mixedlist[{{splitIdx + 1, -1}}]
	return firstSplit, secondSplit
end

function Dataset:__init()
    self.annotationDir = 'joints_2D_GT'
    self.nJoints = 14 
    self.accIdxs = {1,2,3,4,5,6,7,8,9,10,11,12,13}
    -- self.flipRef = {{1,6},   {2,5},   {3,4},
    --                {11,16}, {12,15}, {13,14}}
    -- Pairs of joints for drawing skeleton
    self.skeletonRef = {{1,2,1},    {2,3,1},
                        {4,5,2},    {5,6,2},
                        {7,8,0},    {8,9,0},
                        {10,11,3},   {11,12,3}
                        {13,14,4}}
	self.testFrac = 0.2
	self.valFrac = 0.2

    -- Index reference
    if not opt.idxRef then
        local allIdxs = torch.range(1, 205, 5)
    	opt.idxRef = {}
        opt.idxRef.test, opt.idxRef.train = randomSplit(allIdxs, self.testFrac)

        -- Set up random training/validation split
        opt.idxRef.valid, opt.idxRef.train = randomSplit(opt.idxRef.train, self.valFrac)

        torch.save(opt.save .. '/options.t7', opt)
    end

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
    return paths.concat(opt.dataDir, idx .. '.png' )
end

function Dataset:loadImage(idx)
    return image.load(self:getPath(idx))
end

function Dataset:getPartInfo(idx, verbose)
	-- retrieve joint localizations
	verbose = verbose or true
    local annotationPath = path.concat(opt.dataDir, self.annotationDir)
	local fileName = path.concat(annotationPath, idx .. '.txt')
    local file = io.open(fileName)
    local pts = torch.zeros(14, 2)
    if file then
		if verbose then:
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
        	print('file not found')
		end
    end
    pts:add(1)

    -- center of hand is considered as center of image
    local c = torch.Tensor({320, 240})
    local s = 1
    return pts, c, s
end

function Dataset:normalize(idx)
    -- Used to account for image size variation in PCK measurement
    return 100
end

return M.Dataset

