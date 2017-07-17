datasethelper = {}
function datasethelper.randomSplit(list, fraction)
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

function datasethelper.filesInDir(path)
    local files = {}
    for file in lfs.dir(path) do
        if file ~= "." and file ~= ".." then
            table.insert(files, file)
        end
    end
    return files
end

return datasethelper
