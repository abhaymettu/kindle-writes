-- bpe.lua TEXT N_TRAIN VOCAB_SIZE OUTDIR   Lua 5.1 port of trainer/prepare_tokens.py (same merges, same files).
-- Merges are learned on bytes [0, N_TRAIN) only; train and held-out are encoded separately.
-- Writes OUTDIR/tokens.bin and OUTDIR/vocab.bin in nightwriter's formats.

local path, n_train, vsize, out = arg[1], tonumber(arg[2]), tonumber(arg[3]), arg[4]
local f = assert(io.open(path, "rb")); local raw = f:read("*a"); f:close()
local train, held = raw:sub(1, n_train), raw:sub(n_train + 1)

local function cls(b)
  if b == 32 or (b >= 9 and b <= 13) then return 0 end
  if (b >= 65 and b <= 90) or (b >= 97 and b <= 122) then return 1 end
  if b >= 48 and b <= 57 then return 2 end
  return 3
end
local CLS = {}
for b = 0, 255 do CLS[b] = cls(b) end

-- calls fn(chunk) for each pre-split chunk, mirroring presplit() in prepare_tokens.py
local function presplit(s, fn)
  local i, n, byte = 1, #s, string.byte
  while i <= n do
    local c, j = byte(s, i), nil
    local nc = i < n and CLS[byte(s, i + 1)] or 0
    if c == 32 and i < n and nc ~= 0 then
      j = i + 2
      while j <= n and CLS[byte(s, j)] == nc do j = j + 1 end
    elseif CLS[c] ~= 0 then
      local k = CLS[c]; j = i + 1
      while j <= n and CLS[byte(s, j)] == k do j = j + 1 end
    else
      j = i + 1
      while j <= n and CLS[byte(s, j)] == 0 do j = j + 1 end
      if j <= n and byte(s, j - 1) == 32 and j - 1 > i then j = j - 1 end
    end
    fn(s:sub(i, j - 1))
    i = j
  end
end

local K = 65536
local function learn(text, n_merges)
  local count, words = {}, {}
  presplit(text, function(w)
    local c = count[w]
    if c then count[w] = c + 1 else count[w] = 1; words[#words + 1] = w end
  end)
  local seqs, counts = {}, {}
  local pc, where = {}, {}
  for wi, w in ipairs(words) do
    local ids = { w:byte(1, -1) }
    seqs[wi], counts[wi] = ids, count[w]
    for i = 1, #ids - 1 do
      local p = ids[i] * K + ids[i + 1]
      pc[p] = (pc[p] or 0) + counts[wi]
      local s = where[p]; if not s then s = {}; where[p] = s end
      s[wi] = true
    end
  end
  local merges = {}
  for m = 0, n_merges - 1 do
    local best, bc = nil, -1
    for p, c in pairs(pc) do
      if c > bc or (c == bc and p < best) then best, bc = p, c end
    end
    if not best then break end
    local a, b, new = math.floor(best / K), best % K, 256 + m
    merges[#merges + 1] = { a, b }
    local list = {}
    for wi in pairs(where[best]) do list[#list + 1] = wi end
    for _, wi in ipairs(list) do
      local ids, c = seqs[wi], counts[wi]
      for i = 1, #ids - 1 do
        local p = ids[i] * K + ids[i + 1]
        local v = pc[p] - c
        if v <= 0 then pc[p] = nil else pc[p] = v end
        if where[p] then where[p][wi] = nil end
      end
      local o, i = {}, 1
      while i <= #ids do
        if i < #ids and ids[i] == a and ids[i + 1] == b then o[#o + 1] = new; i = i + 2
        else o[#o + 1] = ids[i]; i = i + 1 end
      end
      seqs[wi] = o
      for i = 1, #o - 1 do
        local p = o[i] * K + o[i + 1]
        pc[p] = (pc[p] or 0) + c
        local s = where[p]; if not s then s = {}; where[p] = s end
        s[wi] = true
      end
    end
    where[best] = nil
  end
  return merges
end

local merges = learn(train, vsize - 256)
local rank = {}
for r, m in ipairs(merges) do rank[m[1] * K + m[2]] = r end
local cache = {}
local function enc_chunk(w)
  local hit = cache[w]; if hit then return hit end
  local ids = { w:byte(1, -1) }
  while #ids > 1 do
    local br = nil
    for i = 1, #ids - 1 do
      local r = rank[ids[i] * K + ids[i + 1]]
      if r and (not br or r < br) then br = r end
    end
    if not br then break end
    local a, b = merges[br][1], merges[br][2]
    local o, i = {}, 1
    while i <= #ids do
      if i < #ids and ids[i] == a and ids[i + 1] == b then o[#o + 1] = 255 + br; i = i + 2
      else o[#o + 1] = ids[i]; i = i + 1 end
    end
    ids = o
  end
  cache[w] = ids
  return ids
end

local function u16(x) return string.char(x % 256, math.floor(x / 256) % 256) end
local function u32(x) return u16(x % K) .. u16(math.floor(x / K)) end
local function u64(x) return u32(x % 4294967296) .. u32(math.floor(x / 4294967296)) end

local parts, nt, ntrain = {}, 0, 0
local buf = {}
local function encode(s)
  presplit(s, function(w)
    for _, t in ipairs(enc_chunk(w)) do
      buf[#buf + 1] = u16(t); nt = nt + 1
      if #buf >= 65536 then parts[#parts + 1] = table.concat(buf); buf = {} end
    end
  end)
end
encode(train); ntrain = nt
encode(held)
parts[#parts + 1] = table.concat(buf)

local V = 256 + #merges
f = assert(io.open(out .. "/tokens.bin", "wb"))
f:write("KWTOKS1\0", u32(V), u32(0), u64(nt), u64(ntrain))
for _, p in ipairs(parts) do f:write(p) end
f:close()

local vocab = {}
for i = 0, 255 do vocab[i] = string.char(i) end
for r, m in ipairs(merges) do vocab[255 + r] = vocab[m[1]] .. vocab[m[2]] end
f = assert(io.open(out .. "/vocab.bin", "wb"))
f:write("KWVOCAB1", u32(V), u32(#merges))
for i = 0, V - 1 do f:write(u16(#vocab[i]), vocab[i]) end
for _, m in ipairs(merges) do f:write(u16(m[1]), u16(m[2])) end
f:close()
print(string.format("vocab %d merges %d tokens %d train_tokens %d", V, #merges, nt, ntrain))
