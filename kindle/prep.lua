-- prep.lua OUTDIR EPUB...   Lua 5.1, runs on the Kindle (busybox unzip) and on the Mac.
-- Pulls the reading text out of DRM-free epubs in spine order and writes OUTDIR/text.txt (plain ASCII)
-- and OUTDIR/split.txt (n_train). Each book is cut at a paragraph boundary near 95%: all books' first
-- parts come first (training), then all books' last parts (held-out), so every book is in both sets.

local out = arg[1]
local books = {}
for i = 2, #arg do books[#books + 1] = arg[i] end
table.sort(books)

local function sh_quote(s) return "'" .. s:gsub("'", "'\\''") .. "'" end

local function unzip(epub, member)
  local f = assert(io.popen("unzip -p " .. sh_quote(epub) .. " " .. sh_quote(member) .. " 2>/dev/null"))
  local s = f:read("*a"); f:close(); return s
end

local function urldecode(s)
  return (s:gsub("%%(%x%x)", function(h) return string.char(tonumber(h, 16)) end))
end

local function dirname(p) return p:match("^(.*)/[^/]*$") or "" end

local function normpath(p)
  local parts = {}
  for seg in p:gmatch("[^/]+") do
    if seg == ".." then parts[#parts] = nil elseif seg ~= "." then parts[#parts + 1] = seg end
  end
  return table.concat(parts, "/")
end

-- UTF-8 code points to ASCII; anything unlisted is dropped.
local MAP = {
  [0x2018] = "'", [0x2019] = "'", [0x201C] = '"', [0x201D] = '"', [0x2014] = "--", [0x2013] = "-",
  [0x2026] = "...", [0xA0] = " ", [0x2002] = " ", [0x2003] = " ", [0x2009] = " ", [0x200A] = " ",
  [0xE0] = "a", [0xE1] = "a", [0xE2] = "a", [0xE4] = "a", [0xE7] = "c", [0xE8] = "e", [0xE9] = "e",
  [0xEA] = "e", [0xEB] = "e", [0xED] = "i", [0xEE] = "i", [0xEF] = "i", [0xF1] = "n", [0xF3] = "o",
  [0xF4] = "o", [0xF6] = "o", [0xFA] = "u", [0xFB] = "u", [0xFC] = "u", [0xC9] = "E", [0xB4] = "'",
}

local function cp_ascii(cp)
  if cp < 128 then return string.char(cp) end
  return MAP[cp] or ""
end

local function to_ascii(s)
  return (s:gsub("[\192-\247][\128-\191]*", function(c)
    local b = c:byte(1); local cp
    if b < 224 then cp = (b % 32) * 64 + (c:byte(2) or 128) % 64
    elseif b < 240 then cp = ((b % 16) * 64 + (c:byte(2) or 128) % 64) * 64 + (c:byte(3) or 128) % 64
    else cp = 0x10000 end
    return cp_ascii(cp)
  end):gsub("[\128-\255]", ""))
end

local ENT = { amp = "&", lt = "<", gt = ">", quot = '"', apos = "'", nbsp = " ", mdash = "--", ndash = "-",
  hellip = "...", rsquo = "'", lsquo = "'", rdquo = '"', ldquo = '"' }

local BLOCK = { p = 1, div = 1, blockquote = 1, li = 1, dd = 1, dt = 1, tr = 1, h1 = 1, h2 = 1, h3 = 1,
  h4 = 1, h5 = 1, h6 = 1, section = 1, table = 1 }

local function html_text(s)
  s = s:gsub("<[Hh][Ee][Aa][Dd].-</[Hh][Ee][Aa][Dd]>", "")
  s = s:gsub("<[Ss][Cc][Rr][Ii][Pp][Tt].-</[Ss][Cc][Rr][Ii][Pp][Tt]>", "")
  s = s:gsub("<[Ss][Tt][Yy][Ll][Ee].-</[Ss][Tt][Yy][Ll][Ee]>", "")
  s = s:gsub("</(%w+)[^>]*>", function(t) if BLOCK[t:lower()] then return "\n\n" end end)
  s = s:gsub("<[Bb][Rr][^>]*>", "\n\n")
  s = s:gsub("<[^>]*>", "")
  s = s:gsub("&#[xX](%x+);", function(h) return cp_ascii(tonumber(h, 16)) end)
  s = s:gsub("&#(%d+);", function(d) return cp_ascii(tonumber(d)) end)
  s = s:gsub("&(%a+);", function(n) return ENT[n] or "" end)
  return s
end

local function paragraphs(s, acc)
  s = to_ascii(s):gsub("\r", "")
  for block in (s .. "\n\n"):gmatch("(.-)\n%s*\n") do
    local p = block:gsub("%s+", " "):gsub("^ ", ""):gsub(" $", "")
    local letters = select(2, p:gsub("%a", ""))
    if #p > 0 and letters * 2 >= #p then acc[#acc + 1] = p end
  end
end

local function book_paragraphs(epub)
  local container = unzip(epub, "META-INF/container.xml")
  local opf = assert(container:match('full%-path="([^"]+)"'), "no opf in " .. epub)
  local o = unzip(epub, opf)
  local base = dirname(opf)
  local man = {}
  for tag in o:gmatch("<[%w:]*item%s[^>]*>") do
    local id, href = tag:match('%sid="([^"]+)"'), tag:match('%shref="([^"]+)"')
    if id and href then man[id] = href end
  end
  local paras = {}
  for tag in o:gmatch("<[%w:]*itemref%s[^>]*>") do
    local href = man[tag:match('idref="([^"]+)"') or ""]
    if href then
      local path = normpath((base ~= "" and base .. "/" or "") .. urldecode(href:gsub("#.*$", "")))
      local part = {}
      paragraphs(html_text(unzip(epub, path)), part)
      local n = 0
      for _, p in ipairs(part) do n = n + #p end
      if n >= 500 then for _, p in ipairs(part) do paras[#paras + 1] = p end end
    end
  end
  return paras
end

local head, tail = {}, {}
local manifest = {}
for _, epub in ipairs(books) do
  local paras = book_paragraphs(epub)
  local total = 0
  for _, p in ipairs(paras) do total = total + #p + 2 end
  local acc, cut = 0, #paras
  for i, p in ipairs(paras) do
    acc = acc + #p + 2
    if acc >= total * 0.95 then cut = i; break end
  end
  for i = 1, cut do head[#head + 1] = paras[i] end
  for i = cut + 1, #paras do tail[#tail + 1] = paras[i] end
  manifest[#manifest + 1] = string.format("%d\t%d\t%s", total, #paras, epub:match("[^/]+$"))
end

local train = table.concat(head, "\n\n") .. "\n\n"
local held = table.concat(tail, "\n\n") .. "\n"
local f = assert(io.open(out .. "/text.txt", "wb")); f:write(train, held); f:close()
f = assert(io.open(out .. "/split.txt", "wb")); f:write(string.format("%d\n", #train)); f:close()
f = assert(io.open(out .. "/books.tsv", "wb")); f:write(table.concat(manifest, "\n"), "\n"); f:close()
print(string.format("books %d train %d held %d", #books, #train, #held))
