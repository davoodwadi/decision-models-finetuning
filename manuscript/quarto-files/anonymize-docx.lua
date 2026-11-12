-- anonymize-docx.lua
-- Double-anonymous review: drop author and affiliation metadata from the
-- Word output only. The qmd keeps its author block; PDF/HTML are unaffected.

local keys = {
  "author", "authors", "affiliations", "institute",
  "by-author", "by-affiliation", "email",
}

function Meta(meta)
  if not FORMAT:match("docx") then
    return nil
  end
  for _, k in ipairs(keys) do
    meta[k] = nil
  end
  return meta
end
