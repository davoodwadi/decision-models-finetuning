-- breakable-apptbl.lua
-- LaTeX only: typeset each appendix table float (apptbl) as a plain longtable.
-- A longtable wrapped in a float cannot break across pages and loses its
-- \endlastfoot bottom rule (the last footer is never output inside a float box).
-- The caption moves into the longtable head and still numbers as "Table W#"
-- through \LTcaptype. The qmd is unchanged, so docx output is unaffected.
-- Must run at post-render, after Quarto builds the float.

local function is_raw(block, pattern)
  return block ~= nil and block.t == "RawBlock" and block.text:match(pattern) ~= nil
end

local function find_longtable(block)
  if is_raw(block, "\\begin{longtable%*}") then
    return block.text
  end
  local found
  if block.t == "Div" then
    block:walk({
      RawBlock = function(rb)
        if rb.text:match("\\begin{longtable%*}") then
          found = rb.text
        end
      end,
    })
  end
  return found
end

-- Raw commands that must stay in front of the table, e.g. \nexttablesize from table-size.lua
local function find_prefixes(block, prefixes)
  local function keep(rb)
    if rb.text:match("^\\nexttablesize") then
      prefixes[#prefixes + 1] = rb.text
    end
  end
  if block.t == "RawBlock" then
    keep(block)
  elseif block.t == "Div" then
    block:walk({ RawBlock = keep })
  end
end

local function caption_tex(block)
  if block.t ~= "Plain" and block.t ~= "Para" then
    return nil
  end
  -- fragment only (no template), keeping the document's citation method
  local opts = { cite_method = PANDOC_WRITER_OPTIONS.cite_method }
  local tex = pandoc.write(pandoc.Pandoc({ block }), "latex", opts)
  return tex:match("\\caption%b{}")
end

-- Rebuild Quarto's uncaptioned longtable* as a captioned longtable:
-- caption + header as \endfirsthead, header repeated as \endhead.
local function to_breakable(tbl, caption)
  local spec, body = tbl:match("\\begin{longtable%*}(%[[^%]]*%]%b{})(.*)$")
  if not spec then
    return nil
  end
  local head, rest = body:match("^(.-)\\endhead(.*)$")
  if not head then
    return nil
  end
  rest = rest:gsub("\\end{longtable%*}", "\\end{longtable}")
  if not rest:match("\\end{longtable}%s*}%s*$") then
    return nil
  end
  return "{\\def\\LTcaptype{apptbl}\n\\begin{longtable}" .. spec .. "\n"
    .. caption .. "\\tabularnewline" .. head
    .. "\\endfirsthead" .. head
    .. "\\endhead" .. rest
end

function Div(div)
  if not FORMAT:match("latex") or not div.classes:includes("quarto-scaffold") then
    return nil
  end
  local content = div.content
  if not is_raw(content[1], "^\\begin{apptbl}") or not is_raw(content[#content], "^\\end{apptbl}") then
    return nil
  end

  local caption, tbl
  local prefixes = {}
  local extras = pandoc.Blocks({})
  for i = 2, #content - 1 do
    local b = content[i]
    find_prefixes(b, prefixes)
    local lt = find_longtable(b)
    if lt then
      if tbl then
        return nil -- more than one table: leave the float alone
      end
      tbl = lt
    elseif not caption and caption_tex(b) then
      caption = caption_tex(b)
    elseif pandoc.utils.stringify(b) ~= "" then
      extras:insert(b) -- e.g. a note already moved inside the float
    end
  end
  if not (caption and tbl) then
    return nil
  end

  local new = to_breakable(tbl, caption)
  if not new then
    return nil
  end
  if #prefixes > 0 then
    new = table.concat(prefixes, "\n") .. "\n" .. new
  end
  local out = pandoc.Blocks({ pandoc.RawBlock("latex", new) })
  out:extend(extras)
  return out
end
