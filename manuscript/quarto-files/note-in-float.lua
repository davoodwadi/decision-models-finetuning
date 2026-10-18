-- note-in-float.lua
-- LaTeX only: move a "Table Note" div that directly follows an appendix float
-- (appfig/apptbl) inside the float, so the note always stays with its
-- figure/table. The qmd keeps the note outside the float div, so docx output is
-- unchanged. Must run at post-render, after Quarto builds the float.
--
-- appfig uses a bottom caption, which the float package always typesets last, so
-- figure notes are wrapped in \floatnote{...} (preamble.tex) to print after the
-- caption. apptbl floats are normally unwrapped first by breakable-apptbl.lua.

local float_envs = { appfig = true, apptbl = true }

local function float_env(block)
  if block.t ~= "Div" or not block.classes:includes("quarto-scaffold") then
    return nil
  end
  local last = block.content[#block.content]
  if not last or last.t ~= "RawBlock" then
    return nil
  end
  local env = last.text:match("^\\end{(%w+)}%%?%s*$")
  if env and float_envs[env] then
    return env
  end
  return nil
end

local function is_note(block)
  return block.t == "Div" and block.attributes["custom-style"] == "Table Note"
end

local function note_blocks(env, note)
  if env ~= "appfig" then
    return note.content
  end
  -- fragment only (no template), keeping the document's citation method
  local opts = { cite_method = PANDOC_WRITER_OPTIONS.cite_method }
  local tex = pandoc.write(pandoc.Pandoc(note.content), "latex", opts)
  tex = tex:gsub("^%s+", ""):gsub("%s+$", "")
  return { pandoc.RawBlock("latex", "\\floatnote{" .. tex .. "}") }
end

function Blocks(blocks)
  if not FORMAT:match("latex") then
    return nil
  end
  local out = pandoc.Blocks({})
  local i = 1
  while i <= #blocks do
    local b = blocks[i]
    local nxt = blocks[i + 1]
    local env = float_env(b)
    if nxt and env and is_note(nxt) then
      local end_block = b.content:remove()
      b.content:extend(note_blocks(env, nxt))
      b.content:insert(end_block)
      out:insert(b)
      i = i + 2
    else
      out:insert(b)
      i = i + 1
    end
  end
  return out
end
