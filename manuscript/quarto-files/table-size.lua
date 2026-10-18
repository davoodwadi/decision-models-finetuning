-- table-size.lua
-- LaTeX only: per-table font size. Add size="scriptsize" (or tiny, footnotesize,
-- small, normalsize) to a table div, e.g.
--   ::: {#apptbl-model-specifications size="scriptsize"}
-- and the table is typeset at that size; other tables keep the default
-- (\footnotesize, see preamble.tex). Puts \nexttablesize{...} at the start of the
-- float's content, which the next longtable consumes (breakable-apptbl.lua keeps
-- it in front of the table). Runs at pre-quarto, where Quarto has already parsed
-- the div into a FloatRefTarget. docx output is unchanged.

local sizes = {
  tiny = true, scriptsize = true, footnotesize = true, small = true, normalsize = true,
}

return {
  {
    FloatRefTarget = function(float)
      local size = float.attributes and float.attributes["size"]
      if not size then
        return nil
      end
      float.attributes["size"] = nil
      if not quarto.doc.is_format("latex") then
        return float
      end
      if not sizes[size] then
        io.stderr:write("table-size.lua: unknown size '" .. size .. "' on #" .. float.identifier .. "\n")
        return float
      end
      local blocks = float.content
      if pandoc.utils.type(blocks) ~= "Blocks" then
        blocks = pandoc.Blocks({ blocks })
      end
      blocks:insert(1, pandoc.RawBlock("latex", "\\nexttablesize{\\" .. size .. "}"))
      float.content = blocks
      return float
    end,
  },
}
