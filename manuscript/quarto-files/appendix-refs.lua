-- appendix-refs.lua
local appendix_letters = {}
local counter = 0
local letters = {'A','B','C','D','E','F','G','H','I','J','K','L'}

function Pandoc(doc)
  -- First pass: scan headers
  doc.blocks:walk({
    Header = function(el)
      if el.identifier:match("^sec%-appendix") or el.identifier:match("^app%-") then
        counter = counter + 1
        appendix_letters[el.identifier] = letters[counter]
      end
    end
  })

  -- Second pass: replace citations
  doc.blocks = doc.blocks:walk({
    Cite = function(el)
      if #el.citations == 1 then
        local id = el.citations[1].id
        if appendix_letters[id] then
          return pandoc.Str("Web Appendix " .. appendix_letters[id])
        end
      end

      local letters_found = {}
      local all_appendix = true
      for _, cite in ipairs(el.citations) do
        if appendix_letters[cite.id] then
          table.insert(letters_found, appendix_letters[cite.id])
        else
          all_appendix = false
          break
        end
      end

      if all_appendix and #letters_found > 1 then
        local joined
        if #letters_found == 2 then
          joined = letters_found[1] .. " and " .. letters_found[2]
        else
          joined = table.concat(letters_found, ", ", 1, #letters_found - 1)
                  .. ", and " .. letters_found[#letters_found]
        end
        return pandoc.Str("Web Appendices " .. joined)
      end

      return el
    end
  })

  return doc
end
