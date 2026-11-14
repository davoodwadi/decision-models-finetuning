function Table(tbl)
  tbl.attr = tbl.attr or pandoc.Attr()
  tbl.attr.attributes['custom-style'] = 'APATable'
  for _, row in ipairs(tbl.head.rows) do
    for _, cell in ipairs(row.cells) do
      if cell.col_span > 1 then
        cell.alignment = pandoc.AlignCenter
      end
    end
  end
  for _, body in ipairs(tbl.bodies) do
    for _, row in ipairs(body.body) do
      for _, cell in ipairs(row.cells) do
        if cell.col_span > 1 then
          cell.alignment = pandoc.AlignCenter
        end
      end
    end
  end
  return tbl
end
