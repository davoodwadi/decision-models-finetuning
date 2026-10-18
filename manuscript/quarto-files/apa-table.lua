function Table(tbl)
  tbl.attr = tbl.attr or pandoc.Attr()
  tbl.attr.attributes['custom-style'] = 'APATable'
  return tbl
end
