************Means and %s of Selection Outcomes for Figure 3************

/**************
This do file calculates means of selection outcomes for beverages and for snacks and outputs results to Excel tables. These results can be used to re-create Figure 3.
**************/

*Bring in the cleaned data (result of DataPrep.do)
cd "$Data"
use VLOData_clean.dta, clear

**Estimate means and SEs for each continuous selection outcome for both beverages and snacks
foreach outcome in item_cal_bev item_sug_bev item_sodium_bev  item_cal_snack item_sug_snack item_satfat_snack item_sodium_snack  {
	foreach cond in 1 2 3 4 5 {
		mean `outcome' if foplcond==`cond'
		mat rtable=r(table)
		mat `outcome'`cond'mean=rtable[1,1]
		mat `outcome'`cond'se = rtable[2,1]	
		mat `outcome'`cond'll = rtable[5,1]
		mat `outcome'`cond'ul = rtable[6,1]
	}
}

*Estimate %s and n's for each categorical selection outcome for both beverages and snacks
foreach outcome in redbev redsnack yellowbev yellowsnack greenbev greensnack  {
	foreach cond in 1 2 3 4 5 {
		prop `outcome' if foplcond==`cond'
		mat rtable=r(table)
		mat `outcome'`cond'mean=rtable[1,2]
		mat `outcome'`cond'se = rtable[2,2]	
		mat `outcome'`cond'll = rtable[5,2]
		mat `outcome'`cond'ul = rtable[6,2]
	}
}

*Output Results to Excel
capture rm "$Results/SelectionMeans.xlsx"
putexcel set "$Results/SelectionMeans.xlsx", replace

putexcel A2=("Beverages")

local b=2
local c=3
local d=4
local e=5

foreach condition in Calorie Green TLL PACE Warn {
	local col1: word `b' of `c(alpha)'
	local col2: word `c' of `c(alpha)'
	local col3: word `d' of `c(alpha)'
	local col4: word `e' of `c(alpha)'
	
	putexcel `col1'1=("`condition'")
	putexcel `col1'2=("Mean or %")
	putexcel `col2'2=("SE")
	putexcel `col3'2=("LL")
	putexcel `col4'2=("UL")
	
	local b=`b'+4
	local c=`c'+4
	local d=`d'+4
	local e=`e'+4
}


local row=3
foreach outcome in item_cal_bev item_sug_bev item_sodium_bev greenbev yellowbev redbev {
	putexcel A`row'=("`outcome'")
	local b=2
	local c=3
	local d=4
	local e=5

	foreach cond in 1 2 3 4 5 {
		local col1: word `b' of `c(alpha)'
		local col2: word `c' of `c(alpha)'
		local col3: word `d' of `c(alpha)'
		local col4: word `e' of `c(alpha)'
		
		putexcel `col1'`row'=matrix(`outcome'`cond'mean), nformat(0.00)
		putexcel `col2'`row'=matrix(`outcome'`cond'se), nformat(0.00)
		putexcel `col3'`row'=matrix(`outcome'`cond'll), nformat(0.00)
		putexcel `col4'`row'=matrix(`outcome'`cond'ul), nformat (0.00)
		
		local b=`b'+4
		local c=`c'+4
		local d=`d'+4
		local e=`e'+4		
	}
local ++row 
}
	
putexcel A`row'=("Snacks")
local ++row 
foreach outcome in item_cal_snack item_sug_snack item_sodium_snack item_satfat_snack greensnack yellowsnack redsnack {
	putexcel A`row'=("`outcome'")
	local b=2
	local c=3
	local d=4
	local e=5

	foreach cond in 1 2 3 4 5 {
		local col1: word `b' of `c(alpha)'
		local col2: word `c' of `c(alpha)'
		local col3: word `d' of `c(alpha)'
		local col4: word `e' of `c(alpha)'
		
		putexcel `col1'`row'=matrix(`outcome'`cond'mean), nformat(0.00)
		putexcel `col2'`row'=matrix(`outcome'`cond'se), nformat(0.00)
		putexcel `col3'`row'=matrix(`outcome'`cond'll), nformat(0.00)
		putexcel `col4'`row'=matrix(`outcome'`cond'ul), nformat (0.00) 
		
		local b=`b'+4
		local c=`c'+4
		local d=`d'+4
		local e=`e'+4		
	}
local ++row 
}


