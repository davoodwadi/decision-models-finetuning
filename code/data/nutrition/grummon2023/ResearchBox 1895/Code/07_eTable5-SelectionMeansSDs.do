**********Means and SDs of Selection Outcomes**********

/**************
This do file calculates means of selection outcomes for beverages and for snacks and outputs results to Excel tables.  These results form eTable 5. 
**************/

*Bring in the cleaned data (result of DataPrep.do)
cd "$Data"
use VLOData_clean.dta, clear

**Calculate means and SDs for all continuous selection outcomes and store in matrices
foreach outcome in item_cal_bev item_sug_bev item_sodium_bev  item_cal_snack item_sug_snack item_satfat_snack item_sodium_snack  {
	foreach cond in 1 2 3 4 5 {
		sum `outcome' if foplcond==`cond'
		mat `outcome'`cond'mean=r(mean)
		mat `outcome'`cond'sd = r(sd)
	}
}

**Calculate proportions and n's for all categorical selection outcomes and store in matrices
foreach outcome in redbev redsnack yellowbev yellowsnack greenbev greensnack {
	foreach cond in 1 2 3 4 5 {
		prop `outcome' if foplcond==`cond'
		mat rtable=r(table)
		*Save % as "SD" so that we output it the second column in the loops below
		**Multiply by 100 so that we display as a % not a proportion
		mat `outcome'`cond'sd=100*rtable[1,2]
		
		count if foplcond==`cond' & `outcome'==1
		*Save N as "mean" so that we output it to the first column in the loops below
		mat `outcome'`cond'mean = r(N)
	}
}

*Output means/SDs and proportions/n's to Excel
capture rm "$Results/SelectionMeans_SDs.xlsx"
putexcel set "$Results/SelectionMeans_SDs.xlsx", replace

putexcel A2=("Beverages")

*Add control mean/SD
putexcel B1=("Calorie")
putexcel B2=("Mean or n")
putexcel C2=("SD or %")

local row=3
foreach outcome in item_cal_bev item_sug_bev item_sodium_bev {
	putexcel A`row'=("`outcome'")
			
	foreach cond in 1 {
		putexcel B`row'=matrix(`outcome'`cond'mean), nformat(0.0)
		putexcel C`row'=matrix(`outcome'`cond'sd), nformat("(0.0)")
	}
local ++row 
}

foreach outcome in greenbev yellowbev redbev  {
	putexcel A`row'=("`outcome'")
	
	foreach cond in 1 {
		putexcel B`row'=matrix(`outcome'`cond'mean), nformat(##)
		putexcel C`row'=matrix(`outcome'`cond'sd), nformat("(##)")
	}
local ++row 
}
		
putexcel A`row'=("Snacks")
local ++row 
foreach outcome in item_cal_snack item_sug_snack item_sodium_snack item_satfat_snack {
	putexcel A`row'=("`outcome'")

	foreach cond in 1 {
		putexcel B`row'=matrix(`outcome'`cond'mean), nformat(0.0)
		putexcel C`row'=matrix(`outcome'`cond'sd), nformat("(0.0)")
	}
local ++row 
}
	
foreach outcome in greensnack yellowsnack redsnack  {
	putexcel A`row'=("`outcome'")
	
	foreach cond in 1 {
		putexcel B`row'=matrix(`outcome'`cond'mean), nformat(####)
		putexcel C`row'=matrix(`outcome'`cond'sd), nformat("(##)")
	}
local ++row 
}

*Now add Green, TLL, PACE, and Warn results 
local b=4 //start Green results in column D
local c=5

foreach condition in Green TLL PACE Warn {
	local col1: word `b' of `c(alpha)'
	local col2: word `c' of `c(alpha)'
	
	putexcel `col1'1=("`condition'")
	putexcel `col1'2=("Mean or N")
	putexcel `col2'2=("SD or %")
	
	local b=`b'+2
	local c=`c'+2

}

local row=3
foreach outcome in item_cal_bev item_sug_bev item_sodium_bev {
	putexcel A`row'=("`outcome'")
		local b=4 //start Green results in column D
		local c=5

	foreach cond in 2 3 4 5 {
		local col1: word `b' of `c(alpha)'
		local col2: word `c' of `c(alpha)'
		
		putexcel `col1'`row'=matrix(`outcome'`cond'mean), nformat(0.0)
		putexcel `col2'`row'=matrix(`outcome'`cond'sd), nformat("(0.0)")

		local b=`b'+2
		local c=`c'+2
		
	}
local ++row 
}
	
foreach outcome in greenbev {
	putexcel A`row'=("`outcome'")
		local b=4 //start Green results in column D
		local c=5
	
	foreach cond in 2 3 4 5 {
		local col1: word `b' of `c(alpha)'
		local col2: word `c' of `c(alpha)'
		
		putexcel `col1'`row'=matrix(`outcome'`cond'mean), nformat(####)
		putexcel `col2'`row'=matrix(`outcome'`cond'sd), nformat("(##)")

		local b=`b'+2
		local c=`c'+2
			
	}
local ++row 
}

	
foreach outcome in yellowbev {
	putexcel A`row'=("`outcome'")
		local b=4 //start Green results in column D
		local c=5
	
	foreach cond in 2 3 4 5 {
		local col1: word `b' of `c(alpha)'
		local col2: word `c' of `c(alpha)'
			
		putexcel `col1'`row'=matrix(`outcome'`cond'mean), nformat(####)
		putexcel `col2'`row'=matrix(`outcome'`cond'sd), nformat("(##)")
	
		local b=`b'+2
		local c=`c'+2

	}
local ++row 
}

	
foreach outcome in redbev {
	putexcel A`row'=("`outcome'")
		local b=4 //start Green results in column D
		local c=5

	foreach cond in 2 3 4 5 {
		local col1: word `b' of `c(alpha)'
		local col2: word `c' of `c(alpha)'
				
		putexcel `col1'`row'=matrix(`outcome'`cond'mean), nformat(####)
		putexcel `col2'`row'=matrix(`outcome'`cond'sd), nformat("(##)")
		
		local b=`b'+2
		local c=`c'+2

	}
local ++row 
}


putexcel A`row'=("Snacks")
local ++row 
foreach outcome in item_cal_snack item_sug_snack item_sodium_snack item_satfat_snack  {
	putexcel A`row'=("`outcome'")
		local b=4 //start Green results in column D
		local c=5

	foreach cond in 2 3 4 5 {
		local col1: word `b' of `c(alpha)'
		local col2: word `c' of `c(alpha)'
			
		putexcel `col1'`row'=matrix(`outcome'`cond'mean), nformat(0.0)
		putexcel `col2'`row'=matrix(`outcome'`cond'sd), nformat("(0.0)")
		
		local b=`b'+2
		local c=`c'+2

	}
local ++row 
}
	
foreach outcome in greensnack {
	putexcel A`row'=("`outcome'")
		local b=4 //start Green results in column D
		local c=5

	foreach cond in 2 3 4 5 {
		local col1: word `b' of `c(alpha)'
		local col2: word `c' of `c(alpha)'
		
		putexcel `col1'`row'=matrix(`outcome'`cond'mean), nformat(####)
		putexcel `col2'`row'=matrix(`outcome'`cond'sd), nformat("(##)")
		
		local b=`b'+2
		local c=`c'+2

	}
local ++row 
}

	
foreach outcome in yellowsnack {
	putexcel A`row'=("`outcome'")
		local b=4 //start Green results in column D
		local c=5

	foreach cond in 2 3 4 5 {
		local col1: word `b' of `c(alpha)'
		local col2: word `c' of `c(alpha)'
		
		putexcel `col1'`row'=matrix(`outcome'`cond'mean), nformat(####)
		putexcel `col2'`row'=matrix(`outcome'`cond'sd), nformat("(##)")
		
		local b=`b'+2
		local c=`c'+2

	}
local ++row 
}

	
foreach outcome in redsnack {
	putexcel A`row'=("`outcome'")
		local b=4 //start Green results in column D
		local c=5

	foreach cond in 2 3 4 5 {
		local col1: word `b' of `c(alpha)'
		local col2: word `c' of `c(alpha)'

		
		putexcel `col1'`row'=matrix(`outcome'`cond'mean), nformat(####)
		putexcel `col2'`row'=matrix(`outcome'`cond'sd), nformat("(##)")
	
		local b=`b'+2
		local c=`c'+2

	}
local ++row 
}


