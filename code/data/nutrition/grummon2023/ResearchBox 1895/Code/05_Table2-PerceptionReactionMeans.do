********* Effects of FOPLs on Perceptions and Reactions - Means and SDs *********

/**************
This do file:
-Calculates means and SDs of secondary psychological outcomes. These are reported in Table 2.
-Runs the do file to calculate ADEs for secondary outcomes; this do file calculates the corrected p-values that are reported in Table 2 in the mean text
-Outputs means, SDs, and p-values for differences from control to Excel. This excel file forms the basis of  Table 2.

**************/

*Bring in the cleaned data (result of DataPrep.do)
cd "$Data"
use VLOData_clean.dta, clear

*Run ImpactPerceptionsReactions do file to calculate p values for comparison to control
do "$Code/10_eTable8-PerceptionReactionADEs.do"

**Calculate means and SDs for continuous variables and store in matrices
foreach outcome in $continvars  {
	foreach cond in 1 2 3 4 5 {
		sum `outcome' if foplcond==`cond'
		mat `outcome'`cond'mean=r(mean)
		mat `outcome'`cond'sd = r(sd)
	}
}

*And calculate %s and n's for binary variables; use same matrix names (mean, sd) as above for ease of outputting these results below
foreach outcome in $binvars {
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

*Output Results to Excel for use in Table 3
capture rm "$Results/PerceptionReactionMeans_SDs_ps.xlsx"
putexcel set "$Results/PerceptionReactionMeans_SDs_ps.xlsx", replace

putexcel A2=("Label Reactions")

*Column headings
putexcel B1=("Calorie")
putexcel B2=("Mean or n")
putexcel C2=("SD or %")

	local b=4 //start Green results in column D
	local c=5
	local d=6
	
	foreach condition in Green TLL PACE Warn {
		local col1: word `b' of `c(alpha)'
		local col2: word `c' of `c(alpha)'
		local col3: word `d' of `c(alpha)'
		
		putexcel `col1'1=("`condition'")
		putexcel `col1'2=("Mean or n")
		putexcel `col2'2=("SD or %")
		putexcel `col3'2=("diff vs. control adj. p")	
		
		local b=`b'+3
		local c=`c'+3
		local d=`d'+3
	}

*Add Means and SDs for each set of outcomes
local row=3

foreach list in labelrxnsordered healthfulordered stigmaordered poslabelordered {
	putexcel A`row'=("`list'")
	local ++row
	foreach outcome in $`list' {
		putexcel A`row'=("`outcome'")
			
	foreach cond in 1 {
		putexcel B`row'=matrix(`outcome'`cond'mean), nformat(0.0)
		putexcel C`row'=matrix(`outcome'`cond'sd), nformat("(0.0)")
	}
		
	*Now add Green, TLL, PACE, and Warn results 
	local b=4 //start Green results in column D
	local c=5
	local d=6
	
	foreach cond in 2 3 4 5 {
		local col1: word `b' of `c(alpha)'
		local col2: word `c' of `c(alpha)'
		local col3: word `d' of `c(alpha)'
		
		putexcel `col1'`row'=matrix(`outcome'`cond'mean), nformat(0.0)
		putexcel `col2'`row'=matrix(`outcome'`cond'sd), nformat("(0.0)")
		putexcel `col3'`row'=matrix(`outcome'`cond'adjp), nformat(.000)
		
		*Replace pvalues <0.001 with "<0.001" or >0.99 if==1.00
		local adjp = `outcome'`cond'adjp[1,1]
		if `adjp'<0.001 {
			putexcel `col3'`row'=("<.001")
		}
		local adjp = `outcome'`cond'adjp[1,1]
		if `adjp'==1.0 {
			putexcel `col3'`row'=(">.99")
		}
		
		*Format sig digits of p-values per journal policy
		local adjp = `outcome'`cond'adjp[1,1]
		if `adjp'>=0.01 & `adjp'<1.0 {
			putexcel `col3'`row', overwritefmt nformat(.00)
		}
			
		local b=`b'+3
		local c=`c'+3
		local d=`d'+3	
	}
	local ++row 
	}
}

*fix formatting for binary variables
foreach column in B D G J M {
	foreach row in 4 10 11 27 {
		putexcel `column'`row', overwritefmt nformat(#,###)
	}
}

foreach column in C E H K N {
	foreach row in 4 10 11 27 {
		putexcel `column'`row', overwritefmt nformat("(##)")
	}
}



	
